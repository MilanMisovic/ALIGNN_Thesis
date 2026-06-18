"""Atomistic LIne Graph Neural Network.

A prototype crystal line graph network dgl implementation.
"""

from typing import Optional, Tuple, Union
import dgl
import dgl.function as fn
import numpy as np
import torch
from dgl.nn import AvgPooling
from typing import Literal
from torch import nn
from torch.nn import functional as F
from alignn.models.utils import RBFExpansion
from pydantic_settings import BaseSettings


class ALIGNNConfig(BaseSettings):
    """Hyperparameter schema for jarvisdgl.models.alignn."""

    name: Literal["alignn"]
    alignn_layers: int = 4
    gcn_layers: int = 4
    atom_input_features: int = 92
    edge_input_features: int = 80
    triplet_input_features: int = 40
    embedding_features: int = 64
    hidden_features: int = 256
    # fc_layers: int = 1
    # fc_features: int = 64
    output_features: int = 1

    # NOTE
    # added these because they were missing attributes
    graphwise_weight: float = 1.0
    atomwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    additional_output_weight: float = 0.0

    # PORE: number of Zeo++ / hMOF pore descriptor scalars to concatenate
    # after the crystal graph readout.  Set to 0 (default) to disable
    # entirely, the model then behaves exactly as before with no changes
    # to any weights or layer sizes.
    # Set to 4 to use [void_fraction, pld, lcd, surface_area_m2cm3]
    # all columns already present in the hMOF JARVIS dataset.
    # The pore tensors are built by pore_utils.py and passed in via train.py.
    pore_features: int = 0

    # if link == log, apply `exp` to final outputs
    # to constrain predictions to be positive
    link: Literal["identity", "log", "logit"] = "identity"
    zero_inflated: bool = False
    classification: bool = False
    num_classes: int = 2
    extra_features: int = 0

    class Config:
        """Configure model settings behavior."""

        env_prefix = "jv_model"


class EdgeGatedGraphConv(nn.Module):
    """Edge gated graph convolution from arxiv:1711.07553.

    see also arxiv:2003.0098.

    This is similar to CGCNN, but edge features only go into
    the soft attention / edge gating function, and the primary
    node update function is W cat(u, v) + b
    """

    def __init__(
        self, input_features: int, output_features: int, residual: bool = True
    ):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.residual = residual
        # CGCNN-Conv operates on augmented edge features
        # z_ij = cat(v_i, v_j, u_ij)
        # m_ij = σ(z_ij W_f + b_f) ⊙ g_s(z_ij W_s + b_s)
        # coalesce parameters for W_f and W_s
        # but -- split them up along feature dimension
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.bn_edges = nn.BatchNorm1d(output_features)

        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)
        self.bn_nodes = nn.BatchNorm1d(output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.

        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        # instead of concatenating (u || v || e) and applying one weight matrix
        # split the weight matrix into three, apply, then sum
        # see https://docs.dgl.ai/guide/message-efficient.html
        # but split them on feature dimensions to update u, v, e separately
        # m = BatchNorm(Linear(cat(u, v, e)))

        # compute edge updates, equivalent to:
        # Softplus(Linear(u || v || e))
        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        # softmax version seems to perform slightly worse
        # that the sigmoid-gated version
        # compute node updates
        # Linear(u) + edge_gates ⊙ Linear(v)
        # g.edata["gate"] = edge_softmax(g, y)
        # g.ndata["h_dst"] = self.dst_update(node_feats)
        # g.update_all(fn.u_mul_e("h_dst", "gate", "m"), fn.sum("m", "h"))
        # x = self.src_update(node_feats) + g.ndata.pop("h")

        # node and edge updates
        x = F.silu(self.bn_nodes(x))
        y = F.silu(self.bn_edges(m))

        if self.residual:
            x = node_feats + x
            y = edge_feats + y

        return x, y


class ALIGNNConv(nn.Module):
    """Line graph update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        """Set up ALIGNN parameters."""
        super().__init__()
        self.node_update = EdgeGatedGraphConv(in_features, out_features)
        self.edge_update = EdgeGatedGraphConv(out_features, out_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        lg: dgl.DGLGraph,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ):
        """Node and Edge updates for ALIGNN layer.

        x: node input features
        y: edge input features
        z: edge pair input features
        """
        g = g.local_var()
        lg = lg.local_var()
        # Edge-gated graph convolution update on crystal graph
        x, m = self.node_update(g, x, y)

        # Edge-gated graph convolution update on crystal graph
        y, z = self.edge_update(lg, m, z)

        return x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""

    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class ALIGNN(nn.Module):
    """Atomistic Line graph network.

    Chain alternating gated graph convolution updates on crystal graph
    and atomistic line graph.
    """

    def __init__(self, config: ALIGNNConfig = ALIGNNConfig(name="alignn")):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        # print(config)
        self.config = config

        # NOTE
        # testing
        print("ALIGNN INIT RECEIVED CONFIG:")
        print("atom_input_features:", config.atom_input_features)
        print("hidden_features:", config.hidden_features)
        print("edge_input_features:", config.edge_input_features)
        # PORE: also print so you can confirm pore pathway is active.
        print("pore_features:", config.pore_features)

        self.classification = config.classification

        self.atom_embedding = MLPLayer(
            config.atom_input_features, config.hidden_features
        )

        self.edge_embedding = nn.Sequential(
            RBFExpansion(
                vmin=0,
                vmax=8.0,
                bins=config.edge_input_features,
            ),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(
                vmin=-1,
                vmax=1.0,
                bins=config.triplet_input_features,
            ),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )

        self.alignn_layers = nn.ModuleList(
            [
                ALIGNNConv(
                    config.hidden_features,
                    config.hidden_features,
                )
                for idx in range(config.alignn_layers)
            ]
        )
        self.gcn_layers = nn.ModuleList(
            [
                EdgeGatedGraphConv(
                    config.hidden_features, config.hidden_features
                )
                for idx in range(config.gcn_layers)
            ]
        )

        self.readout = AvgPooling()
        self.readout_feat = AvgPooling()

        # PORE: when pore_features > 0 we add a small two-layer MLP that
        # embeds the Zeo++ scalars, then concatenate its output with the
        # pooled crystal embedding h before the final linear layer.
        #
        # pore_mlp maps:  pore_features ->> hidden_features // 4
        #   e.g. for the default hidden_features=256 : 5 → 64
        #
        # The final fc layer then maps:
        #   hidden_features + hidden_features // 4 -> output_features
        #   so 256 + 64 = 320 → 1
        #
        # When pore_features == 0 (default) none of this is created and
        # fc is exactly Linear(hidden_features, output_features) as before.
        if config.pore_features > 0:
            pore_hidden = config.hidden_features // 4   # e.g. 64
            # PORE: use LayerNorm instead of BatchNorm1d inside the pore MLP.
            # MLPLayer uses BatchNorm1d internally, which requires batch_size > 1
            # to compute per-channel statistics during training.  This crashes
            # during the test loop where batch_size=1 (last batch of test set).
            # LayerNorm normalises over the feature dimension instead of the
            # batch dimension, so it works correctly for any batch size >= 1.
            #
            # OLD code! (kept for reference, crashes at test time with batch=1):
            # self.pore_mlp = nn.Sequential(
            #     MLPLayer(config.pore_features, pore_hidden),
            #     MLPLayer(pore_hidden, pore_hidden),
            # )
            self.pore_mlp = nn.Sequential(
                nn.Linear(config.pore_features, pore_hidden),
                nn.LayerNorm(pore_hidden),
                nn.SiLU(),
                nn.Linear(pore_hidden, pore_hidden),
                nn.LayerNorm(pore_hidden),
                nn.SiLU(),
            )
            fc_in = config.hidden_features + pore_hidden
        else:
            fc_in = config.hidden_features


        if self.classification:
            # NOTE: fc_in replaces the hard-coded config.hidden_features here
            # so classification also works when pore features are active.
            self.fc = nn.Linear(fc_in, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            # OLD code (kept for reference):
            # self.fc = nn.Linear(config.hidden_features, config.output_features)
            # PORE: fc_in == config.hidden_features when pore_features == 0,
            # so this is identical to the old line when pore is disabled.
            self.fc = nn.Linear(fc_in, config.output_features)

        if config.extra_features != 0:
            # Credit for extra_features work:
            # Gong et al., https://doi.org/10.48550/arXiv.2208.05039
            self.extra_feature_embedding = MLPLayer(
                config.extra_features, config.extra_features
            )
            self.fc3 = nn.Linear(
                config.hidden_features + config.extra_features,
                config.output_features,
            )
            self.fc1 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )
            self.fc2 = MLPLayer(
                config.extra_features + config.hidden_features,
                config.extra_features + config.hidden_features,
            )

        self.link = None
        self.link_name = config.link
        if config.link == "identity":
            self.link = lambda x: x
        elif config.link == "log":
            self.link = torch.exp
            avg_gap = 0.7  # magic number -- average bandgap in dft_3d
            self.fc.bias.data = torch.tensor(
                np.log(avg_gap), dtype=torch.float
            )
        elif config.link == "logit":
            self.link = torch.sigmoid

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
        pore_feats: Optional[torch.Tensor] = None,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)

        pore_feats: optional Tensor of shape (batch_size, pore_features)
            Precomputed Zeo++ / hMOF pore descriptors built by pore_utils.py
            and passed in by train.py via the forward() helper.
            Only used when config.pore_features > 0; ignored otherwise.
            When pore_features > 0 and this is None, a clear ValueError is
            raised so the mistake is caught immediately.
        """
        if len(self.alignn_layers) > 0:
            # print('features2',features.shape)

            g, lg, lat = g
            lg = lg.local_var()

            # angle features (fixed)
            z = self.angle_embedding(lg.edata.pop("h"))
        if self.config.extra_features != 0:
            features = g.ndata["extra_features"]
            # print('g',g)
            # print('features1',features.shape)
            features = self.extra_feature_embedding(features)

        g = g.local_var()
        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        # print("x1", x, x.shape)
        x = self.atom_embedding(x)
        # print("x2", x, x.shape)

        # initial bond features
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        y = self.edge_embedding(bondlength)

        # ALIGNN updates: update node, edge, triplet features
        for alignn_layer in self.alignn_layers:
            x, y, z = alignn_layer(g, lg, x, y, z)

        # gated GCN updates: update node, edge features
        for gcn_layer in self.gcn_layers:
            x, y = gcn_layer(g, x, y)

        # norm-activation-pool-classify
        h = self.readout(g, x)
        # print("h", h, h.shape)
        # print('features',features.shape)
        if self.config.extra_features != 0:
            h_feat = self.readout_feat(g, features)
            # print('h1',h.shape)
            # print('h_feat',h_feat.shape)
            h = torch.cat((h, h_feat), 1)
            # print('h2',h.shape)

            h = self.fc1(h)

            h = self.fc2(h)

            out = self.fc3(h)
        else:
            # NOTE added this for pores
            # PORE: when pore_features > 0, embed the Zeo++ descriptor vector
            # and concatenate it onto h before the final fc layer.
            # pore_feats is passed in from train.py via forward(net, graphs,
            # pore_feats=pore), it is never defined inside this file, it
            # arrives as a function argument from outside (from the DataLoader
            # batch, extracted by _unpack_batch in train.py).
            #
            # When pore_features == 0 (default) this block is skipped entirely
            # and out = self.fc(h) below is identical to the original code.
            if self.config.pore_features > 0:
                if pore_feats is None:
                    raise ValueError(
                        "config.pore_features > 0 but pore_feats was not "
                        "passed to forward(). It should arrive automatically "
                        "from train.py via forward(net, graphs, pore_feats=pore). "
                        "Check that pore_utils.get_pore_loaders() was used to "
                        "build the DataLoaders and that config.model.pore_features "
                        "matches the number of columns in pore_utils.PORE_COLS."
                    )
                # pore_feats shape: (batch_size, pore_features)  e.g. (4, 5)
                p = self.pore_mlp(pore_feats)   # (batch_size, hidden//4)
                h = torch.cat([h, p], dim=-1)    # (batch_size, hidden + hidden//4)

            out = self.fc(h)

        if self.link:
            out = self.link(out)

        if self.classification:
            # out = torch.round(torch.sigmoid(out))
            out = self.softmax(out)
        return torch.squeeze(out)