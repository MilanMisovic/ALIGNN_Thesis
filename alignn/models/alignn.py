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

    # # NOTE
    # # added these because they were missing attributes
    graphwise_weight: float = 1.0
    atomwise_weight: float = 1.0
    gradwise_weight: float = 1.0
    stresswise_weight: float = 0.0
    additional_output_weight: float = 0.0

    # PORE OPTION A: number of Zeo++ / hMOF pore descriptor scalars to
    # concatenate after the crystal graph readout.  Set to 0 (default) to
    # disable entirely.  Set to 3 to use [void_fraction, pld, lcd].
    # The pore tensors are built by pore_utils.py and passed in via train.py.
    pore_features: int = 0

    # PORE OPTION B: bipartite atom-pore graph GNN branch.
    # When True, a small GNN processes G_pore (the bipartite
    # graph built by bipartite_pore_graph.py) and concatenates the resulting
    # pore embedding with the crystal graph embedding before the output head.
    # Requires pore_graph_dataset.get_pore_graph_loaders() in train.py.
    use_pore_graph: bool = False

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
        #   e.g. for the default hidden_features=256: 5 → 64
        #
        # The final fc layer then maps:
        #   hidden_features + hidden_features // 4 → output_features
        #   so 256 + 64 = 320 ->> 1
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
            # OLD code (kept for reference, crashes at test time with batch=1):
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

        # PORE OPTION B: bipartite pore graph GNN branch
        # One round of cross-message-passing between atom and pore nodes,
        # then mean-pool pore nodes to get a pore_hidden-dim embedding.
        # Architecture:
        #   atom feats (92-dim cgcnn) -> Linear -> pore_hidden
        #   pore feats (2-dim)        -> Linear -> pore_hidden
        #   edge dist  (1-dim)        -> Linear -> pore_hidden  (edge gate)
        #   1x pore->atom message passing
        #   1x atom->pore message passing
        #   mean pool over pore nodes -> (batch, pore_hidden)
        #   LayerNorm + SiLU
        # Concatenated with crystal graph embedding h before fc layer.
        if config.use_pore_graph:
            pore_hidden = config.hidden_features // 4   # 64
            self.pore_atom_emb    = nn.Linear(92,          pore_hidden)
            self.pore_node_emb    = nn.Linear(2,           pore_hidden)
            self.pore_edge_emb    = nn.Linear(1,           pore_hidden)
            self.pore_msg_p2a     = nn.Linear(pore_hidden, pore_hidden)
            self.pore_msg_a2p     = nn.Linear(pore_hidden, pore_hidden)
            self.pore_readout_mlp = nn.Sequential(
                nn.Linear(pore_hidden, pore_hidden),
                nn.LayerNorm(pore_hidden),
                nn.SiLU(),
            )
            # Override fc_in to include pore embedding dimension
            fc_in = config.hidden_features + pore_hidden

        if self.classification:
            # NOTE: fc_in replaces the hard-coded config.hidden_features here
            # so classification also works when pore features are active.
            self.fc = nn.Linear(fc_in, config.num_classes)
            self.softmax = nn.LogSoftmax(dim=1)
        else:
            # fc_in == config.hidden_features when both pore options disabled,
            # so this is identical to the original line in that case.
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

    def _encode_pore_graph(self, G: dgl.DGLHeteroGraph) -> torch.Tensor:
        """Encode the bipartite atom-pore graph into a per-structure embedding.

        One round of message passing in each direction:
          pore -> atom: each atom aggregates messages from its pore neighbours
          atom -> pore: each pore aggregates messages from its wall atoms

        Then mean-pool over pore nodes to get one vector per structure.
        DGL batches multiple structures into one disconnected graph, so
        mean pooling is done per connected component automatically.

        Args:
            G : batched DGL heterograph from pore_graph_dataset.py
                nodes['atom'].data['h'] : (N_atoms_total, 92)
                nodes['pore'].data['h'] : (N_pores_total, 2)
                edges['wall'].data['d'] : (E_total, 1)

        Returns:
            Tensor (batch_size, pore_hidden)
        """
        # Embed node and edge features
        h_atom = torch.relu(self.pore_atom_emb(G.nodes['atom'].data['h']))
        h_pore = torch.relu(self.pore_node_emb(G.nodes['pore'].data['h']))
        h_edge = torch.relu(self.pore_edge_emb(G.edges['wall'].data['d']))

        # Message passing: pore -> atom
        # Each atom aggregates from pore nodes connected to it.
        # We store h_pore on pore nodes, then use DGL's built-in mean aggregation.
        G.nodes['pore'].data['_h'] = h_pore
        G.nodes['atom'].data['_h'] = h_atom
        G.edges['wall_rev'].data['_e'] = h_edge   # atom<-pore direction
        G.edges['wall'].data['_e']     = h_edge   # for atom->pore pass
        G.update_all(
            message_func=fn.copy_u('_h', '_m'),
            reduce_func=fn.mean('_m', '_agg'),
            etype='wall_rev',
        )
        # Atoms that have no pore neighbours get zero aggregation (empty pores)
        if '_agg' not in G.nodes['atom'].data:
            agg_atom = torch.zeros_like(h_atom)
        else:
            agg_atom = G.nodes['atom'].data['_agg']
        h_atom = torch.relu(h_atom + self.pore_msg_p2a(agg_atom))

        # Message passing: atom -> pore
        # Each pore aggregates from its wall atoms.
        G.nodes['atom'].data['_h'] = h_atom
        # use fn.u_mul_e to weight messages by edge features
        G.update_all(
        message_func=fn.u_mul_e('_h', '_e', '_m'),  # edge-weighted
        reduce_func=fn.mean('_m', '_agg'),
        etype='wall',
        )
        if '_agg' not in G.nodes['pore'].data:
            agg_pore = torch.zeros_like(h_pore)
        else:
            agg_pore = G.nodes['pore'].data['_agg']
        h_pore = torch.relu(h_pore + self.pore_msg_a2p(agg_pore))

        # Readout: mean pool pore nodes per structure
        G.nodes['pore'].data['_h'] = h_pore
        # dgl.mean_nodes works on homogeneous graphs; use readout on the pore subgraph
        pore_emb = dgl.mean_nodes(G, '_h', ntype='pore')  # (batch, pore_hidden)
        pore_emb = self.pore_readout_mlp(pore_emb)

        return pore_emb

    def forward(
        self,
        g: Union[Tuple[dgl.DGLGraph, dgl.DGLGraph], dgl.DGLGraph],
        pore_feats: Optional[torch.Tensor] = None,
        G_pore: Optional[dgl.DGLHeteroGraph] = None,
    ):
        """ALIGNN : start with `atom_features`.

        x: atom features (g.ndata)
        y: bond features (g.edata and lg.ndata)
        z: angle features (lg.edata)

        pore_feats: optional Tensor of shape (batch_size, pore_features)
            Option A: precomputed Zeo++ scalars from pore_utils.py.
            Only used when config.pore_features > 0.

        G_pore: optional DGL heterograph (batched) from pore_graph_dataset.py
            Option B: bipartite atom-pore graph built by bipartite_pore_graph.py.
            Only used when config.use_pore_graph = True.
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

        # PORE OPTION B: encode bipartite pore graph and concatenate
        if self.config.use_pore_graph:
            if G_pore is None:
                raise ValueError(
                    "config.use_pore_graph=True but G_pore was not passed "
                    "to forward(). Check pore_graph_dataset.get_pore_graph_loaders() "
                    "is used and train.py passes G_pore=batch[3]."
                )
            p = self._encode_pore_graph(G_pore)  # (batch, pore_hidden)
            h = torch.cat([h, p], dim=-1)        # (batch, hidden + pore_hidden)

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