import os, torch
from jarvis.db.figshare import data as jdata
from jarvis.core.atoms import Atoms
from bipartite_pore_graph import build_bipartite_pore_graph
from tqdm import tqdm

OUTPUT_PATH = '/home/milan/Downloads/alignn/Thesis/Pore_plan_b/pore_graphs_1pct.pt'

if __name__ == '__main__':
    raw    = jdata('hmof')
    N      = int(len(raw) * 0.01)
    subset = raw[:N]

    pore_graphs = torch.load(OUTPUT_PATH, weights_only=False) if os.path.exists(OUTPUT_PATH) else {}
    print(f'Resuming: {len(pore_graphs)} done, {N - len(pore_graphs)} remaining')

    for entry in tqdm(subset):
        jid = str(entry['id'])
        if jid in pore_graphs:
            continue
        try:
            G = build_bipartite_pore_graph(Atoms.from_dict(entry['atoms']))
        except Exception:
            G = None
        pore_graphs[jid] = G
        if len(pore_graphs) % 100 == 0:
            torch.save(pore_graphs, OUTPUT_PATH)

    torch.save(pore_graphs, OUTPUT_PATH)
    print(f'Done. {sum(1 for v in pore_graphs.values() if v is None)} failed/no pores')