"""Compare two h128 checkpoints on the production export val split, with a
leak-free subset (val lines the OLD model never trained on, from the export
manifest createdAt cutoff) -- full-val NLL across corpus growth is leakage-
poisoned in the old model's favor (see CLAUDE.md progress #15).

Usage (from handwriting-generation/, with ../data/export at the SAME snapshot
the new model trained on, or the split will differ):
  python eval_models.py <manifest.json> [old.pt new.pt] [cutoff ISO timestamp]
"""
import json, os, sys, torch, torch.nn as nn
sys.path.insert(0, ".")
from train import collate_fn, evaluate
from handwriting_dataset import ExportDataset, split_by_text
from model import HandwritingRNN
from torch.utils.data import DataLoader, Subset

dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
ds = ExportDataset(); train_idx, val_idx = split_by_text(ds, val_frac=0.2)
print(f"export {len(ds)} lines; split train {len(train_idx)} / val {len(val_idx)}")

# Champion corpus = chunk texts recorded before the 2026-09-06 00:42 UTC export
man = json.load(open(sys.argv[1]))
man = man["recordings"] if isinstance(man, dict) else man
cutoff = sys.argv[4] if len(sys.argv) > 4 else "2026-09-06T00:42"
champ_texts = {r["text"] for r in man if r["userId"] == 2 and r["createdAt"] < cutoff}
thin = {json.loads(l)["text"] for l in open("../collection-app/curated/thin-2026-09.jsonl") if l.strip()}
val_meta = [(i, ds.meta[i]["text"]) for i in val_idx]
leakfree = [i for i, t in val_meta if t not in champ_texts]
lf_thin = [i for i in leakfree if any(ds.meta[i]["text"] in s or s in ds.meta[i]["text"] for s in thin)]
lf_wiki = [i for i in leakfree if i not in set(lf_thin)]
print(f"val lines unseen by champion (leak-free): {len(leakfree)}  [thin-synthetic {len(lf_thin)}, other {len(lf_wiki)}]; seen-by-champion {len(val_idx)-len(leakfree)}")

crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0).to(dev), reduction="none")
def load(path):
    m = HandwritingRNN(input_size=4, hidden_size=128, num_layers=3, dropout=0.0, embedding_size=8).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=True)); return m.eval()
old_pt = sys.argv[2] if len(sys.argv) > 2 else "best_model.eot.pt"
new_pt = sys.argv[3] if len(sys.argv) > 3 else "runs/2026-10-07-retrain/best_model.pt"
models = {f"old ({old_pt})": load(old_pt), f"new ({new_pt})": load(new_pt)}
subsets = {"full val": val_idx, "leak-free": leakfree, "  leak-free/thin-synthetic": lf_thin, "  leak-free/other": lf_wiki,
           "seen-by-champion (leaky)": [i for i in val_idx if i not in set(leakfree)]}
print(f"{'subset':<28}{'n':>5} | {'champion total':>15}{'xy':>9}{'pen':>8} | {'new total':>11}{'xy':>9}{'pen':>8} | {'delta':>7}")
for name, idx in subsets.items():
    if not idx: continue
    dl = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, collate_fn=collate_fn)
    res = []
    with torch.no_grad():
        for m in models.values():
            res.append(evaluate(m, dl, crit, dev))
    (ct, cx, cp), (nt, nx, np_) = res
    print(f"{name:<28}{len(idx):>5} | {ct:>15.4f}{cx:>9.4f}{cp:>8.4f} | {nt:>11.4f}{nx:>9.4f}{np_:>8.4f} | {ct-nt:>+7.4f}")
