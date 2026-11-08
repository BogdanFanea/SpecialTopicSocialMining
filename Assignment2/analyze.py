"""Select six validation examples reproducibly for qualitative inspection."""
import csv
import json
from pathlib import Path

import numpy as np

from main import HERE, ROOT, auroc, read_rows


def read_scores(path, expected, label_by_id=None):
    with path.open(newline="", encoding="utf-8") as file:
        entries = list(csv.DictReader(file))
    if label_by_id is None:
        scores = {int(r["meme_id"]): float(r["hateful_probability"]) for r in entries}
    else:
        scores = {int(r["id"]): float(r["score"]) for r in entries}
        if any(int(r["label"]) != label_by_id[int(r["id"])] for r in entries):
            raise ValueError(f"Baseline labels disagree with the dev split: {path}")
    if len(entries) != len(scores) or set(scores) != set(expected):
        raise ValueError(f"Scores do not match dev IDs: {path}")
    return np.array([scores[i] for i in expected])


def main():
    output = HERE / "artifacts"
    results = json.loads((output / "results.json").read_text())
    op = next(t for t in results["trials"] if t["lambda"] == results["selected_op_lambda"])
    # Analyze the FOP regularized variant at the fixed first seed.
    run = op["runs"][0]
    seed = run["seed"]
    _, rows = read_rows("dev")
    ids = [r["id"] for r in rows]
    y = np.array([r["label"] for r in rows])
    by_id = dict(zip(ids, y))
    baseline_path = ROOT / "Assignment1" / "artifacts" / "predictions"
    baselines = {}
    for key in ("clip", "mpnet"):
        baselines[key] = [read_scores(baseline_path / key / f"dev_seed_{s}.csv", ids, by_id)
                          for s in (42, 43, 44)]
    image = baselines["clip"][(42, 43, 44).index(seed)]
    text = baselines["mpnet"][(42, 43, 44).index(seed)]
    fused = read_scores((output / run["checkpoint"]).with_suffix(".csv"), ids)
    p = fused >= 0.5
    changed = ((image >= 0.5) != p) | ((text >= 0.5) != p)
    selected = []
    groups = [
        ("False positive", p & (y == 0), fused),
        ("False negative", ~p & (y == 1), 1 - fused),
        ("Corrected unimodal decision", changed & (p == y),
         np.maximum(abs(fused - image), abs(fused - text))),
    ]
    for name, mask, priority in groups:
        order = sorted(np.flatnonzero(mask), key=lambda i: (-priority[i], ids[i]))
        for i in order[:2]:
            selected.append({"id": ids[i], "label": int(y[i]), "kind": name,
                             "image": float(image[i]), "text": float(text[i]),
                             "fop": float(fused[i]), "img": rows[i]["img"]})
    summary = {}
    for key, scores in baselines.items():
        aucs = [auroc(y, s) for s in scores]
        summary[key] = {"seed_aurocs": aucs, "mean": float(np.mean(aucs)),
                        "std": float(np.std(aucs, ddof=1))}
    payload = {"seed": seed, "lambda": op["lambda"], "threshold": 0.5, "cases": selected,
               "baseline_dev": summary}
    (output / "analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
