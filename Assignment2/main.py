"""Train and score FOP on Assignment 1's cached Hateful Memes features."""
import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from retrieval_model import FOP, cross_entropy, orthogonal_loss


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENCODERS = ("clip", "mpnet")


def read_rows(split):
    source = ROOT / "Assignment1" / "data" / f"{split}.jsonl"
    return source, [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]


def load_split(split, specs):
    source, rows = read_rows(split)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate ID in {source}")
    # Git may check out the same JSONL with different line endings.
    raw = source.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    digests = {hashlib.sha256(data).hexdigest() for data in (raw, lf, lf.replace(b"\n", b"\r\n"))}
    features = []
    for key in ENCODERS:
        path = ROOT / "Assignment1" / "artifacts" / "features" / key / f"{split}.npz"
        with np.load(path, allow_pickle=False) as cache:
            x, cached_ids = cache["features"], cache["ids"]
            meta = json.loads(str(cache["metadata"]))
        index = {int(meme_id): i for i, meme_id in enumerate(cached_ids)}
        if (meta["source_sha256"] not in digests or meta["encoder"] != specs[key] or meta["limit"]
                or not meta["frozen"] or len(index) != len(cached_ids) or set(index) != set(ids)
                or x.shape != (len(ids), specs[key]["dimension"]) or not np.isfinite(x).all()):
            raise ValueError(f"Misaligned or invalid cache: {path}")
        features.append(x[[index[meme_id] for meme_id in ids]].astype(np.float32))
    labels = None if split == "test" else np.array([r["label"] for r in rows], dtype=int)
    if labels is not None and set(labels) != {0, 1}:
        raise ValueError(f"Expected both binary labels in {source}")
    return (*features, labels, ids)


def auroc(labels, scores):
    """Mann–Whitney AUROC with half credit for tied scores."""
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    n_pos = int(labels.sum())
    rank_sum = average_ranks[inverse][labels == 1].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * (len(labels) - n_pos)))


def batches(labels, size, rng):
    count = (len(labels) + size - 1) // size
    negative, positive = (np.array_split(rng.permutation(np.flatnonzero(labels == y)), count)
                          for y in (0, 1))
    for a, b in zip(negative, positive):
        if min(len(a), len(b)) < 2:
            raise ValueError("Batch size too small for at least two examples per class")
        yield rng.permutation(np.concatenate((a, b)))


def make_model(config, specs, fusion, seed):
    dims = [specs[key]["dimension"] for key in ENCODERS]
    return FOP(dims, config["dimension"], config["gate_hidden"], config["dropout"], fusion, seed)


def fit(config, specs, train, dev, fusion, weight, seed):
    model = make_model(config, specs, fusion, seed)
    rng = np.random.default_rng(seed)
    first = {name: np.zeros_like(p) for name, p in model.params.items()}
    second = {name: np.zeros_like(p) for name, p in model.params.items()}
    lr, decay = config["learning_rate"], config["weight_decay"]
    image, text, labels, _ = train
    step = 0
    best_auc = -1
    history = []
    for epoch in range(1, config["epochs"] + 1):
        loss_sum = 0
        for idx in batches(labels, config["batch_size"], rng):
            logits, z, backward = model.forward(image[idx], text[idx], training=True)
            loss, grad = cross_entropy(logits, labels[idx])
            extra = 0
            if weight:
                op, op_grad = orthogonal_loss(z, labels[idx])
                loss += weight * op
                extra = weight * op_grad
            grads = backward(grad, extra)
            step += 1
            for name, param in model.params.items():
                g = grads[name]
                m, v = first[name], second[name]
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                param *= 1 - lr * decay
                param -= (lr / (1 - 0.9**step)) * m / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            loss_sum += float(loss) * len(idx)
        scores = cross_entropy(model.forward(*dev[:2])[0])
        auc = auroc(dev[2], scores)
        history.append({"epoch": epoch, "loss": loss_sum / len(labels), "dev_auroc": auc})
        if auc > best_auc:
            best_auc, best_epoch = auc, epoch
            state = {k: v.copy() for k, v in (model.params | model.stats).items()}
    metrics = {"seed": seed, "epoch": best_epoch, "dev_auroc": best_auc, "history": history,
               "parameters": sum(p.size for p in model.params.values())}
    return state, metrics


def save_checkpoint(path, state, config, specs, fusion, weight, metrics):
    metadata = {"config": config, "encoders": specs, "fusion": fusion, "lambda": weight,
                **{k: v for k, v in metrics.items() if k != "history"}}
    np.savez_compressed(path, **state, metadata=json.dumps(metadata))


def predict(path, split):
    with np.load(path, allow_pickle=False) as checkpoint:
        metadata = json.loads(str(checkpoint["metadata"]))
        specs = metadata["encoders"]
        model = make_model(metadata["config"], specs, metadata["fusion"], metadata["seed"])
        for state in (model.params, model.stats):
            for name in state:
                state[name] = checkpoint[name]
    data = load_split(split, specs)
    return data[3], data[2], cross_entropy(model.forward(*data[:2])[0])


def write_scores(path, ids, scores):
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid probabilities")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("meme_id", "hateful_probability"))
        writer.writerows(zip(ids, map(float, scores), strict=True))


def train_all(config, output):
    full_specs = json.loads((ROOT / "Assignment1" / "config.json").read_text())["encoders"]
    specs = {key: full_specs[key] for key in ENCODERS}
    train = load_split("train", specs)
    dev = load_split("dev", specs)
    if set(train[3]) & set(dev[3]):
        raise ValueError("Overlapping training and validation IDs")
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    settings = [("linear", 0), ("gated", 0)]
    settings.extend(("gated", weight) for weight in config["op_weights"])
    for fusion, weight in settings:
        runs = []
        for seed in config["seeds"]:
            state, metrics = fit(config, specs, train, dev, fusion, weight, seed)
            name = f"{fusion}_lambda{weight:g}_seed{seed}.npz"
            save_checkpoint(output / name, state, config, specs, fusion, weight, metrics)
            runs.append({"checkpoint": name, **metrics})
            print(f"{name}: epoch {metrics['epoch']}, dev AUROC {metrics['dev_auroc']:.5f}", flush=True)
        aucs = [run["dev_auroc"] for run in runs]
        trials.append({"fusion": fusion, "lambda": weight, "mean": float(np.mean(aucs)),
                       "std": float(np.std(aucs, ddof=1)), "runs": runs})
    selected_op = max(trials[2:], key=lambda trial: trial["mean"])
    selected = max([trials[0], trials[1], selected_op], key=lambda trial: trial["mean"])
    # Use a predetermined seed for deployment, without a second round of seed selection.
    best = selected["runs"][0]["checkpoint"]
    results = {"trials": trials, "selected_op_lambda": selected_op["lambda"], "selected": best,
               "environment": {"python": platform.python_version(), "numpy": np.__version__,
                               "platform": platform.platform(), "device": "CPU"}}
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    for trial in (trials[0], trials[1], selected_op):
        for run in trial["runs"]:
            checkpoint = output / run["checkpoint"]
            ids, _, scores = predict(checkpoint, "dev")
            write_scores(checkpoint.with_suffix(".csv"), ids, scores)
    print(f"Selected: {best}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--config", type=Path, default=HERE / "config.json")
    train.add_argument("--output", type=Path, default=HERE / "artifacts")
    score = commands.add_parser("predict")
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--split", choices=("dev", "test"), default="test")
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train_all(json.loads(args.config.read_text()), args.output)
    else:
        ids, _, scores = predict(args.checkpoint, args.split)
        write_scores(args.output, ids, scores)


if __name__ == "__main__":
    main()
