"""Frozen unimodal encoders + identical one-logit linear probes.

Run `python baselines.py --help`; see README.md for the complete protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [int(r["id"]) for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty data or duplicate IDs in {path}")
    return rows


def aligned_test_labels(test_rows, annotation_rows):
    annotations = {int(r["id"]): r for r in annotation_rows}
    if len(annotations) != len(annotation_rows) or set(annotations) != {int(r["id"]) for r in test_rows}:
        raise ValueError("Test annotations must match the complete test ID set exactly")
    for r in test_rows:
        other = annotations[int(r["id"])]
        if any(other.get(k) != r[k] for k in ("text", "img")):
            raise ValueError(f"Test input mismatch for ID {r['id']}")
        if type(other.get("label")) is not int or other["label"] not in (0, 1):
            raise ValueError(f"Invalid test label for ID {r['id']}")
    return {i: r["label"] for i, r in annotations.items()}


def prepare(data_dir, config):
    """Preserve original splits; download only the separately stored annotations."""
    rows = {s: read_jsonl(data_dir / f"{s}.jsonl") for s in ("train", "dev", "test")}
    id_sets = {s: {int(r["id"]) for r in rs} for s, rs in rows.items()}
    for a, b in itertools.combinations(id_sets, 2):
        if id_sets[a] & id_sets[b]:
            raise ValueError(f"Overlapping sample IDs: {a}, {b}")
    for split, rs in rows.items():
        for r in rs:
            if not isinstance(r.get("text"), str) or not r["text"].strip():
                raise ValueError(f"Missing text for {split}/{r['id']}")
            image_path = (data_dir / r["img"]).resolve()
            if not image_path.is_relative_to(data_dir.resolve()) or not image_path.is_file():
                raise ValueError(f"Missing or invalid image path for {split}/{r['id']}")
            if split != "test" and (type(r.get("label")) is not int or r["label"] not in (0, 1)):
                raise ValueError(f"Invalid label for {split}/{r['id']}")
    spec = config["test_annotations"]
    target = data_dir / "external" / "test_seen.jsonl"
    if target.exists():
        raw = target.read_bytes()
    else:
        with urllib.request.urlopen(spec["url"], timeout=60) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Downloaded test annotations do not match the pinned SHA-256")
    annotations = [json.loads(line) for line in raw.splitlines() if line.strip()]
    labels = aligned_test_labels(rows["test"], annotations)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(raw)
    audit = {s: {"size": len(rs), "sha256": sha256(data_dir / f"{s}.jsonl"),
                 "class_counts": {str(y): sum(r.get("label") == y for r in rs) for y in (0, 1)}
                 if s != "test" else None} for s, rs in rows.items()}
    audit["test"]["mirror_class_counts"] = {str(y): sum(v == y for v in labels.values()) for y in (0, 1)}
    audit["test_annotations"] = spec
    write_json(target.parent / "provenance.json", audit)
    return audit


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def load_encoder(key, spec, device):
    if key == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        model.fc = torch.nn.Identity()
        processor = weights.transforms()
    elif key == "clip":
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
        model = CLIPVisionModelWithProjection.from_pretrained(spec["checkpoint"], revision=spec["revision"])
        processor = CLIPImageProcessor.from_pretrained(spec["checkpoint"], revision=spec["revision"])
    else:
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(spec["checkpoint"], revision=spec["revision"])
        processor = AutoTokenizer.from_pretrained(spec["checkpoint"], revision=spec["revision"])
    model.requires_grad_(False).eval().to(device)
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    return model, processor


def check_features(x, ids, expected_ids, dimension):
    if x.shape != (len(expected_ids), dimension) or not np.isfinite(x).all():
        raise ValueError("Invalid feature shape or nonfinite embeddings")
    if not np.array_equal(ids, np.asarray(expected_ids, dtype=np.int64)):
        raise ValueError("Embedding IDs/order do not match the dataset")


def extract(args, config):
    torch.set_num_threads(4)
    device = select_device(args.device)
    for key in args.encoders:
        spec = config["encoders"][key]
        pending = []
        for split in args.splits:
            source = args.data_dir / f"{split}.jsonl"
            rows = read_jsonl(source)
            if args.limit:
                rows = rows[:args.limit]
            ids = [int(r["id"]) for r in rows]
            path = args.output_dir / "features" / key / f"{split}.npz"
            expected = {"encoder": spec, "source_sha256": sha256(source), "limit": args.limit}
            if path.exists() and not args.force:
                with np.load(path, allow_pickle=False) as cached:
                    check_features(cached["features"], cached["ids"], ids, spec["dimension"])
                    metadata = json.loads(str(cached["metadata"]))
                    if any(metadata.get(k) != v for k, v in expected.items()):
                        raise ValueError(f"Stale feature cache: {path}; use --force")
                print(f"{key}/{split}: verified cache", flush=True)
            else:
                pending.append((split, rows, ids, path, expected))
        if not pending:
            continue
        print(f"Loading {key} on {device}", flush=True)
        model, processor = load_encoder(key, spec, device)
        for split, rows, ids, path, metadata in pending:
            started = time.monotonic()
            chunks = []
            truncated = 0
            if spec["modality"] == "text":
                lengths = processor([r["text"] for r in rows], truncation=False, padding=False,
                                    return_length=True)["length"]
                truncated = sum(n > spec["max_length"] for n in lengths)
            with torch.inference_mode():
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size]
                    if spec["modality"] == "image":
                        images = []
                        for row in batch:
                            with Image.open(args.data_dir / row["img"]) as img:
                                images.append(img.convert("RGB"))
                        if key == "resnet50":
                            inputs = torch.stack([processor(img) for img in images]).to(device)
                            features = model(inputs)
                        else:
                            inputs = processor(images=images, return_tensors="pt").to(device)
                            features = model(**inputs).image_embeds
                    else:
                        inputs = processor([r["text"] for r in batch], padding=True, truncation=True,
                                           max_length=spec["max_length"], return_tensors="pt").to(device)
                        tokens = model(**inputs).last_hidden_state
                        mask = inputs["attention_mask"].unsqueeze(-1).to(tokens.dtype)
                        features = (tokens * mask).sum(1) / mask.sum(1).clamp_min(1)
                        features = torch.nn.functional.normalize(features, p=2, dim=1)
                    assert not features.requires_grad
                    chunks.append(features.cpu().float().numpy())
                    if (start // args.batch_size) % 25 == 0:
                        print(f"{key}/{split}: {min(start+len(batch),len(rows))}/{len(rows)}", flush=True)
            x = np.concatenate(chunks)
            check_features(x, ids, ids, spec["dimension"])
            metadata.update({"device": str(device), "seconds": time.monotonic()-started,
                             "truncated_examples": truncated, "frozen": True, "dtype": "float32"})
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, features=x, ids=np.asarray(ids, dtype=np.int64),
                                metadata=json.dumps(metadata))
            temporary.replace(path)
            print(f"{key}/{split}: saved {x.shape}, {metadata['seconds']:.1f}s, truncated={truncated}", flush=True)
        del model, processor
        if device.type == "mps":
            torch.mps.empty_cache()


def load_features(args, config, key, split):
    source = args.data_dir / f"{split}.jsonl"
    rows = read_jsonl(source)
    with np.load(args.output_dir / "features" / key / f"{split}.npz", allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["limit"]:
            rows = rows[:metadata["limit"]]
        x, ids = data["features"].copy(), data["ids"].copy()
    check_features(x, ids, [r["id"] for r in rows], config["encoders"][key]["dimension"])
    if metadata["encoder"] != config["encoders"][key] or metadata["source_sha256"] != sha256(source):
        raise ValueError("Feature metadata does not match current data/configuration")
    return x, ids, rows, metadata


def metrics(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores)
    if labels.shape != scores.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Expected one finite hateful-class probability per example")
    return {"auroc": float(roc_auc_score(labels, scores)),
            "accuracy_at_0.5": float(accuracy_score(labels, scores >= 0.5))}


def fit_probe(x_train, y_train, x_dev, y_dev, *, seed, lr, weight_decay, batch_size, epochs):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(seed)
    best_auc, best = -1.0, None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        loss_sum = 0.0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_train[indices]).squeeze(1), y_train[indices])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        model.eval()
        with torch.inference_mode():
            scores = model(x_dev).squeeze(1).sigmoid().numpy()
        auc = metrics(y_dev, scores)["auroc"]
        history.append({"epoch": epoch, "train_bce": loss_sum/len(x_train), "validation_auroc": auc})
        if auc > best_auc:
            best_auc = auc
            best = {"state_dict": {k: v.detach().clone() for k, v in model.state_dict().items()},
                    "epoch": epoch, "validation_auroc": auc, "seed": seed}
    return best, history


def train(args, config):
    # A single CPU thread avoids oversubscription for these tiny matrix operations.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    settings = config["training"]
    for key in args.encoders:
        started = time.monotonic()
        x, _, train_rows, train_meta = load_features(args, config, key, "train")
        v, _, dev_rows, dev_meta = load_features(args, config, key, "dev")
        scaler = StandardScaler().fit(x)
        mean_tensor = torch.from_numpy(scaler.mean_.astype(np.float32))
        scale_tensor = torch.from_numpy(scaler.scale_.astype(np.float32))
        xt = (torch.from_numpy(x) - mean_tensor) / scale_tensor
        xv = (torch.from_numpy(v) - mean_tensor) / scale_tensor
        yt = torch.tensor([r["label"] for r in train_rows], dtype=torch.float32)
        yv = np.asarray([r["label"] for r in dev_rows])
        trials, winner, winner_mean = [], None, -1.0
        for lr, decay in itertools.product(settings["learning_rates"], settings["weight_decays"]):
            runs = []
            histories = []
            for seed in config["seeds"]:
                run, history = fit_probe(xt, yt, xv, yv, seed=seed, lr=lr, weight_decay=decay,
                                         batch_size=settings["batch_size"], epochs=settings["epochs"])
                runs.append(run)
                histories.append({"seed": seed, "history": history})
            mean = float(np.mean([r["validation_auroc"] for r in runs]))
            trials.append({"learning_rate": lr, "weight_decay": decay, "mean_validation_auroc": mean,
                           "runs": [{k: v for k, v in r.items() if k != "state_dict"} for r in runs],
                           "histories": histories})
            print(f"{key}: lr={lr:g} wd={decay:g} validation AUROC={mean:.6f}", flush=True)
            if mean > winner_mean:
                winner_mean, winner = mean, (lr, decay, runs)
        lr, decay, selected = winner
        destination = args.output_dir / "checkpoints" / key
        destination.mkdir(parents=True, exist_ok=True)
        for run in selected:
            checkpoint = {**run, "encoder": config["encoders"][key],
                          "mean": mean_tensor,
                          "scale": scale_tensor,
                          "training": {**settings, "learning_rate": lr, "weight_decay": decay},
                          "train_source_sha256": train_meta["source_sha256"],
                          "dev_source_sha256": dev_meta["source_sha256"]}
            torch.save(checkpoint, destination / f"seed_{run['seed']}.pt")
        summary = {"encoder": config["encoders"][key], "selected_learning_rate": lr,
                   "selected_weight_decay": decay, "seeds": config["seeds"],
                   "best_epochs": {str(r["seed"]): r["epoch"] for r in selected},
                   "validation_auroc_mean": winner_mean,
                   "validation_auroc_std": float(np.std([r["validation_auroc"] for r in selected], ddof=1)),
                   "training_seconds": time.monotonic()-started,
                   "training_examples": len(train_rows), "validation_examples": len(dev_rows),
                   "trials": trials}
        write_json(destination / "selection.json", summary)
        print(f"{key}: selected lr={lr:g}, wd={decay:g}, {summary['training_seconds']:.1f}s", flush=True)


def score_checkpoint(checkpoint, x):
    model = torch.nn.Linear(x.shape[1], 1)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.inference_mode():
        normalized = (torch.from_numpy(x) - checkpoint["mean"]) / checkpoint["scale"]
        return model(normalized).squeeze(1).sigmoid().numpy()


def write_predictions(path, ids, labels, scores):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "label", "score", "prediction_at_0.5"])
        writer.writerows((int(i), int(y), float(s), int(s >= .5)) for i, y, s in zip(ids, labels, scores, strict=True))


def evaluate(args, config):
    torch.set_num_threads(1)
    test_labels = None
    if "test" in args.splits:
        prepare(args.data_dir, config)
        test_labels = aligned_test_labels(read_jsonl(args.data_dir / "test.jsonl"),
                                         read_jsonl(args.data_dir / "external" / "test_seen.jsonl"))
    for key in args.encoders:
        result = {}
        selection = json.loads((args.output_dir / "checkpoints" / key / "selection.json").read_text())
        for split in args.splits:
            x, ids, rows, metadata = load_features(args, config, key, split)
            labels = np.asarray([test_labels[int(i)] for i in ids] if split == "test" else [r["label"] for r in rows])
            seed_results = []
            for seed in config["seeds"]:
                checkpoint = torch.load(args.output_dir / "checkpoints" / key / f"seed_{seed}.pt", weights_only=True,
                                        map_location="cpu")
                if checkpoint["encoder"] != config["encoders"][key]:
                    raise ValueError("Checkpoint encoder mismatch")
                for training_split in ("train", "dev"):
                    if checkpoint[f"{training_split}_source_sha256"] != sha256(args.data_dir / f"{training_split}.jsonl"):
                        raise ValueError("Training/validation data changed after checkpoint selection")
                scores = score_checkpoint(checkpoint, x)
                values = metrics(labels, scores)
                if split == "dev" and not np.isclose(values["auroc"], checkpoint["validation_auroc"], atol=1e-6):
                    raise ValueError("Reloaded checkpoint AUROC differs from model selection")
                seed_results.append({"seed": seed, **values})
                write_predictions(args.output_dir / "predictions" / key / f"{split}_seed_{seed}.csv", ids, labels, scores)
            result[split] = {"size": len(ids), "auroc_mean": float(np.mean([r["auroc"] for r in seed_results])),
                             "auroc_std": float(np.std([r["auroc"] for r in seed_results], ddof=1)),
                             "runs": seed_results,
                             "label_source": config["test_annotations"] if split == "test" else "Supplied course labels"}
        write_json(args.output_dir / "metrics" / f"{key}.json", {"selection": {k: v for k, v in selection.items() if k != "trials"},
                                                                "evaluation": result})
        print(key, json.dumps({s: {k: v for k, v in r.items() if k.startswith('auroc')} for s, r in result.items()}), flush=True)
    available = {}
    for key in config["encoders"]:
        path = args.output_dir / "metrics" / f"{key}.json"
        if path.exists():
            available[key] = json.loads(path.read_text())
    write_json(args.output_dir / "results.json", available)
    with (args.output_dir / "results.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["encoder", "checkpoint", "validation_auroc_mean", "validation_auroc_std", "test_auroc_mean", "test_auroc_std"])
        for key, r in available.items():
            ev = r["evaluation"]
            writer.writerow([key, config["encoders"][key]["checkpoint"],
                             ev.get("dev", {}).get("auroc_mean"), ev.get("dev", {}).get("auroc_std"),
                             ev.get("test", {}).get("auroc_mean"), ev.get("test", {}).get("auroc_std")])


def analyze(args, config):
    results = json.loads((args.output_dir / "results.json").read_text())
    best = {modality: max((key for key, spec in config["encoders"].items() if spec["modality"] == modality),
                          key=lambda key: results[key]["evaluation"]["dev"]["auroc_mean"])
            for modality in ("image", "text")}
    scores = {}
    for modality, key in best.items():
        with (args.output_dir / "predictions" / key / "dev_seed_42.csv").open() as stream:
            scores[modality] = {int(r["id"]): float(r["score"]) for r in csv.DictReader(stream)}
    cases = []
    for row in read_jsonl(args.data_dir / "dev.jsonl"):
        i = int(row["id"])
        if i not in scores["image"] or i not in scores["text"]:
            continue
        image, text = scores["image"][i], scores["text"][i]
        cases.append({**row, "image_score": image, "text_score": text,
                      "image_prediction": int(image >= .5), "text_prediction": int(text >= .5),
                      "disagreement": (image >= .5) != (text >= .5)})
    cases.sort(key=lambda r: (-int(r["disagreement"]), -abs(r["image_score"]-r["text_score"]), int(r["id"])))
    output = {"best_encoders_by_validation": best, "threshold": .5, "seed": 42,
              "selection_rule": "Disagreements first, then descending absolute score gap, then ascending ID",
              "disagreement_count": sum(r["disagreement"] for r in cases), "examples": cases[:5]}
    write_json(args.output_dir / "analysis.json", output)
    print(json.dumps(output, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "extract", "train", "evaluate", "analyze"])
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--encoders", nargs="+", choices=["resnet50", "clip", "minilm", "mpnet"])
    parser.add_argument("--splits", nargs="+", choices=["train", "dev", "test"])
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32, help="Feature extraction batch size; classifier batch size comes from config")
    parser.add_argument("--limit", type=int, default=0, help="Smoke tests only: first N examples per split, 0 = full data")
    parser.add_argument("--force", action="store_true", help="Re-extract existing feature caches")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        parser.error("batch size must be positive and limit nonnegative")
    config = json.loads(args.config.read_text())
    args.encoders = args.encoders or list(config["encoders"])
    args.splits = args.splits or (["dev", "test"] if args.command == "evaluate" else ["train", "dev", "test"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        print(json.dumps(prepare(args.data_dir, config), indent=2))
    else:
        {"extract": extract, "train": train, "evaluate": evaluate, "analyze": analyze}[args.command](args, config)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "packages": {p: importlib.metadata.version(p) for p in
                                ["torch", "torchvision", "transformers", "huggingface-hub", "numpy", "Pillow", "scikit-learn", "scipy"]}}
    write_json(args.output_dir / "environment.json", environment)


if __name__ == "__main__":
    main()
