# Assignment 2 · Multimodal Hateful Meme Classification

Bogdan Fanea, Lena Bryll, Natalie Stadler

Run these commands from the repository root with Python 3.12.

```sh
python -m pip install -r Assignment2/requirements.txt
python Assignment2/main.py train
python Assignment2/analyze.py
```

The course split files and Assignment 1 validation predictions are already in
`Assignment1/`. Training needs the CLIP and MPNet train/dev feature caches under
`Assignment1/artifacts/features/`. Inference uses the corresponding test caches.
If you do not already have the caches, place the course images in
`Assignment1/data/img/`, install the dependencies in
[`Assignment1/requirements.lock.txt`](../Assignment1/requirements.lock.txt), and
generate the embeddings from the repository root:

```sh
python Assignment1/baselines.py extract --encoders clip mpnet --device auto --batch-size 32
```

See the [Assignment 1 setup instructions](../Assignment1/README.md) for details.

For held-out inference with the included selected checkpoint, run the command below. If you retrain with a different configuration, use the checkpoint filename in the `selected` field of `Assignment2/artifacts/results.json`.

```text
python Assignment2/main.py predict --checkpoint Assignment2/artifacts/gated_lambda0_seed42.npz --split test --output Assignment2/artifacts/test_predictions.csv
```

The CSV columns are `meme_id,hateful_probability`, with one row per test ID in course split order. The official course test labels are unavailable. Validation AUROC is used throughout the comparison. The public mirror labels used in Assignment 1's exploratory evaluation are not used here. The test prediction file is ready for submission through the course or challenge protocol.

## Design

- ID-based joins validate exact split membership, cache dimensions, encoder specifications, finite embeddings and the original annotation SHA-256. Labels come from the original course train/dev JSONL.
- The frozen encoder pair is CLIP ViT-B/32 (512 image features) and all-mpnet-base-v2 (768 text features). Exact revisions and feature preprocessing are in `Assignment1/config.json` and the trained checkpoints.
- Each reference projection branch has a linear layer, BatchNorm, ReLU, 0.5 dropout and rowwise L2 normalization. The two trainable fusion choices are two-scalar linear weighted addition and a featurewise gate whose attention MLP uses a 128-wide Linear/BatchNorm/ReLU block, a second linear layer, and sigmoid. The head has two logits.
- Optimizer is AdamW, learning rate 0.001, weight decay 0.01, 20 epochs, maximum 128 examples per batch and seeds 42, 43 and 44. Each epoch shuffles each label separately and partitions both shuffled index lists into 67 chunks, then pairs and reshuffles the chunks. Every item appears once per epoch and every batch includes at least two samples from each class.
- Each run keeps the earliest epoch with maximal validation AUROC. Select the positive OP weight from 0.1, 0.3 and 1.0 using mean validation AUROC, then select the highest mean validation AUROC among linear CE, gated CE and gated CE+OP. The first configured seed is used for the submitted checkpoint to avoid another round of seed selection. Ties retain configuration order. Standard deviations are sample SDs across three seeds.
- The NumPy implementation follows the operations in the official FOP code. It implements explicit gradients and includes affine BatchNorm parameters and inference running statistics in the checkpoints. NumPy/CPU training is sufficient because the embeddings are cached. Matched gated CE and CE+OP runs use the same initialization, batch order and dropout masks for a given seed.

## Outputs

- `artifacts/results.json` includes all validation trial histories, checkpoint selections, per-seed values and the execution environment.
- `artifacts/*.npz` are trained heads, projections and fusion modules including BatchNorm running state and complete metadata.
- `artifacts/analysis.json` records reproducibly selected validation cases and baseline AUROCs recomputed from Assignment 1 predictions.
- `artifacts/test_predictions.csv` contains held-out probabilities from the selected checkpoint.
- [`output/pdf/report.pdf`](output/pdf/report.pdf) contains the technical report and references. Run `python Assignment2/build_report.py` from the repository root to rebuild it.

## Sources

- [Saeed et al., FOP](https://arxiv.org/abs/2112.10483) and the [official implementation](https://github.com/msaadsaeed/FOP).
- [Kiela et al., Hateful Memes Challenge](https://arxiv.org/abs/2005.04790).
- [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) and [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
