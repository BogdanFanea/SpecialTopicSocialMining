# Assignment 1: frozen unimodal baselines

Bogdan Fanea, Lena Bryll, Natalie Stadler

Repository: https://github.com/BogdanFanea/SpecialTopicSocialMining

Four independent frozen encoders classify the provided memes with the same
single linear layer. No features are fused and no encoder is fine-tuned.

## Setup and execution

Python 3.12.5 was used on an Apple M3 Mac with 16 GB unified memory. Create an
isolated environment and install the pinned direct dependencies:

The submission bundle includes the split annotations. Copy the supplied image
folder to `data/img/` before running the commands; the images are not bundled.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m unittest -v test_baselines
python baselines.py prepare
python baselines.py extract --device auto --batch-size 32
python baselines.py train
python baselines.py evaluate
python baselines.py analyze
```

The first extraction downloads pretrained weights; subsequent runs use the
Hugging Face/TorchVision caches and cached embeddings. `auto` chooses CUDA,
then Apple MPS, then CPU. Only extraction uses the GPU; classifier training
uses one CPU thread for reproducibility and to avoid overhead on small batches.
The setup used during development reused installed packages through a virtual
environment with `--system-site-packages`; a clean environment can use the
commands above. `requirements.txt` pins direct dependencies; `requirements.lock.txt`
also pins their transitive dependencies. The exact execution environment is in
`artifacts/environment.json`.

Each encoder may be run separately with `--encoders resnet50`, `clip`, `minilm`,
or `mpnet`. Each extraction/evaluation split can be selected with `--splits`.
Defaults are all splits for extraction and `dev test` for evaluation.

For a smoke run, use a separate output directory so partial caches cannot be
mistaken for the final experiment:

```sh
python baselines.py extract --output-dir artifacts/smoke --limit 64
python baselines.py train --output-dir artifacts/smoke
python baselines.py evaluate --output-dir artifacts/smoke
python baselines.py analyze --output-dir artifacts/smoke
```

`--force` re-extracts cached embeddings. Caches otherwise reject changed data,
checkpoint configurations, or limits. `--data-dir`, `--output-dir`, and `--config`
accept alternate locations. Original split files and images are never modified.

## Data and recovered test annotations

The supplied release, also mirrored at
[emily49/hateful-memes](https://huggingface.co/datasets/emily49/hateful-memes),
has 8,500 training examples (5,450 non-hateful / 3,050 hateful), 500 validation
examples (250 / 250), and 1,000 test examples without labels. All 10,000 local
images are present. The split IDs are disjoint; confounder images or texts can
still recur, as intended by the dataset construction. The supplied split
membership and train/validation labels are authoritative for this assignment.

`prepare` downloads **only** the 119,580-byte labeled `test_seen.jsonl` from:

https://huggingface.co/datasets/neuralcatcher/hateful_memes/resolve/d201c488dc7024623d1ecbcc987b3f132c4c2e12/test_seen.jsonl

SHA-256:
`dd0d387bacd67ce4db39fcfd805a0b4b9e10fe0feaafc3b189b41c4ef2e36a16`

This file contains exactly the same 1,000 IDs, texts, and image paths as the
supplied `test.jsonl`; its labels are 510 non-hateful / 490 hateful. It was
also independently retrieved from
[William Berrios's Kaggle mirror](https://www.kaggle.com/datasets/williamberrios/hateful-memes)
and found byte-identical on 2026-10-17. A published research implementation
[documents using this Kaggle source for test labels](https://github.com/YiLunLee/missing_aware_prompts/blob/9aa538159f4623de1936d25daee880ec47c062be/DATA.md).

**Provenance limitation:** the mirror does not establish that these labels are
the original competition ground truth. The official
[Meta fine-grained repository](https://github.com/facebookresearch/fine_grained_hateful_memes/blob/845c5f8628209ae6f8d2f391aeb6eea2bb007a07/data/annotations/test.jsonl)
still supplies an unlabeled test file. Label versions differ: the neuralcatcher
mirror disagrees with the supplied course release on **823 training labels and
three validation labels**; the official fine-grained annotations disagree on
833 training and four validation labels. None of those replacements is applied.
Our test metrics are therefore explicitly **AUROC against public-mirror labels**,
not authenticated competition scores. Do not claim direct leaderboard comparability.
The README's historical future release refers to an additional *unseen* split,
not a promise to publish test ground truth. That different split is not used here.

Downloaded annotations stay in `data/external/test_seen.jsonl`, separately from
the supplied files. `data/external/provenance.json` records the source revision,
checksum, original split hashes, and class counts. Label alignment is by ID,
never by row position. A checksum, duplicate ID, incomplete ID set, changed
text/image path, or invalid label causes an error before the file is accepted.

## Encoders and preprocessing

All checkpoints, immutable Hugging Face revisions, dimensions, pooling methods,
and sources are in `config.json` and copied into feature/checkpoint metadata.

| Key | Exact checkpoint | Dimension | Input and pooling |
|---|---|---:|---|
| `resnet50` | `torchvision.models.ResNet50_Weights.IMAGENET1K_V2` (`resnet50-11ad3fa6.pth`) | 2048 | RGB pixels; global average pooling, original classification layer removed |
| `clip` | `openai/clip-vit-base-patch32` | 512 | RGB pixels; pooled vision CLS token and pretrained visual projection |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Supplied text; attention-mask mean pooling, then L2 normalization |
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | 768 | Supplied text; attention-mask mean pooling, then L2 normalization |

ResNet uses its weights' official transforms: shorter-edge resize to 232,
bilinear interpolation, a 224x224 center crop, scaling to [0,1], and mean
`[0.485,0.456,0.406]` / standard deviation `[0.229,0.224,0.225]`.
CLIP uses the pinned checkpoint's `CLIPImageProcessor`: bicubic shorter-edge
resize to 224 and 224x224 center crop, scaling to [0,1], and mean
`[0.48145466,0.4578275,0.40821073]` / standard deviation
`[0.26862954,0.26130258,0.27577711]`.
Neither image branch receives the separate meme text or text prompts. CLIP's
text tower is unused. Text visible inside the meme remains in the pixels.

The text branches use the checkpoint's `AutoTokenizer`, dynamic batch padding,
right truncation, and maximum lengths (including special tokens) of 256 for
MiniLM and 384 for MPNet. They preserve the supplied strings; matching tokenizer
normalization is applied automatically. Mean pooling includes every token with
attention mask 1, including non-padding special tokens. Actual truncation counts
are recorded per split in feature metadata. Encoders run in evaluation and
inference mode with gradients disabled. Embeddings are saved as float32.

## Classifier and selection protocol

- Fit a per-feature `StandardScaler` on **training embeddings only**, independently
  for each encoder. Constant dimensions get scale 1. Store float32 mean/scale
  with the trained classifier and apply the identical transformation at inference.
- Classifier: `torch.nn.Linear(d, 1)` including bias; no hidden layer, dropout,
  fusion, encoder fine-tuning, class weighting, augmentation, or threshold tuning.
- Objective: mean `BCEWithLogitsLoss`, hateful = 1; score = `sigmoid(logit)`.
- AdamW with its defaults (`betas=(0.9,0.999)`, `eps=1e-8`), batch size 256,
  shuffled training examples each epoch, and 50 epochs per trial. Weight decay
  applies to weight and bias. Learning rate is constant; no scheduler or early stop.
- The same six configurations for each encoder, ordered by learning rate then
  weight decay: `{0.0001,0.001,0.01}` x `{0,0.01}`.
- Run all configurations with seeds 42, 43, and 44 (72 fits total). For each
  fit retain the epoch with highest validation AUROC; ties retain the earliest.
  Select the configuration with highest mean of the three best validation
  AUROCs; configuration ties retain the first in the grid order.
- This uses validation repeatedly for tuning and checkpoint selection; the
  reported validation scores are selected estimates, not independent test evidence.
- After selection, reload the three selected checkpoints and calculate validation
  and test metrics with the same evaluation function. Do not retrain on train+dev.
  Test labels are used only for evaluation, never gradients, feature preprocessing,
  checkpoint selection, model ranking for case analysis, or hyperparameter search.
- Report mean and **sample** standard deviation (`ddof=1`) of per-seed AUROCs.
  Seed averaging is a summary, not an ensemble. AUROC always uses continuous
  hateful-class probabilities, not the 0.5 decisions. Accuracy and the five-example
  error analysis alone use the fixed threshold 0.5 (`score >= 0.5` is hateful).
- CPU classifier training uses fixed Python, NumPy, and PyTorch seeds, an explicit
  seeded shuffle generator, one CPU thread, and deterministic algorithms.
  GPU embedding extraction is fixed but bitwise equality across hardware/library
  versions is not promised. Reuse the caches when exact local replay matters.

## Outputs

- `artifacts/features/<encoder>/<split>.npz`: embeddings, row IDs, checkpoint and
  preprocessing metadata, input checksum, extraction device/time, truncation count.
- `artifacts/checkpoints/<encoder>/seed_<seed>.pt`: linear weight/bias, standardization
  statistics, seed, chosen epoch, checkpoint identity, training settings, data hashes.
- `artifacts/checkpoints/<encoder>/selection.json`: complete six-setting search,
  every epoch's training loss/validation AUROC, selected settings and epochs.
- `artifacts/predictions/<encoder>/<split>_seed_<seed>.csv`: ID, true label, continuous
  hateful probability, and 0.5 prediction. Test labels in this output are mirrored labels.
- `artifacts/metrics/<encoder>.json`, `artifacts/results.json`, and
  `artifacts/results.csv`: per-seed evaluation and aggregate results.
- `artifacts/analysis.json`: strongest image/text encoder selected by mean validation
  AUROC, number of disagreements, and five cases ordered by disagreement first,
  descending absolute score gap, then ascending ID. Scores are from **seed 42**.
- `artifacts/error_analysis.json`: the same cases with notes grounded in visual
  inspection; these are interpretation hypotheses, not feature-attribution evidence.
- `report.tex`: self-contained, two-page report. Open it in a LaTeX editor for
  compilation and PDF preview; no external figure or bibliography files are needed.
- `output/pdf/report.pdf`: the exported, visually checked two-page PDF.
- `output/presentation/hateful_memes_results_v2.pptx`: the final presentation.

If a local TeX installation is available, export the PDF with two passes so all
bibliography references resolve:

```sh
mkdir -p output/pdf
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf report.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=output/pdf report.tex
```

Checkpoints can be loaded without encoder downloads to score cached features:

```python
import numpy as np
import torch
from baselines import score_checkpoint

checkpoint = torch.load("artifacts/checkpoints/clip/seed_42.pt", weights_only=True)
with np.load("artifacts/features/clip/test.npz", allow_pickle=False) as cached:
    scores = score_checkpoint(checkpoint, cached["features"])
```

Image files and pretrained encoder weights are not redistributed as part
of the code submission. Include the supplied images locally; checkpoints are
retrieved from their exact identifiers. Feature caches are reproducible and may
be omitted from submission; include the small trained classifier checkpoints.

## Sources and acknowledgments

- Kiela et al. (2020), [The Hateful Memes Challenge](https://arxiv.org/abs/2005.04790).
  Retain the supplied `data/LICENSE.txt`. If publishing example meme images, use
  the attribution requested in `data/README.md`.
- [TorchVision ResNet-50](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet50.html)
  for the pretrained CNN, weights, and preprocessing implementation.
- [CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32) and
  Radford et al. (2021), [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020).
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  and [MPNet model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2).
  Masked mean pooling and L2 normalization follow their published Transformers examples.
- [Hugging Face Transformers](https://github.com/huggingface/transformers),
  [PyTorch](https://pytorch.org), and [scikit-learn](https://scikit-learn.org)
  supply model loading, optimization, standardization, and evaluation.
- Test-annotation mirrors and the research source are cited above.
