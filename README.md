# The Manga Task

A system that reads three consecutive English manga pages, extracts the story
text in reading order, and labels who says each line — keeping the same
character name consistent across all three pages.

Built with **Qwen2-VL-7B-Instruct (4-bit) + LoRA**, fine-tuned on the task's
development set, and run on the 15 test sequences.

---

## How it works

The task is to take 3 consecutive manga pages and output, for each page, the
story text in reading order with a speaker label per line. Simple to describe,
tricky in practice — especially getting the *same* character labelled the same
way across all three pages.

The pipeline went through four stages:

**1. Baseline.** Started with Qwen2-VL-7B in 4-bit, asking it to read one page
at a time with a detailed prompt (right-to-left reading order, what counts as
text, what to skip). This worked okay for reading the text but was bad at
speaker consistency — each page was processed in isolation, so the model had no
way to know that `character1` on page 1 is the same person as `character1` on
page 3.

**2. Design choice.** The obvious fix is to feed all three pages together. I
measured it: it needs ~19 GB, which doesn't fit on the 12 GB card. So I went
with a per-page setup instead — each page is processed alone, but pages 2 and 3
get a short text block containing the earlier pages' extracted JSON. The model
learns to reuse the same labels from that context.

**3. Fine-tune.** LoRA (r=16) on the language-model layers of Qwen2-VL, 4-bit
base, ~54 minutes to train 3 epochs on one GPU. Loss only on the answer tokens.

**4. Inference.** Same prompt and context design as training, plus two small
safety nets: collapse repeated lines, and if the output looks like a repetition
loop, regenerate once with a mild repetition penalty. A format check runs before
the submission is written.

Full details, all score tables, failure modes, and dead ends are in
[`experiment_log.md`](experiment_log.md). That's the authoritative record —
this README is just the summary.

---

## Results

Held-out validation, 16 sequences (macro average):

| Metric                     | Baseline | Fine-tuned     |
| -------------------------- | -------- | -------------- |
| `text_order_score`       | 0.52     | **0.85** |
| `balanced_joint_f1`      | 0.17     | **0.35** |
| `matched_token_coverage` | 0.62     | **0.90** |

Text extraction improved a lot. Speaker consistency is better but is still the
weakest part — that's the main thing to push on next.

---

## Repository layout

byte-applied-ai-manga-task-2026/
├── README.md
├── experiment_log.md # full write-up with all numbers
├── pyproject.toml # dependencies
├── uv.lock
├── .gitignore
│
├── dataset/
│ ├── sequences.json # which 3 images belong to each sequence
│ ├── score.py # provided scorer
│ ├── sample_submission.jsonl # 15 test IDs in submission format
│ ├── development/ # dev images + labels.jsonl
│ └── test/ # 15 test sequences
│
├── notebook/
│ ├── 01_baseline.ipynb # zero-shot baseline
│ ├── 02_finetune.ipynb # LoRA training + validation
│ └── 03_inference.ipynb # final test inference
│
├── adapters/lora_v1/ # saved LoRA adapters (epoch_1, 2, 3)
├── logs/ # training loss logs
└── outputs/ # predictions + scores for each run

### What each folder contains

| Folder                  | Contents                                                                                                                                                                                                                                                                                                                                              |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`dataset/`**  | All task data and the provided scorer. `sequences.json` maps each sequence to its 3 image paths; `development/` holds the 80 labelled dev sequences with `labels.jsonl`; `test/` holds the 15 unlabelled test sequences; `sample_submission.jsonl` lists the 15 test IDs in submission format; `score.py` is the provided scoring script. |
| **`notebook/`** | The three notebooks that make up the full pipeline, run in order: baseline → fine-tune → inference.                                                                                                                                                                                                                                                 |
| **`adapters/`** | Saved LoRA adapter weights from fine-tuning. `lora_v1/` holds `epoch_1/`, `epoch_2/` and `epoch_3/` — one per training epoch. Epoch 3 is used for the final test run.                                                                                                                                                                        |
| **`logs/`**     | Training logs. `finetune_v2_perpage.jsonl` records train loss, val loss and elapsed time after each epoch.                                                                                                                                                                                                                                          |
| **`outputs/`**  | Predictions and raw model output from every run: `baseline/` (zero-shot dev predictions), `finetuned_v1/` (epoch-2 validation predictions), `finetuned_v1_epoch3/` (epoch-3 validation predictions) and `test_final/` (the final submission).                                                                                                 |

### What each notebook does

| Notebook                         | Purpose                                                                                                                                                                                                                                                                     |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`01_baseline.ipynb`**  | Loads Qwen2-VL-7B in 4-bit and runs zero-shot inference on all 80 development sequences, one page at a time. Covers dataset exploration, the prompt, repetition handling, and the loop that saves dev predictions to `outputs/baseline/`.                                 |
| **`02_finetune.ipynb`**  | The training notebook. Runs a memory test comparing joint 3-page vs per-page setups, builds the per-page dataset with prior-page context, attaches LoRA, trains for 3 epochs, and saves adapters to `adapters/lora_v1/`. Also runs validation predictions for each epoch. |
| **`03_inference.ipynb`** | Loads the epoch-3 adapter and runs the 15 test sequences using the same prompt and context design as training. Applies loop/truncation guards, validates the output format, and writes the final submission to `outputs/test_final/submission.jsonl`.                     |

---

## How to run

**Requirements:** Python ≥3.12, a CUDA GPU with ~12 GB VRAM, and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed.

### 1. Install dependencies

```bash
git clone https://github.com/RahulSingh-X01/byte-applied-ai-manga-task-2026.git
cd byte-applied-ai-manga-task-2026
uv sync
```

This installs everything from `pyproject.toml` / `uv.lock`, including PyTorch from the CUDA 12.8 index.

### 2. Download the LoRA adapters

The trained adapter weights are hosted on Hugging Face (too large to commit to the repo):

**https://huggingface.co/RahulSingh-X01/manga-lora-epochs-adapters**

The repo contains the three epoch folders (`epoch_1/`, `epoch_2/`, `epoch_3/`). Download them into `adapters/lora_v1/`:

```bash
uv run huggingface-cli download RahulSingh-X01/manga-lora-epochs-adapters \
  --local-dir adapters/lora_v1
```

After this, `adapters/lora_v1/` should contain `epoch_1/`, `epoch_2/` and `epoch_3/`. Epoch 3 is used for the final test run.

### 3. Launch Jupyter

```bash
uv run jupyter notebook
```

Open the `notebook/` folder. The notebooks use relative paths, so run them with `notebook/` as the working directory.

### 4. Run the pipeline in order

| Step | Notebook               | What it does                                                               |
| ---- | ---------------------- | -------------------------------------------------------------------------- |
| 1    | `01_baseline.ipynb`  | Zero-shot Qwen2-VL on the 80 dev sequences, saves to `outputs/baseline/` |
| 2    | `02_finetune.ipynb`  | Trains LoRA for 3 epochs, saves adapters to `adapters/lora_v1/`          |
| 3    | `03_inference.ipynb` | Runs the 15 test sequences, writes `outputs/test_final/submission.jsonl` |

Tip: in `02_finetune.ipynb`, set `DEBUG_STEPS = 4` for a quick memory check before the full run. In `03_inference.ipynb`, keep `OVERWRITE = False` unless you want to regenerate predictions.

### 5. Score predictions

```bash
uv run python dataset/score.py \
  --references dataset/development/labels.jsonl \
  --predictions <your_predictions.jsonl> \
  --output <scores.json>
```

Headline metrics: `text_order_score` and `balanced_joint_f1`. Full details in `experiment_log.md`.

## Conclusion

The baseline showed that reading text is easy — keeping speaker labels
consistent across pages is the hard part. Fine-tuning Qwen2-VL-7B with LoRA on
a per-page setup with prior-page context fixed most of that: `text_order_score`
rose from 0.52 to 0.85 and `matched_token_coverage` from 0.62 to 0.90, with
speaker consistency (0.35) still the weakest point and the clear next target.
All experiments, including the failures, are documented in
`experiment_log.md`.

---

## Contribution: what I did vs what AI helped with

### What I did (all key decisions and experimentation)

- Designed the overall project approach and repo structure.
- Chose Qwen2-VL-7B-Instruct as the base model.
- Chose 4-bit NF4 quantization to fit the model on a 12 GB GPU.
- Decided to actually test the 3-page joint setup before committing to it, and
  designed and ran the memory test comparing joint 3-page vs per-page setups
  (peak memory, tokens, time per example). Based on the results, made the call
  to go per-page with prior-page text context when the joint setup did not fit.
- Designed the LoRA configuration (target modules, rank, alpha, dropout).
- Designed the training setup: split strategy, teacher forcing with earlier
  pages' labels as context, loss masking on answer tokens only.
- Diagnosed baseline failure modes (repetition loops, invented speakers,
  ignored exclusions) and decided how to handle each one.
- Made the call to use the scorer over val loss when picking the epoch.
- Ran all experiments, inspected outputs, and interpreted the results.

### What AI helped with

- **Boilerplate code:** data loading, sequence/image iteration, sanity-report
  cells, format-check cells.
- **Model loading, 4-bit quantization and LoRA attachment code.** I understand
  these components well — I chose the quantization scheme (NF4 with double
  quant and bf16 compute), the LoRA target modules, rank, alpha and dropout,
  and the gradient checkpointing setup — but for the implementation I leaned on
  AI to save time. My focus was on deciding *what* to train and *why* rather
  than retyping the HuggingFace boilerplate, and I verified every run by
  inspecting outputs and losses rather than trusting the code blindly.
- **README and `experiment_log.md`** were drafted from my own notes. I wrote the
  content myself and used AI for grammar, spelling, and converting it into
  clean structured markdown.
- **Debugging help:** understanding error messages and suggesting parameter
  names.

All design decisions, experiment choices, and analysis are mine. No hosted AI
inference APIs were used to produce training targets or test predictions.
