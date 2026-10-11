# Experiment Log

## Baseline v1: zero-shot Qwen2-VL-7B-Instruct

**Date:** 2026-10-08
**Goal:** Establish a reference score before fine-tuning.

### Configuration

- Model: Qwen/Qwen2-VL-7B-Instruct, 4-bit NF4 quantization (bitsandbytes), RTX 5070 Ti 12GB
- Inference: one call per page (the 3 pages of a sequence are processed independently)
- Decoding: greedy (`do_sample=False`), `max_new_tokens=1536`, no repetition penalty
- Prompt: detailed INCLUDE / EXCLUDE prompt with speaker-label rules (saved in notebook)
- Parser: regex extraction of complete `{...}` objects, speaker casing fix (UNKNOWN/NARRATION),
  loop guard that stops at the first duplicate/overlapping text
- Run time: 37.4 min for 80 dev sequences (240 pages)
- Output files: outputs/baseline_predictions.jsonl, baseline_raw.jsonl, baseline_scores.json

### Results (macro average over 80 dev sequences, missing_sequences = [])

| Metric                      | Score  |
| --------------------------- | ------ |
| text_order_score            | 0.4951 |
| balanced_joint_f1           | 0.1683 |
| speaker_accuracy_on_matched | 0.4448 |
| joint_f1                    | 0.2771 |
| matched_token_coverage      | 0.6006 |

### Problems found while building the baseline (and fixes)

1. **JSON cut off.** `max_new_tokens=1024` was too small. Raised it.
2. **Model output "..." repeatedly / garbage keys.** Unclear cause at the time. I changed
   several things at once (pixel limits, repetition penalty, prompt), so I can't attribute it
   to one change. Lesson: change one variable at a time.
3. **Repetition loops** (the same narration sentence repeated until the token limit).
   - `repetition_penalty=1.5` broke the JSON schema and spelling.
   - `repetition_penalty=1.1` stopped loops but degraded speaker labels.
   - `no_repeat_ngram_size=8` produced invented labels (COACH, NARRATIVE...).
   - **Final fix:** no generation constraints, dedup at the parsed-object level.
4. **Overly strict prompt returned `[]`** for a page that has text.
5. **Resolution check:** a full page becomes about 1518 image tokens, so images were not
   being squashed.

### Observed failure modes (from inspecting seq_2032620aa4e4ac7f, one sequence)

- Real dialogue often labelled NARRATION.
- Invented speaker names taken from the dialogue ("OKAWA-SENPAI").
- Distinct characters collapsed into UNKNOWN.
- Reading order not matching the reference.
- Missed lines (small background text).
- Dataset-specific exclusions not followed: recap box on page 2 ("LAST TIME...") was
  extracted but the reference is empty; sound effects ("HOOFBEATS", "CLACK") and the
  decorative title were included.
- Known limitation of my parser: the loop guard would also drop a legitimate repeated line
  such as "..." appearing twice on one page.

### Interpretation

- Text recovery is moderate (matched_token_coverage 0.60, text_order_score 0.50).
- Speaker identity is the weakest part (balanced_joint_f1 0.17), which is expected because
  each page is processed independently, so labels cannot be consistent across pages.
- speaker_accuracy_on_matched (0.44) only counts lines that already matched, so it overstates
  speaker quality.

### Next

Fine-tune with LoRA on a train/validation split of the dev set (whole sequences kept together).
Targets: cross-page speaker consistency, line segmentation and order, dataset-specific exclusions.

## Fine-tuning v1: per-page LoRA with text context from earlier pages

**Date:** <fill></fill>11-10-2026
**Goal:** Fix the baseline's main weaknesses: speaker labels and consistency across pages,
line segmentation and order, and the dataset's exclusion conventions.

### Design decision: joint 3-page vs per-page (memory test)

| Setup                                                       | Image tokens | Input tokens | Peak memory        | Time per example |
| ----------------------------------------------------------- | ------------ | ------------ | ------------------ | ---------------- |
| 3 pages in one call, full resolution                        | 4,554        | 4,947        | 19.2 GB (two runs) | 51.6 s / 84.8 s  |
| 1 page + text context, full resolution (4 longest examples) | 1,518        | 1,890        | 11.9 GB            | 35.6 s           |

- The joint setup exceeds the card (about 11.9 GB), so the driver spilled into system RAM.
  I infer that from the peak being above VRAM capacity and from the slowdown.
- I did not downscale images, because small text is exactly what the model has to read.
  I did not test downscaling, so I can't say how much it would have hurt.
- **Decision:** one page per example. Pages 2 and 3 receive the labels of the earlier pages as
  text. This gives 240 page examples from 80 sequences (192 train / 48 val).
- **Known limitation:** the text context carries labels and dialogue but not what the
  characters look like, so matching a face on page 2 to `character1` from page 1 is only
  partly possible. A possible follow-up is to add the previous page as a downscaled second image.
- Data stats: answer tokens per page mean 141, max 411; longest context text 701 tokens.

### Setup

- Model: Qwen2-VL-7B-Instruct, 4-bit NF4, bf16 compute, gradient checkpointing
- LoRA: r=16, alpha=32, dropout=0.05, on q/k/v/o/gate/up/down projections of the language model
  (40,370,176 trainable params = 0.48% of 8.33B)
- Training: AdamW, lr 1e-4, cosine schedule with 10% warmup, batch size 1, grad accumulation 4,
  grad-clip 1.0, 3 epochs, seed 42
- Loss only on answer tokens (prompt and image tokens masked with -100)
- Split: 64 train / 16 val sequences, whole sequences kept together, seeded, saved in `split.json`
- Training context uses the ground-truth labels of earlier pages (teacher forcing).
  Inference uses the model's own predictions for earlier pages.
- TF32 flags were enabled. I did not measure any effect on memory.

### Training results

| Epoch | Train loss | Val loss | Elapsed  |
| ----- | ---------- | -------- | -------- |
| 1     | 0.2184     | 0.1509   | 19.9 min |
| 2     | 0.0988     | 0.1118   | 36.8 min |
| 3     | 0.0611     | 0.1126   | 53.8 min |

- Train loss kept falling while val loss stopped improving after epoch 2, a possible
  sign of mild overfitting. The epoch 2 vs 3 gap (0.0008) is tiny and based on 48 examples.
- Peak memory 12.2 to 12.4 GB, above the card's capacity, so some spill is likely.
- Total time was 53.8 min.
- Val loss is computed with ground-truth context, so it does not measure the error-compounding
  that happens at inference.

### Results on the 16 held-out validation sequences (macro average)

| Metric                      | Baseline (zero-shot) | Fine-tuned (epoch 2) | Delta  |
| --------------------------- | -------------------- | -------------------- | ------ |
| text_order_score            | 0.5239               | 0.8418               | +0.318 |
| balanced_joint_f1           | 0.1651               | 0.3611               | +0.196 |
| speaker_accuracy_on_matched | 0.4376               | 0.6221               | +0.185 |
| joint_f1                    | 0.2762               | 0.5636               | +0.287 |
| matched_token_coverage      | 0.6227               | 0.8943               | +0.272 |

Both columns use the same 16 sequences (baseline predictions filtered to those IDs).

### Interpretation

- All five metrics improved. Text recovery rose the most (order score 0.52 to 0.84,
  coverage 0.62 to 0.89).
- Speaker identity is still the weakest part: `balanced_joint_f1` is 0.36.
  Speaker consistency has room to improve.

### Caveats

- 16 validation sequences is small; differences of a few points are likely noise.
- The epoch was chosen by val loss, not by the scorer.
- Error propagation (a wrong page-1 prediction feeding pages 2 and 3) has not been measured.
- Scoring on all 80 dev sequences would include the 64 training sequences, so it would
  overestimate generalisation. The 16-sequence number is the held-out result.

### Problems encountered while setting up fine-tuning

1. GPU memory readings jumped around (5.68 GB vs 13.63 GB after "model load"). The 13.63 GB reading
   was most likely a second model copy loaded in the same kernel. Fix: restart the kernel
   before each real run.
2. Debug-run `SystemExit` is expected: it is how the memory test stops.

### Next

1. Score epoch 3 as well as epoch 2, to check whether val loss picked the better epoch.
