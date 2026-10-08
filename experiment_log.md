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
