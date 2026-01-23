# Distil-Whisper Training Guide

Complete guide for creating and training distilled Whisper models with reduced encoder/decoder layers.

---

## Overview

**Goal**: Create smaller, faster Whisper models while maintaining quality through knowledge distillation.

### Recommended Architecture: 16 Encoder + 2 Decoder

| Model | Encoder | Decoder | Size (BF16) | Use Case |
|-------|---------|---------|-------------|----------|
| Teacher | 32 layers | 4 layers | ~1.5 GB | Source (frozen) |
| **Student (recommended)** | **16 layers** | **2 layers** | **~0.4 GB** | **Production** |
| Student (current) | 32 layers | 2 layers | ~0.8 GB | High quality |
| Student (smallest) | 8 layers | 2 layers | ~0.3 GB | Edge devices |

---

## Step 1: Create Student Model (16 Encoder + 2 Decoder)

```bash
cd /workspace/STT/dis-hamsa

python distil-whisper/training/create_student_model.py \
  --teacher_checkpoint nadsoft/Best_ASR_Model_s2 \
  --encoder_layers 16 \
  --decoder_layers 2 \
  --save_dir ./hamsa-distil-16enc-2dec \
  --cache_dir /workspace/dis-cache
```

**Expected Output:**
```
============================================================
STUDENT MODEL SUMMARY
============================================================
Encoder layers: 16
Decoder layers: 2
Total parameters: ~400,000,000 (400.0M)
Estimated size (FP16/BF16): ~0.40 GB
============================================================
```

---

## Step 2: Run Distillation Training

**Note:** Encoder is NOT frozen because we reduced it (32→16). It needs to adapt.

```bash
accelerate launch distil-whisper/training/run_distillation_multilingual.py \
  --model_name_or_path "./hamsa-distil-16enc-2dec" \
  --teacher_model_name_or_path "nadsoft/Best_ASR_Model_s2" \
  --train_dataset_path "/workspace/STT/Stage2/data/normalize_train" \
  --eval_dataset_path "/workspace/STT/data/general_normalized_test" \
  --text_column_name "normalized_text" \
  --eval_text_column_name "normalized_text" \
  --cache_dir "/workspace/dis-cache" \
  --push_to_hub \
  --hub_model_id "nadsoft/hamsa-distil-16enc-2dec" \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --eval_steps 2000 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --warmup_steps 1000 \
  --max_steps 80000 \
  --output_dir "./hamsa-distil-16enc-2dec-trained" \
  --gradient_checkpointing \
  --dtype "bfloat16" \
  --logging_steps 25 \
  --save_steps 2000 \
  --attn_implementation "sdpa" \
  --report_to "tensorboard"
```

---

## Step 3: Resume from Checkpoint (if training stops)

Same command (will auto-detect checkpoint):

```bash
accelerate launch distil-whisper/training/run_distillation_multilingual.py \
  --model_name_or_path "./hamsa-distil-16enc-2dec" \
  --teacher_model_name_or_path "nadsoft/Best_ASR_Model_s2" \
  --train_dataset_path "/workspace/STT/Stage2/data/normalize_train" \
  --eval_dataset_path "/workspace/STT/data/general_normalized_test" \
  --text_column_name "normalized_text" \
  --eval_text_column_name "normalized_text" \
  --cache_dir "/workspace/dis-cache" \
  --do_train \
  --do_eval \
  --predict_with_generate \
  --eval_steps 2000 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 16 \
  --learning_rate 1e-5 \
  --warmup_steps 1000 \
  --max_steps 80000 \
  --output_dir "./hamsa-distil-16enc-2dec-trained" \
  --gradient_checkpointing \
  --dtype "bfloat16" \
  --logging_steps 25 \
  --save_steps 2000 \
  --attn_implementation "sdpa" \
  --report_to "tensorboard"
```

---

## Step 4: View TensorBoard Logs

```bash
tensorboard --logdir ./hamsa-distil-16enc-2dec-trained/runs --port 6006
```

Open: `http://localhost:6006`

---

## Key Training Settings

| Setting | Value | Why |
|---------|-------|-----|
| `--freeze_encoder` | **NOT used** | Encoder reduced, needs training |
| `--per_device_train_batch_size` | 32 | Lower (training encoder uses more memory) |
| `--learning_rate` | 1e-5 | Higher LR for full model training |
| `--max_steps` | 80000 | More steps needed |
| `--warmup_steps` | 1000 | Longer warmup for stability |

---

## Dataset Format

Parquet files with columns:
- `audio`: Audio data
- `normalized_text`: Transcription
- `label`: Language (arabic_only, english_only, etc.)

---

## Language Mapping

```python
LANGUAGE_MAP = {
    'arabic_only': 'ar',
    'english_only': 'en',
    'no_language': 'ar',
    'arabic': 'ar',
    'english': 'en',
}
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| No space left | `rm -rf ~/.cache/huggingface/datasets/*` |
| Flash attention error | Use `--attn_implementation "sdpa"` |
| OOM (out of memory) | Reduce batch size to 16 |
| Resume training | Don't use `--overwrite_output_dir` |

---

## Files

| File | Purpose |
|------|---------|
| `create_student_model.py` | Create smaller student |
| `run_distillation_multilingual.py` | Train with KL-divergence |
