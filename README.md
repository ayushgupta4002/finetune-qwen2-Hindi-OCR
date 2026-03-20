# Finetune Qwen2-VL-2B for Hindi OCR

Fine-tunes the [Qwen2-VL-2B-Instruct](https://huggingface.co/unsloth/Qwen2-VL-2B-Instruct) vision-language model on a Hindi OCR dataset using [Unsloth](https://github.com/unslothai/unsloth) and LoRA adapters.

## Dataset

Uses the [`damerajee/hindi-ocr`](https://huggingface.co/datasets/damerajee/hindi-ocr) dataset from Hugging Face — image/text pairs where the model learns to transcribe Hindi text from images.

## Setup

```bash
pip install unsloth transformers==5.3.0 trl==0.22.2 datasets sentencepiece protobuf
```

> Restart the kernel after installation before running the training script.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `unsloth/Qwen2-VL-2B-Instruct` |
| LoRA rank (`r`) | 16 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Max steps | 40 |
| Learning rate | 3e-4 |
| Optimizer | `adamw_8bit` |
| Max sequence length | 2048 |

All vision, language, attention, and MLP layers are fine-tuned.

## Usage

**Train:**
```bash
python finetune_qwen2_vl_2b_hindi_ocr.py
```

Checkpoints are saved to `./outputs/`.

**Inference** (included at the bottom of the script):
```python
image = dataset[0]["image"]
instruction = "Write the Hindi representation for this image."
# tokenize and generate...
```

## Notes

- Designed to run on Kaggle (T4 GPU) or any CUDA-enabled environment.
- `load_in_4bit = False` — full precision LoRA, not quantized.
- Gradient checkpointing enabled via Unsloth's custom implementation for memory efficiency.
