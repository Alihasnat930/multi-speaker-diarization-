# Training Guide for Dental Voice Intelligence System

This guide covers fine-tuning the models for optimal performance in dental clinical settings.

## Overview

The system uses three main models that can be fine-tuned:

1. **ASR Model** - For accurate transcription of dental terminology
2. **Speaker Recognition** - For better dentist/patient differentiation
3. **SOAP Generator** - For generating high-quality clinical notes

## 1. Fine-tuning ASR for Dental Conversations

### Prerequisites

- Dataset of dental conversations with accurate transcripts
- Recommended: 50+ hours of audio for significant improvement
- Audio format: 16kHz, mono WAV files
- Transcripts: Plain text with dental terminology

### Dataset Structure

```
dental_asr_dataset/
├── train/
│   ├── audio/
│   │   ├── conversation_001.wav
│   │   ├── conversation_002.wav
│   │   └── ...
│   └── transcripts/
│       ├── conversation_001.txt
│       ├── conversation_002.txt
│       └── ...
├── valid/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Training Script

```python
#!/usr/bin/env python3
"""
Fine-tune SpeechBrain ASR model on dental conversations.
"""
import speechbrain as sb
from speechbrain.dataio.dataio import read_audio
import torch

# Configure training
hparams_file = 'configs/asr_dental_train.yaml'
run_opts = {"device": "cuda:0"}

# Create experiment
asr_brain = sb.Brain(
    modules={},
    opt_class=torch.optim.Adam,
    hparams=sb.load_hyperpyyaml(hparams_file),
    run_opts=run_opts
)

# Prepare data
train_data, valid_data, test_data = prepare_dental_dataset()

# Fine-tune
asr_brain.fit(
    epoch_counter=asr_brain.hparams.epoch_counter,
    train_set=train_data,
    valid_set=valid_data,
    train_loader_kwargs={"batch_size": 8},
    valid_loader_kwargs={"batch_size": 8}
)

# Save model
torch.save(asr_brain.modules.state_dict(), 'models/asr_dental_finetuned/model.ckpt')
```

### Quick Fine-tuning (Few-shot)

If you have limited data (<10 hours):

```bash
# Use SpeechBrain's recipes with pre-trained model
cd speechbrain/recipes/LibriSpeech/ASR/transformer

# Modify train.yaml to use pre-trained weights and your data
# Set number_of_epochs: 5 for quick adaptation

python train.py configs/dental_finetune.yaml
```

## 2. Fine-tuning SOAP Generator (LLM)

### Dataset Format

Create a JSONL file with conversation-SOAP pairs:

```json
{"conversation": "Dentist: How are you feeling?\nPatient: I have tooth pain...", "soap": {"subjective": "Patient reports...", "objective": "Examination reveals...", "assessment": "Diagnosis...", "plan": "Treatment plan..."}}
{"conversation": "...", "soap": {...}}
```

### Using LoRA for Efficient Fine-tuning

```python
#!/usr/bin/env python3
"""
Fine-tune Mistral/Llama for SOAP note generation using LoRA.
"""
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Load base model
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    torch_dtype=torch.float16
)

# Configure LoRA
lora_config = LoraConfig(
    r=16,                       # LoRA rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Attention layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = load_dataset('json', data_files='dental_soap_dataset.jsonl')

def format_prompt(example):
    """Format conversation as instruction prompt"""
    conv = example['conversation']
    soap = example['soap']
    
    soap_text = f"""SUBJECTIVE: {soap['subjective']}
OBJECTIVE: {soap['objective']}
ASSESSMENT: {soap['assessment']}
PLAN: {soap['plan']}"""
    
    prompt = f"""<s>[INST] You are a dental clinical documentation assistant. Generate a SOAP note from this conversation:

{conv}

[/INST] {soap_text}</s>"""
    
    return {"text": prompt}

# Format dataset
dataset = dataset.map(format_prompt)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/dental_soap_lora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation']
)

trainer.train()

# Save LoRA weights
model.save_pretrained("./models/dental_soap_lora")
tokenizer.save_pretrained("./models/dental_soap_lora")
```

### Dataset Creation Tips

**Good SOAP examples:**

```json
{
  "conversation": "Dentist: What brings you in today?\nPatient: I've had this throbbing pain in my back tooth for three days\nDentist: Which tooth exactly?\nPatient: Upper right, the one in the back\nDentist: Let me examine. I see decay on tooth 15. Does it hurt when I tap?\nPatient: Yes, very much\nDentist: You have a deep cavity that's reached the nerve. We need a root canal\nPatient: How long will that take?\nDentist: About 90 minutes today, then a crown in two weeks",
  
  "soap": {
    "subjective": "Patient reports 3-day history of throbbing pain in upper right posterior tooth. Pain is severe and worsens with pressure.",
    
    "objective": "Clinical examination reveals deep occlusal caries on tooth #15 (upper right second molar). Positive response to percussion test. No visible swelling or abscess. X-ray shows caries extending to pulp chamber.",
    
    "assessment": "Irreversible pulpitis, tooth #15, secondary to deep caries. Requires endodontic treatment.",
    
    "plan": "Perform root canal therapy on tooth #15 today under local anesthesia. Prescribe amoxicillin 500mg TID for 7 days and ibuprofen 400mg for pain. Schedule crown preparation in 2 weeks. Patient educated on procedure and gave informed consent."
  }
}
```

**Recommended dataset size:**
- Minimum: 100 conversation-SOAP pairs
- Good: 500+ pairs
- Excellent: 1000+ pairs

### Data Augmentation

Use GPT-4 or Claude to generate synthetic training data:

```python
# Generate synthetic dental conversations and SOAPs
# Use this only to augment real data, not replace it

import anthropic

def generate_training_pair():
    prompt = """Generate a realistic dental consultation conversation 
    and corresponding SOAP note. Include:
    - Patient complaint
    - Dentist examination
    - Clinical findings
    - Treatment plan
    
    Format as JSON with 'conversation' and 'soap' keys."""
    
    # Call API to generate...
    # Then review and curate manually
```

## 3. Speaker Recognition Fine-tuning

For better speaker discrimination in your specific clinic:

```python
#!/usr/bin/env python3
"""
Fine-tune speaker recognition on your clinic's audio.
"""
from speechbrain.processing.features import MFCC
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
import torch

# Load pre-trained ECAPA-TDNN
model = ECAPA_TDNN.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec"
)

# Prepare your clinic's speaker samples
# Structure:
# clinic_speakers/
#   dentist_1/
#     sample_001.wav
#     sample_002.wav
#   dentist_2/
#     ...
#   patient_samples/
#     ...

# Fine-tune with metric learning
# (Contrastive loss to improve separation)

# Save fine-tuned model
torch.save(model.state_dict(), 'models/spkrec_clinic_finetuned.ckpt')
```

## 4. Evaluation

### ASR Evaluation

```python
from jiwer import wer, cer

# Calculate Word Error Rate
ground_truth = ["the patient has a cavity in the upper molar"]
hypothesis = ["the patient has a cavity in the upper molar"]

error_rate = wer(ground_truth, hypothesis)
print(f"Word Error Rate: {error_rate:.2%}")
```

### SOAP Quality Evaluation

Manual review is essential. Use rubric:

| Criteria | Score (1-5) |
|----------|-------------|
| Subjective completeness | |
| Objective accuracy | |
| Assessment correctness | |
| Plan appropriateness | |
| Dental terminology usage | |

### Speaker ID Evaluation

```python
# Calculate Equal Error Rate (EER)
from sklearn.metrics import roc_curve

# Get embeddings for test set
# Calculate similarities
# Plot ROC curve
```

## 5. Continuous Improvement

### Active Learning Pipeline

1. **Collect real consultations** (with consent)
2. **Manual review** - Clinician reviews and corrects transcripts/SOAPs
3. **Add to training set** - Augment dataset with validated examples
4. **Periodic re-training** - Monthly or quarterly fine-tuning
5. **A/B testing** - Compare old vs new model performance

### Monitoring

Track these metrics in production:

```python
# Log to database
metrics = {
    'timestamp': datetime.now(),
    'wer': 0.05,                  # Word error rate
    'speaker_accuracy': 0.98,      # Speaker ID accuracy
    'soap_review_rate': 0.15,      # % SOAPs requiring edits
    'latency_p95': 450,            # 95th percentile latency (ms)
}
```

## 6. Quick Start Training Scripts

We provide ready-to-use training scripts:

```bash
# ASR fine-tuning
python scripts/train_asr.py \
  --dataset /path/to/dental_conversations \
  --base_model speechbrain/asr-transformer-transformerlm-librispeech \
  --output models/asr_dental_finetuned \
  --epochs 10 \
  --batch_size 8

# SOAP generator fine-tuning
python scripts/train_soap_generator.py \
  --dataset dental_soap_dataset.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --output models/dental_soap_lora \
  --lora_r 16 \
  --epochs 3 \
  --learning_rate 2e-4

# Speaker recognition fine-tuning
python scripts/train_spkrec.py \
  --dataset clinic_speakers/ \
  --base_model speechbrain/spkrec-ecapa-voxceleb \
  --output models/spkrec_clinic_finetuned \
  --epochs 20
```

## 7. Pre-trained Checkpoints

After training, organize models:

```
models/
├── asr_dental_finetuned/
│   ├── model.ckpt
│   ├── tokenizer/
│   └── config.yaml
├── dental_soap_lora/
│   ├── adapter_model.bin
│   └── adapter_config.json
├── spkrec_clinic_finetuned.ckpt
└── enrollments/
    ├── Dentist.pkl
    └── Patient.pkl
```

Update configs to use fine-tuned models:

```python
# app/asr_service.py
model_source = "./models/asr_dental_finetuned"

# app/summarizer_local.py
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
lora_path = "./models/dental_soap_lora"
```

## Support

For training assistance, see:
- SpeechBrain documentation: https://speechbrain.readthedocs.io/
- Hugging Face PEFT: https://huggingface.co/docs/peft/
- Contact: training-support@yourorg.com
