# =====================================================================
# Task 5: Mental Health Support Chatbot (Fine-Tuning Architecture Setup)
# =====================================================================

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
import torch

print("GPU Available Device Verification:", torch.cuda.is_available())

# Load empathetic dialogue structures dataset matrix data frames
dataset = load_dataset('empathetic_dialogues')
print("Dataset structure:")
print(dataset)

# Initialize Tokenizer and Pre-Trained Model instances matching types
model_name = 'distilgpt2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Configure necessary properties tokens constraints
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Preprocessing Mapping Function
def preprocess(example):
    text = f"User: {example['utterance']}\nTherapist: {example['utterance']}"
    return tokenizer(text, truncation=True, max_length=128, padding='max_length')

# Select mini testing scale slice tracking subset context
small_train = dataset['train'].select(range(1000))
tokenized = small_train.map(preprocess, remove_columns=small_train.column_names)
tokenized = tokenized.add_column('labels', tokenized['input_ids'])

# Fine-Tuning Hyperparameters Configuration Framework Block
training_args = TrainingArguments(
    output_dir='./mental_health_chatbot',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,  # Optimizes compute footprint natively using Colab T4 hardware limits
    report_to='none'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized
)

print("\nStarting Model Fine-Tuning Train Optimization Phase...")
trainer.train()
print("Fine-tuning complete!")

# Inference Generation Pipeline Engineering Pipeline Module
chatbot = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0)

def mental_health_response(user_message):
    prompt = f"User: {user_message}\nTherapist:"
    result = chatbot(prompt, max_new_tokens=80, do_sample=True,
                     temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    generated = result[0]['generated_text']
    return generated.split('Therapist:')[-1].strip()

# Final Evaluation Pipeline Tracking Verification Probe Test Check
test_inputs = [
    'I feel really anxious today.',
    'I am stressed about work and cannot sleep.'
]

print("\n--- Model Inference Sample Generations ---")
for msg in test_inputs:
    print(f'User: {msg}')
    print(f'Bot: {mental_health_response(msg)}\n')