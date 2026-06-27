# =====================================================================
# Task 4: General Health Query Chatbot (Prompt Engineering)
# =====================================================================

import requests
HF_TOKEN = "API TOKEN KEY"
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# System Prompt Engineering Design Layout
SYSTEM_PROMPT = """You are a helpful and friendly medical information assistant.
Your job is to answer general health questions in simple, clear language.

IMPORTANT RULES:
1. Always recommend seeing a doctor for serious symptoms.
2. Never diagnose diseases or prescribe medications.
3. Keep answers short (3-5 sentences).
4. If a question is dangerous or inappropriate, politely decline.
Always end with: 'Please consult a healthcare professional for advice.'"""


def ask_health_question(user_question):
    # Core Keyword Safety Layer Validation Checklist
    blocked_keywords = ['overdose', 'self-harm', 'suicide', 'illegal drug']
    for keyword in blocked_keywords:
        if keyword.lower() in user_question.lower():
            return "I cannot assist with that query. Please contact a healthcare professional or an emergency helpline immediately."

    # Structure proper execution template sequence format
    prompt = f"[INST] {SYSTEM_PROMPT}\n\nUser question: {user_question} [/INST]"

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 250, "temperature": 0.4}
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        # Parse text output cleanly past instructions tags boundaries
        clean_text = result[0]['generated_text'].split('[/INST]')[-1].strip()
        return clean_text
    else:
        return f"Error {response.status_code}: {response.text}"


# Automated script execution batch test sequence validation array
print("--- Automated Evaluation Test Verification ---")
questions = [
    'What causes a sore throat?',
    'Is paracetamol safe for children?',
    'How much water should I drink per day?'
]

for q in questions:
    print(f'\nQ: {q}')
    print(f'A: {ask_health_question(q)}')
    print('-' * 60)

# Optional Deployment Interactive Runtime Environment Loop Execution Toggle
# To run live chat, simply uncomment the execution framework block layout below:
"""
print("\nHealth Query Chatbot System initialized. Type 'quit' to terminate.")
while True:
    user_input = input('You: ').strip()
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print('Goodbye! Stay healthy!')
        break
    if not user_input:
        continue
    print(f'Bot: {ask_health_question(user_input)}\n')
"""