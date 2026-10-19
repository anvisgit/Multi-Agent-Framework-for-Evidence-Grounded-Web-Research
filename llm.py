import ollama
MODEL = "qwen2.5:1.5b"
def ask(prompt, model=MODEL):
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]