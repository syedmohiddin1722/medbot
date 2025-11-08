from flask import Flask, render_template, request, jsonify, session
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from datetime import datetime

app = Flask(__name__)
app.secret_key = "medbot_secret_key"  # for session-based memory

# --------------------------------------------------
# ⚙️ Load the merged MedBot model
# --------------------------------------------------
MODEL_PATH = "./flan_medbot_merged"
print(f"🔹 Loading model from {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()


# --------------------------------------------------
# 🧠 Core function — factual medical response generator
# --------------------------------------------------
def generate_answer(user_input):
    """Generate an answer using session memory for factual context."""
    # Retrieve chat history
    chat_history = session.get("chat_history", [])

    # Build recent context (limit to last 4 turns)
    context = ""
    for turn in chat_history[-4:]:
        context += f"User: {turn['user']}\nMedBot: {turn['bot']}\n"

    # Define system role — polite, medical tone
    system_prompt = (
        "You are MedBot, a compassionate, precise, and trustworthy AI medical assistant.\n"
        "Your goal is to provide medically accurate, evidence-based, and concise explanations.\n"
        "Avoid humor, emotions, speculation, or sarcasm. "
        "If uncertain, say you cannot determine that precisely and suggest seeing a doctor.\n\n"
    )

    # Final prompt with context
    prompt = f"{system_prompt}{context}User: {user_input}\nMedBot:"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    # Generate factual answer
    outputs = model.generate(
        **inputs,
        max_new_tokens=220,
        temperature=0.4,       # lower = more factual, less creative
        top_p=0.85,
        top_k=40,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        length_penalty=1.0,
        early_stopping=True,
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Clean any unintended prefixes
    if "MedBot:" in answer:
        answer = answer.split("MedBot:")[-1].strip()

    # Prevent generic filler responses
    banned_phrases = [
        "I'm sorry", "As an AI", "I am not a doctor", "cannot provide medical advice"
    ]
    for phrase in banned_phrases:
        if phrase.lower() in answer.lower():
            answer = "Based on available medical information, it’s best to consult a healthcare professional for an accurate diagnosis."

    # Store chat in session
    chat_history.append({"user": user_input, "bot": answer})
    session["chat_history"] = chat_history

    return answer


# --------------------------------------------------
# 🌐 Flask Routes
# --------------------------------------------------
@app.route("/")
def index():
    # Reset chat each new session
    session["chat_history"] = []
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form["question"].strip()
    answer = generate_answer(user_question)
    timestamp = datetime.now().strftime("%I:%M %p")
    return jsonify({"answer": answer, "timestamp": timestamp})


if __name__ == "__main__":
    app.run(debug=True)
