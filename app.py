import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

# Set your API key as an environment variable before running:
# export OPENAI_API_KEY="sk-..."
client = OpenAI()

@app.route("/qualtrics-response", methods=["POST"])
def qualtrics_response():
    # Qualtrics Web Service usually sends form-encoded params
    user_text = request.form.get("prompt")
    if not user_text:
        # Fallback if you later choose JSON
        data = request.get_json(silent=True) or {}
        user_text = data.get("prompt")

    if not user_text:
        return jsonify({"error": "No prompt provided"}), 400

    # Build the messages for ChatGPT
    messages = [
        {
            "role": "system",
            "content": (
                "You are an empathetic but non-clinical AI. "
                "Read the person's experience and respond in a supportive, "
                "respectful way in about 150–200 words. "
                "Do NOT give medical advice or diagnosis."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
    )

    reply = completion.choices[0].message.content

    # This JSON shape is important for Qualtrics mapping
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)