from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/qualtrics-response", methods=["POST"])
def qualtrics_response():
    user_text = request.form.get("prompt", "")
    # Just echo back what we got
    return jsonify({"reply": f"ECHO FROM SERVER: {user_text}"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
