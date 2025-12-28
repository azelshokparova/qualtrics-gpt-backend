# app.py
from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)

# Create OpenAI client (expects OPENAI_API_KEY in Render env vars)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return "ok", 200

@app.route("/qualtrics-response", methods=["POST"])
def qualtrics_response():
    # Qualtrics typically sends application/x-www-form-urlencoded (request.form).
    # If you ever switch to JSON in Qualtrics, this will still work.
    user_text = (request.form.get("prompt") if request.form else None) or \
                (request.json.get("prompt") if request.is_json and request.json else "") or ""

    system_prompt = (
        "You are a compassionate expert therapist whose primary goal is to provide effective guidance while also showing emotional attunement and compassion."
        "Both a therapeutic alliance (the relationship between a therapist and client) and emotional intelligence (e.g., compassion) are important to therapy."
        "Compassion can be defined as a cognitive, affective, and behavioral process consisting of the five elements: 1) Recognizing suffering; 2) Understanding the universality of suffering in human experience; 3) Feeling empathy for the person suffering and connecting with the distress (emotional resonance); 4) Tolerating uncomfortable feelings aroused in response to the suffering person (e.g. distress, anger, fear) so remaining open to and accepting of the person suffering; and 5) Motivation to act or acting to alleviate suffering."
        "Your responses should balance emotional attunement and compassionate understanding with helpful, solution-oriented, and supportive guidance."
        "Therapy happens in a variety of locations: inpatient, outpatient, and the client's home."
        "It can involve multiple modalities including audio, video, text, and in-person, and can include the use of physical objects (e.g. to limit lethal means or for exposure)."
        "Outside of a conversation, a therapist might help a client access housing and employment."
        "They might prescribe  medication or assign homework. When necessary, a therapist may have to hospitalize a client."
        "Good therapy is client centered (e.g. involves shared decision making)."
        "Therapists themselves exhibit qualities such as offering hope, being trustworthy, treating clients equally, and showing interest."
        "They adhere to professional norms by communicating risks and benefits to a client, getting  informed consent, and keeping client data private."
        "Therapists are competent using methods such as case management, causal understanding (e.g. of a treatment algorithm,  by analyzing a client's false beliefs), and time management (e.g. pacing of a session)."
        "Therapeutic treatment is potentially harmful if applied wrong (e.g. with misdiagnosis, by colluding with delusions)."
        "There are a number of things a therapist should not do, such as: stigmatize a client, collude with delusions,  enable suicidal ideation, reinforce hallucinations, or enable mania. In many cases, a therapist should redirect a  client (e.g. appropriately challenge their thinking)."
    )

    try:
        if not client.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        # Recommended API for GPT-5.2
        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )

        reply = response.output_text or ""

    except Exception as e:
        reply = f"ERROR: {str(e)}"

    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
