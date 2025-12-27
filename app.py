from flask import Flask, request, jsonify
from openai import OpenAI
import os
import time
import uuid
import logging

app = Flask(__name__)

# ---- Logging setup (Render shows these in the Logs tab) ----
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger("qualtrics-gpt-backend")

# Create OpenAI client (expects OPENAI_API_KEY in Render env vars)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.get("/health")
def health():
    return "ok", 200

@app.route("/qualtrics-response", methods=["POST"])
def qualtrics_response():
    req_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    def elapsed_ms():
        return int((time.perf_counter() - t0) * 1000)

    # 1) Log: request received
    logger.info(f"[{req_id}] REQUEST received t+{elapsed_ms()}ms content_type={request.content_type}")

    # Safely extract prompt from form or JSON
    try:
        user_text = (request.form.get("prompt") if request.form else None) or \
                    (request.json.get("prompt") if request.is_json and request.json else "") or ""
        user_text = user_text.strip()
        logger.info(f"[{req_id}] REQUEST parsed len={len(user_text)} t+{elapsed_ms()}ms")
    except Exception as e:
        logger.exception(f"[{req_id}] REQUEST parse_error t+{elapsed_ms()}ms: {e}")
        return jsonify({"reply": "Thank you for your response. We were unable to process it at this time."}), 200

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
        "This is a one-time, single-message interaction. Provide your best complete response in one turn. Do NOT ask any follow-up questions (including “can you tell me more…?”). Do not invite further conversation or request additional details."
    )

    reply = ""
    try:
        if not client.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")

        # 2) Log: OpenAI call start
        logger.info(f"[{req_id}] OPENAI start t+{elapsed_ms()}ms")

        t_api = time.perf_counter()

        response = client.responses.create(
            model="gpt-5.2",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
        )

        api_ms = int((time.perf_counter() - t_api) * 1000)

        reply = (response.output_text or "").strip()

        # 3) Log: OpenAI call end
        logger.info(f"[{req_id}] OPENAI end api_ms={api_ms} reply_chars={len(reply)} t+{elapsed_ms()}ms")

    except Exception as e:
        # Log full error details for you (not shown to participants)
        logger.exception(f"[{req_id}] OPENAI error t+{elapsed_ms()}ms: {e}")

        # Participant-safe fallback message
        reply = (
            "Thank you for sharing. The chatbot was unable to generate a response at this time. "
            "Please proceed to the next page."
        )

    # 4) Log: response returned + total time
    logger.info(f"[{req_id}] RESPONSE returned chars={len(reply)} total_ms={elapsed_ms()}")

    return jsonify({"reply": reply}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)
