

import os
import json
import speech_recognition as sr
from flask import Flask, request, Response, render_template
from werkzeug.utils import secure_filename
from openai import OpenAI
HF_TOKEN = os.getenv("HF_TOKEN")

# -----------------------------
# Config
# -----------------------------
UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# LLM Client (HF Router)
# -----------------------------
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# UI Route
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# Voice → Text (Google Speech)
# -----------------------------
def voice_to_text(wav_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_path) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)

# -----------------------------
# Extract Website Details
# -----------------------------
def extract_with_llm(text):
    prompt = f"""
Extract website details from the text below.

Text: "{text}"

Return ONLY valid JSON:
{{
  "name": "",
  "type": "",
  "style": "",
  "services": []
}}
"""
    completion = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:novita",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return json.loads(completion.choices[0].message.content.strip())

# -----------------------------
# Generate HTML
# -----------------------------
def generate_html(data):
    services_html = "".join(
        f"<li>{s}</li>" for s in data.get("services", [])
    )

    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>{data.get("name", "Website")}</title>
</head>
<body>
  <h1>{data.get("name", "")} - {data.get("type", "")}</h1>
  <p>Style: {data.get("style", "")}</p>

  <h3>Services</h3>
  <ul>
    {services_html}
  </ul>
</body>
</html>
"""

# -----------------------------
# API Endpoint (WAV only)
# -----------------------------
@app.route("/generate-website", methods=["POST"])
def generate_website():
    if "audio" not in request.files:
        return {"error": "No file uploaded"}, 400

    uploaded_file = request.files["audio"]
    filename = secure_filename(uploaded_file.filename)

    if not filename.lower().endswith(".wav"):
        return {"error": "Only WAV files supported"}, 400

    input_path = os.path.join(UPLOAD_FOLDER, filename)
    uploaded_file.save(input_path)

    text = voice_to_text(input_path)
    details = extract_with_llm(text)
    html = generate_html(details)

    return Response(html, mimetype="text/html")

# -----------------------------
# Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
