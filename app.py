from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
import speech_recognition as sr
import tempfile
import os
import json

app = FastAPI()

# =========================
# LLM SETUP (HuggingFace)
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set")

llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# =========================
# LLM EXTRACTION
# =========================
def extract_with_llm(text: str):
    prompt = f"""
Extract website details from the following text.
Return ONLY valid JSON.

Text:
"{text}"

Format:
{{
  "name": "business name",
  "type": "Hospital | School | Restaurant | Company",
  "style": "Modern | Minimal | Professional",
  "services": ["service1", "service2"]
}}
"""

    try:
        response = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1

        return json.loads(content[start:end])

    except Exception:
        return {
            "name": "My Business",
            "type": "Business",
            "style": "Modern",
            "services": []
        }

# =========================
# HTML GENERATOR
# =========================
def generate_html(data):
    services_html = ""
    if data["services"]:
        services_html += "<h2>Our Services</h2><div class='services'>"
        for s in data["services"]:
            services_html += f"<div class='service'>{s}</div>"
        services_html += "</div>"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{data['name']}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f5f5f5;
            padding: 30px;
        }}
        .container {{
            max-width: 800px;
            margin: auto;
            background: white;
            padding: 40px;
            border-radius: 10px;
        }}
        h1 {{
            text-align: center;
        }}
        .services {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .service {{
            background: #f0f0f0;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{data['name']}</h1>
        <p style="text-align:center;">
            {data['type']} • {data['style']} Design
        </p>
        {services_html}
    </div>
</body>
</html>"""

# =========================
# API ENDPOINT
# =========================
@app.post("/generate")
async def generate_website(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file uploaded")

    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await audio.read())
            temp_path = tmp.name

        # Speech to Text (NO PyAudio needed)
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        os.remove(temp_path)

        # LLM extraction
        website_data = extract_with_llm(text)

        # Generate HTML
        html = generate_html(website_data)

        return {
            "transcription": text,
            "html": html
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
