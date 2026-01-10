from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import OpenAI
import speech_recognition as sr
import tempfile
import os
import json

app = FastAPI()

# =========================
# ENV
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not set")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# =========================
# LLM
# =========================
def extract_with_llm(text: str):
    prompt = f"""
Extract website details and return ONLY JSON.

Text:
"{text}"

JSON format:
{{
  "name": "business name",
  "type": "Hospital | School | Restaurant | Company",
  "style": "Modern | Minimal | Professional",
  "services": ["service1", "service2"]
}}
"""

    try:
        res = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        msg = res.choices[0].message.content
        return json.loads(msg[msg.find("{"): msg.rfind("}") + 1])

    except Exception:
        return {
            "name": "My Business",
            "type": "Business",
            "style": "Modern",
            "services": []
        }

# =========================
# HTML
# =========================
def generate_html(data):
    services = "".join(
        f"<div class='service'>{s}</div>" for s in data["services"]
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<style>
body {{ font-family: Arial; background:#f5f5f5; padding:30px }}
.container {{ max-width:800px; background:white; padding:40px; margin:auto }}
.service {{ background:#eee; padding:10px; margin:10px 0 }}
</style>
</head>
<body>
<div class="container">
<h1>{data['name']}</h1>
<p>{data['type']} • {data['style']}</p>
{services}
</div>
</body>
</html>"""

# =========================
# API
# =========================
@app.post("/generate")
async def generate(audio: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(await audio.read())
            path = f.name

        r = sr.Recognizer()
        with sr.AudioFile(path) as src:
            audio_data = r.record(src)
            text = r.recognize_google(audio_data)

        os.remove(path)

        data = extract_with_llm(text)
        return {"html": generate_html(data)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "API is running",
        "endpoints": ["/generate", "/docs"]
    }


