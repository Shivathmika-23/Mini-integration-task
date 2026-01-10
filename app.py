from fastapi import FastAPI, UploadFile, File
import speech_recognition as sr
import tempfile
import os
import json
from openai import OpenAI

app = FastAPI()

# --- LLM Setup ---
HF_TOKEN = os.getenv("HF_TOKEN")
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

@app.get("/")
def root():
    return {"status": "API running"}

# --- LLM extraction (safe) ---
def extract_with_llm(text):
    try:
        prompt = f"""Extract website details from this text and return only JSON:

Text: "{text}"

Return format:
{{
  "name": "business name",
  "type": "website type (Hospital/School/Restaurant/Company)",
  "style": "design style (Modern/Minimal/Professional)",
  "services": ["service1", "service2"]
}}"""
        completion = llm_client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        content = completion.choices[0].message.content.strip()
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > 0:
            return json.loads(content[start:end])
    except Exception as e:
        return {"error": f"LLM extraction failed: {str(e)}"}
    return {"name": "Default Business", "type": "Business", "style": "Modern", "services": []}

# --- HTML generator ---
def generate_html(data):
    try:
        name = data.get('name', 'My Business')
        website_type = data.get('type', 'Business')
        style = data.get('style', 'Modern')
        services = data.get('services', [])
        services_html = ""
        if services:
            services_html = '<h2>Our Services</h2><div class="services">'
            for service in services:
                services_html += f'<div class="service"><h3>{service}</h3></div>'
            services_html += '</div>'
        return f"""<html><head><title>{name}</title></head>
        <body>
        <h1>{name}</h1>
        <p>{website_type} - {style} Style</p>
        {services_html}
        </body></html>"""
    except Exception as e:
        return f"<p>HTML generation error: {str(e)}</p>"

# --- /generate endpoint ---
@app.post("/generate")
async def generate_website(audio: UploadFile = File(...)):
    response = {"debug": []}
    try:
        # Save audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await audio.read()
            tmp.write(content)
            path = tmp.name
        response["debug"].append(f"Saved file: {path}")
    except Exception as e:
        response["error"] = f"Failed to save audio: {str(e)}"
        return response

    try:
        # Convert to text
        r = sr.Recognizer()
        with sr.AudioFile(path) as source:
            audio_data = r.record(source)
            try:
                text = r.recognize_google(audio_data)
            except Exception as e:
                text = f"[Recognition failed: {str(e)}]"
        response["debug"].append(f"Recognized text: {text}")
    except Exception as e:
        response["error"] = f"SpeechRecognition error: {str(e)}"
        return response
    finally:
        try:
            os.unlink(path)
        except:
            pass

    try:
        # Extract website data
        website_data = extract_with_llm(text)
        response["debug"].append(f"Website data: {website_data}")
    except Exception as e:
        response["error"] = f"LLM extraction error: {str(e)}"
        return response

    try:
        html = generate_html(website_data)
        response["html"] = html
    except Exception as e:
        response["error"] = f"HTML generation error: {str(e)}"
        return response

    return response
