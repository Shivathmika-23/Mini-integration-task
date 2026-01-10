from fastapi import FastAPI, UploadFile, File, HTTPException
import speech_recognition as sr
import tempfile
import os
import json
import os
from openai import OpenAI

app = FastAPI()

# --- LLM Setup ---
HF_TOKEN = os.getenv("HF_TOKEN")
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

# --- Root / health check ---
@app.get("/")
def root():
    return {"status": "API is running", "endpoints": ["/generate", "/docs"]}

# --- LLM extraction function ---
def extract_with_llm(text):
    prompt = f"""Extract website details from this text and return only JSON:

Text: "{text}"

Return format:
{{
  "name": "business name",
  "type": "website type (Hospital/School/Restaurant/Company)",
  "style": "design style (Modern/Minimal/Professional)",
  "services": ["service1", "service2"]
}}"""

    try:
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
    except:
        pass
    
    return {"name": "My Business", "type": "Business", "style": "Modern", "services": []}

# --- HTML generation ---
def generate_html(data):
    name = data['name']
    website_type = data['type']
    style = data['style']
    services = data['services']
    
    services_html = ""
    if services:
        services_html = '<h2>Our Services</h2><div class="services">'
        for service in services:
            services_html += f'<div class="service"><h3>{service}</h3></div>'
        services_html += '</div>'
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>{name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; }}
        h1 {{ color: #333; text-align: center; }}
        .type {{ text-align: center; color: #666; margin: 20px 0; }}
        .services {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }}
        .service {{ background: #f8f9fa; padding: 15px; border-radius: 5px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{name}</h1>
        <div class="type">{website_type} - {style} Style</div>
        {services_html}
    </div>
</body>
</html>"""

# --- Main generate endpoint ---
@app.post("/generate")
async def generate_website(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    try:
        # Save uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
            
        # Convert audio to text
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        
        os.unlink(temp_file_path)
        
        # Extract website data
        website_data = extract_with_llm(text)
        
        # Generate HTML
        html = generate_html(website_data)
        
        return {"html": html}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
