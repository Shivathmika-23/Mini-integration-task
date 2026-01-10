from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import speech_recognition as sr
import tempfile
import os
import json

app = Flask(__name__)

# LLM Setup
HF_TOKEN = os.getenv("HF_TOKEN")
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)

def extract_with_llm(text):
    """Extract website details using LLM"""
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

def generate_html(data):
    """Generate HTML website"""
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_website():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Convert audio to text
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            audio_file.save(temp_file.name)
            temp_file_path = temp_file.name
            
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_file_path) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
        
        os.unlink(temp_file_path)
        
        # Extract details using LLM
        website_data = extract_with_llm(text)
        
        # Generate HTML
        html = generate_html(website_data)
        
        return jsonify({
            "business_name": website_data["name"],
            "website_type": website_data["type"],
            "style": website_data["style"],
            "services": website_data["services"],
            "html": html
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
