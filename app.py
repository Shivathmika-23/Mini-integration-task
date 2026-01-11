from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
import json

app = Flask(__name__)

# -----------------------------
# Hugging Face Router LLM Setup
# -----------------------------
HF_TOKEN =os.getenv("HF_TOKEN")
llm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# LLM Extraction Function
# -----------------------------
def extract_with_llm(text):
    prompt = f"""
You are a precise information extraction system.

Your task is to extract structured website requirements from the given text.

Text:
\"\"\"{text}\"\"\"

Extraction Rules:
- Extract ONLY what is explicitly stated or very clearly implied.
- Do NOT guess names, services, or styles.
- Do NOT infer information that is not mentioned.
- If a field is missing, return an empty string or empty list.
- Normalize values (e.g., "heart care" → "Cardiology").
- Return ONLY valid JSON.
- Do NOT include explanations, markdown, or extra text.

Fields to extract:
- name: Business or website name
- type: Website type (Hospital, School, Restaurant, Company, etc.)
- style: Design style (Modern, Minimal, Professional, etc.)
- services: List of services offered

Output format (strict):
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

    content = completion.choices[0].message.content.strip()

    # Safety: extract JSON only
    content = content[content.find("{"): content.rfind("}") + 1]
    return json.loads(content)


# -----------------------------
# Generate Website API
# -----------------------------
@app.route("/generate", methods=["POST"])
def generate():
    try:
        data = request.get_json()
        text = data.get("text", "")
        
        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Call LLM extractor
        extracted = extract_with_llm(text)

        # Fallbacks (never break UI)
        business_name = extracted.get("name", "My Business") 
        website_type = extracted.get("type", "Business") 
        style = extracted.get("style", "Modern") 
        services = extracted.get("services", [])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # -----------------------------
    # HTML Generation
    # -----------------------------
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{business_name}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background: rgba(255,255,255,0.95);
                padding: 40px;
                border-radius: 15px;
                text-align: center;
                margin-bottom: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .header h1 {{
                color: #2c3e50;
                font-size: 3em;
                margin-bottom: 10px;
            }}
            .header .type {{
                color: #7f8c8d;
                font-size: 1.2em;
                margin-bottom: 20px;
            }}
            .content {{
                background: rgba(255,255,255,0.95);
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            }}
            .services {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .service-card {{
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                border-left: 4px solid #3498db;
                transition: transform 0.3s ease;
            }}
            .service-card:hover {{
                transform: translateY(-5px);
            }}
            .style-badge {{
                display: inline-block;
                background: #3498db;
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 0.9em;
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{business_name}</h1>
                <div class="type">{website_type}</div>
                <div class="style-badge">{style} Design</div>
            </div>
            
            <div class="content">
                <h2>Our Services</h2>
                <div class="services">
                    {''.join(f'<div class="service-card"><h3>{s}</h3><p>Professional {s.lower()} services tailored to your needs.</p></div>' for s in services)}
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    return jsonify({
        "business_name": business_name,
        "website_type": website_type,
        "style": style,
        "services": services,
        "html": html
    })


if __name__ == "__main__":
    app.run(debug=True)
