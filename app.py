from flask import Flask, request, jsonify, render_template
from openai import OpenAI
import os
import json

app = Flask(__name__)

# -----------------------------
# Hugging Face Router LLM Setup
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")

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
    data = request.get_json()
    text = data.get("text", "")

    # Call LLM extractor
    extracted = extract_with_llm(text)

    # Fallbacks (never break UI)
    business_name = extracted.get("name") 
    website_type = extracted.get("type") 
    style = extracted.get("style") 
    services = extracted.get("services")

    # -----------------------------
    # HTML Generation
    # -----------------------------
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{business_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 40px;
                background-color: #f4f4f4;
            }}
            .container {{
                background: white;
                padding: 25px;
                border-radius: 10px;
            }}
            h1 {{
                color: #2c3e50;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>{business_name}</h1>
            <h3>Type: {website_type}</h3>
            <p><strong>Style:</strong> {style}</p>

            <h3>Services</h3>
            <ul>
                {''.join(f"<li>{s}</li>" for s in services)}
            </ul>
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
