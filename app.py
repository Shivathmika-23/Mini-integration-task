from flask import Flask, request, jsonify, render_template, send_file
from openai import OpenAI
import os
import json
import tempfile

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
You are a precise information extraction assistant.

Extract website details from the text below.

Text:
\"\"\"{text}\"\"\"

Rules:
- Use only what is stated or clearly implied
- Do NOT guess or hallucinate
- If a value is missing, return empty string or empty list
- Normalize values (e.g., heart care → Cardiology)
- Output ONLY valid JSON (no explanation, no markdown)

Output format:
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
    content = content[content.find("{"): content.rfind("}") + 1]

    return json.loads(content)


# -----------------------------
# Generate Website
# -----------------------------
@app.route("/generate", methods=["POST"])
def generate():
    data = request.get_json()
    text = data.get("text", "")

    extracted = extract_with_llm(text)

    business_name = extracted.get("name") or "My Business"
    website_type = extracted.get("type") or "Website"
    style = extracted.get("style") or "Basic"
    services = extracted.get("services") or ["General Services"]

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

    # -----------------------------
    # CREATE REAL HTML FILE
    # -----------------------------
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    with open(temp_file.name, "w", encoding="utf-8") as f:
        f.write(html)

    return jsonify({
        "business_name": business_name,
        "website_type": website_type,
        "style": style,
        "services": services,
        "html": html,
        "file_path": temp_file.name
    })


# -----------------------------
# Download HTML File
# -----------------------------
@app.route("/download-html", methods=["POST"])
def download_html():
    data = request.get_json()
    file_path = data.get("file_path")

    return send_file(
        file_path,
        as_attachment=True,
        download_name="generated_website.html"
    )


if __name__ == "__main__":
    app.run(debug=True)
