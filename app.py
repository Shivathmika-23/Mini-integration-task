import json
import os
from flask import Flask, request, Response, render_template
from openai import OpenAI
HF_TOKEN = os.getenv("HF_TOKEN")


# -----------------------------
# LLM Client (Hugging Face Router)
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
# UI
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------
# LLM: TEXT → JSON
# -----------------------------
def extract_with_llm(text):
    prompt = f"""
You are an information extraction assistant.

Read the text below and extract website details.

Text:
\"\"\"{text}\"\"\"

Instructions:
- Extract the website name, website type, design style, and services.
- Use ONLY the information explicitly stated or clearly implied in the text.
- Do NOT guess or add new information.
- If a value is missing, return an empty string or empty array.
- Return ONLY valid JSON.
- Do NOT include explanations, comments, or markdown.

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

    content = completion.choices[0].message.content
    content = content[content.find("{"):content.rfind("}") + 1]
    return json.loads(content)

# -----------------------------
# JSON → HTML
# -----------------------------
def generate_html(data):
    services_html = "".join(f"<li>{s}</li>" for s in data.get("services", []))

    return f"""
<!DOCTYPE html>
<html>
<head>
  <title>{data.get("name", "Website")}</title>
  <style>
    body {{
      font-family: Arial;
      padding: 40px;
      background: #f4f6f8;
    }}
    h1 {{ color: #2c3e50; }}
    ul {{ margin-top: 10px; }}
  </style>
</head>
<body>
  <h1>{data.get("name")} - {data.get("type")}</h1>
  <p><b>Style:</b> {data.get("style")}</p>
  <h3>Services</h3>
  <ul>{services_html}</ul>
</body>
</html>
"""

# -----------------------------
# API: TEXT → WEBSITE
# -----------------------------
@app.route("/generate-website-text", methods=["POST"])
def generate_website_text():
    data = request.json
    text = data.get("text")

    if not text:
        return {"error": "No text provided"}, 400

    details = extract_with_llm(text)
    html = generate_html(details)

    return Response(html, mimetype="text/html")

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
