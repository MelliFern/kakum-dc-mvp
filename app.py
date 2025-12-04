import os
from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

# System prompt for Dream Chaser application helper
SYSTEM_PROMPT = """
You are helping a student draft a scholarship application for the 
"Dream Chaser" program by the Light Your Mind Foundation (LYM).

Using the answers provided, write a clear, honest, and encouraging 
application in the student's voice (simple English, not too fancy).

Structure the application using these sections:

1. Personal Background
2. Academic Journey
3. Financial Situation
4. Support System & Challenges
5. Future Dreams & Career Plan
6. How the Dream Chaser Scholarship Will Help

While drafting, keep in mind these evaluation criteria:
- Feasibility of Career Choice
- Initiative & Self-Drive
- Mentorship & Role Models
- Resilience & Problem-Solving
- Financial Need & Support System

Do NOT invent facts. Use only what the student has shared. 
If some information is missing, write gently around it without making things up.
Keep the tone warm, respectful, and authentic.
"""

def build_user_prompt(form_data):
    """Turn form fields into a clean text prompt for the model."""
    lines = [
        f"Name: {form_data.get('name', '')}",
        f"Age: {form_data.get('age', '')}",
        f"City/Country: {form_data.get('location', '')}",
        "",
        "Academic Background:",
        form_data.get("academic_background", ""),
        "",
        "Financial Situation:",
        form_data.get("financial_situation", ""),
        "",
        "Support System / Family Structure:",
        form_data.get("support_system", ""),
        "",
        "Future Dreams & Plans:",
        form_data.get("future_plans", ""),
        "",
        "Anything else the student wants to share:",
        form_data.get("extra_info", "")
    ]
    return "\n".join(lines)

@app.route("/", methods=["GET"])
def form():
    """Show the Dream Chaser form."""
    return render_template("form.html")

@app.route("/generate", methods=["POST"])
def generate():
    """Generate the scholarship application draft using OpenAI."""
    user_prompt = build_user_prompt(request.form)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # you can change model later if needed
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.7,
        max_tokens=1200,
    )

    application_text = response.choices[0].message.content
    return render_template("result.html", application_text=application_text)

if __name__ == "__main__":
    # For local + deployment: host/port are configurable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
