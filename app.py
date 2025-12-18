import os
import csv
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, render_template, request
from openai import OpenAI

from flask import send_file, abort

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)

CSV_FILE_PATH = "data/kakum_responses.csv"

# -------- PROMPTS -------- #

SYSTEM_PROMPT_SUMMARY = """
You are helping Melissa from the Light Your Mind Foundation understand a 10th grade student's mindset.

You will be given answers from a student about:
- who they are
- their studies
- their dreams for the future
- their support system
- their challenges
- their financial situation
- their access to technology
- how hopeful and confident they feel

Please create a clear, simple summary in English using these sections:

1. Student Snapshot (age, grade, school, location + 2–3 words that describe them)
2. Academic Profile (favorite subjects, difficult subjects, study pattern)
3. Future Dreams & Career Direction (what they want to become + how clear/realistic it seems)
4. Support System & Role Models (who supports them, who they talk to, any role models)
5. Challenges & Barriers (study-related, emotional, financial, or environmental)
6. Financial Context (how likely they are to need financial support for higher education)
7. Hopefulness & Confidence (how hopeful and confident they feel about their future)
8. Technology Access (phone/internet/app comfort level)
9. Suggestions for Kakum (2–3 ideas on what kind of guidance or features would help this student)

Rules:
- Do NOT invent facts. Only use what the student shared.
- If something is missing, say "Not clearly mentioned" instead of guessing.
- Keep the tone kind, respectful, and practical.
- This summary is for Melissa and the Kakum team (not for the student directly).
"""

SYSTEM_PROMPT_APPLICATION = """
You are helping a 10th grade student write an application letter to the Light Your Mind Foundation (LYM).

LYM Foundation has agreed to sponsor 2 months of coaching to help students prepare for their Board exams.
The student is applying for this scholarship/funding.

You will be given the student's answers about:
- their background and studies
- their future dreams
- their challenges
- their financial situation
- their support system
- their level of hope and confidence

Write a short, clear application letter in simple English, as if written by the student.

Structure:
- Start with a respectful greeting: "Dear Light Your Mind Foundation team,"
- Brief self-introduction (name, class, school, location)
- Mention their academic interests and goals
- Explain why Board exam coaching support is important for them
- Explain their financial situation in simple, honest words
- Mention any challenges they face (study environment, family responsibilities, etc.)
- Express how this 2-month coaching will help them in their exams and future
- End with gratitude and a polite closing

Tone:
- Respectful, sincere, and hopeful
- Simple English (10th grade level)
- No exaggeration and no invented facts

Rules:
- Do NOT add details that are not present in the student's answers.
- You may combine and rephrase their answers, but not fabricate new information.
- If some details are missing (for example, exact marks), you can keep it general.
- The letter should be 250–450 words.
"""
SYSTEM_PROMPT_ENCOURAGEMENT = """
You are writing a warm, positive, encouraging message to a 10th grade student from the LYM team.

Context:
- They completed a career/intake form.
- The message should be something they can revisit later as they grow and build their career.

Write a message in simple English (10th grade level), 180–300 words.
Tone: kind, hopeful, motivating, not preachy.
Rules:
- Use only information the student shared. Do not invent facts.
- If some details are missing, keep it general and encouraging.
- Do NOT mention scholarships, selection, rejection, or funding.

Structure:
- message should sound like its coming from the team and not individual person
- A warm opening
- 2–3 strengths you notice from their answers
- 2–3 practical next steps they can take in the next few months
- A positive closing

Formatting rules (important):
- Write 4 to 6 short paragraphs.
- Each paragraph must be 1–2 sentences max.
- Add a blank line between paragraphs.
"""


def build_user_prompt(form_data):
    """Turn form fields into a clean text prompt for the model."""
    lines = [
        "KAKUM 10TH GRADE INTAKE RESPONSES",
        "",
        f"Name: {form_data.get('name', '')}",
        f"Age: {form_data.get('age', '')}",
        f"Gender: {form_data.get('gender', '')}",
        f"School: {form_data.get('school', '')}",
        f"Location: {form_data.get('location', '')}",
        f"Email: {form_data.get('email', '')}",
        f"Scholarship 10k (Yes/No): {form_data.get('scholarship_10k', '')}",
        "",
        "ACADEMIC PROFILE",
        f"Favorite subjects: {form_data.get('favorite_subjects', '')}",
        f"Subjects they find difficult: {form_data.get('difficult_subjects', '')}",
        f"Daily study hours: {form_data.get('study_hours', '')}",
        f"Tuitions/coaching: {form_data.get('tuitions', '')}",
        "",
        "SELF-PERCEPTION",
        f"Describe yourself in 3 words: {form_data.get('self_words', '')}",
        f"Something they are proud of: {form_data.get('proud_of', '')}",
        f"Something they want to improve: {form_data.get('improve', '')}",
        "",
        "FUTURE DREAMS & PLANS",
        f"Career goal: {form_data.get('career_goal', '')}",
        f"Why this path: {form_data.get('career_reason', '')}",
        f"What they think they need to reach this goal: {form_data.get('needs_for_goal', '')}",
        f"Know anyone in that field: {form_data.get('knows_someone_in_field', '')}",
        "",
        "SUPPORT SYSTEM & EXPECTATIONS",
        f"Who they ask for advice: {form_data.get('advice_from', '')}",
        f"Expectations from teachers/school: {form_data.get('expect_school', '')}",
        f"Expectations from adults/the world: {form_data.get('expect_world', '')}",
        f"If they could ask for one kind of help: {form_data.get('one_help', '')}",
        "",
        "HOPE & CONFIDENCE",
        f"Hopefulness (1-10): {form_data.get('hope_score', '')}",
        f"Confidence (1-10): {form_data.get('confidence_score', '')}",
        f"Biggest worry about the future: {form_data.get('biggest_worry', '')}",
        "",
        "FINANCIAL SITUATION",
        f"Family can support higher education: {form_data.get('family_support_education', '')}",
        f"Need a scholarship: {form_data.get('need_scholarship', '')}",
        f"Biggest financial concerns: {form_data.get('financial_concerns', '')}",
        "",
        "CHALLENGES & BARRIERS",
        f"Problems that make it hard to study: {form_data.get('study_problems', '')}",
        f"What stops them from thinking big: {form_data.get('limits_big_thinking', '')}",
        f"Biggest challenge right now: {form_data.get('biggest_challenge', '')}",
        "",
        "TECHNOLOGY ACCESS",
        f"Has smartphone: {form_data.get('has_smartphone', '')}",
        f"Internet access: {form_data.get('internet_access', '')}",
        f"Used AI apps before: {form_data.get('ai_usage', '')}",
        f"Which AI app: {form_data.get('ai_which', '')}",

    ]

    return "\n".join(lines)


def save_to_csv(form_data, summary_text, application_text, encouragement_text, output_type, csv_path=CSV_FILE_PATH):

    """
    Save form responses + AI summary + scholarship application to a CSV file.
    Each submission becomes one row.
    """
    # Make sure the folder exists (data/)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    fieldnames = [
        "timestamp",
        "name",
        "age",
        "gender",
        "school",
        "location",
        "email",
        "favorite_subjects",
        "difficult_subjects",
        "study_hours",
        "tuitions",
        "self_words",
        "proud_of",
        "improve",
        "career_goal",
        "career_reason",
        "needs_for_goal",
        "knows_someone_in_field",
        "advice_from",
        "expect_school",
        "expect_world",
        "one_help",
        "hope_score",
        "confidence_score",
        "biggest_worry",
        "family_support_education",
        "need_scholarship",
        "financial_concerns",
        "study_problems",
        "limits_big_thinking",
        "biggest_challenge",
        "has_smartphone",
        "internet_access",
        "ai_usage",
        "ai_which",
        "ai_summary",
        "scholarship_application",
        "scholarship_10k",
        "output_type",
        "encouragement_message",
    ]

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "name": form_data.get("name", ""),
        "age": form_data.get("age", ""),
        "gender": form_data.get("gender", ""),
        "school": form_data.get("school", ""),
        "location": form_data.get("location", ""),
        "email": form_data.get("email", ""),
        "favorite_subjects": form_data.get("favorite_subjects", ""),
        "difficult_subjects": form_data.get("difficult_subjects", ""),
        "study_hours": form_data.get("study_hours", ""),
        "tuitions": form_data.get("tuitions", ""),
        "self_words": form_data.get("self_words", ""),
        "proud_of": form_data.get("proud_of", ""),
        "improve": form_data.get("improve", ""),
        "career_goal": form_data.get("career_goal", ""),
        "career_reason": form_data.get("career_reason", ""),
        "needs_for_goal": form_data.get("needs_for_goal", ""),
        "knows_someone_in_field": form_data.get("knows_someone_in_field", ""),
        "advice_from": form_data.get("advice_from", ""),
        "expect_school": form_data.get("expect_school", ""),
        "expect_world": form_data.get("expect_world", ""),
        "one_help": form_data.get("one_help", ""),
        "hope_score": form_data.get("hope_score", ""),
        "confidence_score": form_data.get("confidence_score", ""),
        "biggest_worry": form_data.get("biggest_worry", ""),
        "family_support_education": form_data.get("family_support_education", ""),
        "need_scholarship": form_data.get("need_scholarship", ""),
        "financial_concerns": form_data.get("financial_concerns", ""),
        "study_problems": form_data.get("study_problems", ""),
        "limits_big_thinking": form_data.get("limits_big_thinking", ""),
        "biggest_challenge": form_data.get("biggest_challenge", ""),
        "has_smartphone": form_data.get("has_smartphone", ""),
        "internet_access": form_data.get("internet_access", ""),
        "ai_usage": form_data.get("ai_usage", ""),
        "ai_which": form_data.get("ai_which", ""),
        "ai_summary": summary_text,
        "scholarship_application": application_text,
        "scholarship_10k": form_data.get("scholarship_10k", ""),
        "output_type": output_type,
        "encouragement_message": encouragement_text,
    }

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


@app.route("/", methods=["GET"])
def form():
    """Show the Kakum 10th grade intake form."""
    return render_template("form.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate either scholarship application OR encouragement message based on scholarship selection."""
    user_prompt = build_user_prompt(request.form)

    scholarship_choice = (request.form.get("scholarship_10k", "") or "").strip().lower()

    # 1) Internal profile summary (keep this for CSV analysis; not shown on results page)
    summary_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_SUMMARY},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
        max_tokens=1200,
    )
    summary_text = summary_response.choices[0].message.content

    # 2) Conditional output
    application_text = ""
    encouragement_text = ""
    encouragement_response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_ENCOURAGEMENT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        max_tokens=500,
        )
    encouragement_text = encouragement_response.choices[0].message.content
    encouragement_text = encouragement_text.replace("\n", "\n\n")
    output_type = "encouragement"

    # Save to CSV (update function signature in next step)
    try:
        save_to_csv(request.form, summary_text, application_text, encouragement_text, output_type)
    except Exception as e:
        print(f"Error saving to CSV: {e}")

    return render_template(
        "result.html",
        student_name=request.form.get("name", ""),
        output_type=output_type,
        application_text=application_text,
        encouragement_text=encouragement_text,
    )

@app.route("/admin/download-csv")
def download_csv():
    secret = request.args.get("key")

    # Check authorization key
    if secret != os.getenv("ADMIN_KEY"):
        return abort(403)

    csv_path = CSV_FILE_PATH

    if not os.path.exists(csv_path):
        return "CSV file not found", 404

    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="kakum_responses.csv"
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
