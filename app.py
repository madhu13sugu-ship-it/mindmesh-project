from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types
from werkzeug.security import generate_password_hash, check_password_hash

import os
import base64
import random
import time
import requests
import json
import sqlite3
import secrets

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)


# ==========================================
# API KEYS
# ==========================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
BREVO_API_KEY = os.environ.get("BREVO_API_KEY")
BREVO_SENDER_EMAIL = os.environ.get("BREVO_SENDER_EMAIL")
JOOBLE_API_KEY = os.environ.get("JOOBLE_API_KEY")

JOOBLE_API_BASE_URL = os.environ.get(
    "JOOBLE_API_BASE_URL",
    "https://jooble.org/api"
).rstrip("/")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY is not set")

if not BREVO_SENDER_EMAIL:
    raise ValueError("BREVO_SENDER_EMAIL is not set")


# ==========================================
# GEMINI CONFIGURATION
# ==========================================

client = genai.Client(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-3.5-flash-lite"


def generate_with_retry(contents):

    delays = [2, 4, 8]

    for attempt in range(4):

        try:
            print("Gemini attempt:", attempt + 1)

            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents
            )

        except Exception as e:

            error_text = str(e)

            print("Gemini attempt failed:", repr(e))

            temporary_error = any(
                term in error_text.lower()
                for term in (
                    "503",
                    "unavailable",
                    "high demand",
                    "temporarily",
                    "429",
                    "rate limit"
                )
            )

            if temporary_error and attempt < 3:

                wait_time = delays[attempt]

                print("Retrying in", wait_time, "seconds...")

                time.sleep(wait_time)

            else:
                raise


def remove_json_fences(text):

    text = (text or "").strip()

    if text.startswith("```"):

        lines = text.splitlines()

        if lines:
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    return text


def get_json_body():

    return request.get_json(silent=True) or {}


# ==========================================
# SQLITE DATABASE
# ==========================================

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mindmesh_users.db"
)


def get_db_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    return connection


def init_db():

    with get_db_connection() as db:

        db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                token TEXT PRIMARY KEY,
                email TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)


def create_session(email):

    token = secrets.token_urlsafe(32)

    with get_db_connection() as db:

        db.execute(
            """
            INSERT INTO user_sessions (token, email)
            VALUES (?, ?)
            """,
            (token, email)
        )

    return token


init_db()


# ==========================================
# OTP STORAGE
# ==========================================

# OTPs are stored temporarily in memory.
# They are cleared when the server restarts.

otp_storage = {}


# ==========================================
# AI TUTOR - TEXT
# ==========================================

@app.route("/ask", methods=["POST"])
def ask_ai():

    try:

        data = get_json_body()

        question = str(
            data.get("question", "")
        ).strip()

        if not question:

            return jsonify({
                "answer": "Please enter a question."
            }), 400

        response = generate_with_retry(question)

        answer = (
            getattr(response, "text", None) or ""
        ).strip()

        if not answer:

            return jsonify({
                "answer": "I could not generate an answer."
            }), 502

        return jsonify({
            "answer": answer
        }), 200

    except Exception as e:

        print("GEMINI ERROR:", repr(e))

        return jsonify({
            "answer": "The AI service is temporarily busy. Please try again."
        }), 500


# ==========================================
# AI TUTOR - IMAGE
# ==========================================

@app.route("/analyze-image", methods=["POST"])
def analyze_image():

    try:

        data = get_json_body()

        image_base64 = str(
            data.get("image", "")
        ).strip()

        question_prompt = str(
            data.get(
                "question",
                "Read the question in this image carefully and solve it."
            )
        ).strip()

        if not image_base64:

            return jsonify({
                "answer": "No image was received."
            }), 400

        if (
            "," in image_base64
            and image_base64.lower().startswith("data:image")
        ):

            image_base64 = image_base64.split(",", 1)[1]

        image_bytes = base64.b64decode(
            image_base64,
            validate=True
        )

        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )

        response = generate_with_retry([
            image_part,
            question_prompt
        ])

        answer = (
            getattr(response, "text", None) or ""
        ).strip()

        if not answer:

            return jsonify({
                "answer": "I could not understand the image."
            }), 502

        return jsonify({
            "answer": answer
        }), 200

    except Exception as e:

        print("IMAGE GEMINI ERROR:", repr(e))

        return jsonify({
            "answer": "The AI service could not analyze the image. Please try again."
        }), 500


# ==========================================
# MOCK INTERVIEW - AI ANSWER EVALUATION
# ==========================================

@app.route("/mock-interview/evaluate", methods=["POST"])
def evaluate_mock_interview():

    try:

        data = get_json_body()

        question = str(
            data.get("question", "")
        ).strip()

        answer = str(
            data.get("answer", "")
        ).strip()

        if not question or not answer:

            return jsonify({
                "success": False,
                "message": "Please provide both the question and your answer."
            }), 400

        prompt = f"""
You are an experienced interview coach for college students and freshers.
Evaluate the student's answer fairly for correctness, relevance,
completeness, and clarity. Give partial credit where appropriate.
For personal or HR questions, evaluate relevance and clarity without
assuming there is one correct personal answer. Do not invent achievements.

QUESTION:
{question}

STUDENT ANSWER:
{answer}

Return ONLY one valid JSON object, with no Markdown fences and these keys:
{{
  "score": 0,
  "evaluation": "Brief evaluation",
  "strengths": ["Strength 1"],
  "missing_points": ["Missing point 1"],
  "improved_answer": "A stronger sample answer",
  "suggestion": "One practical interview tip"
}}

The score must be a number from 0 to 10.
Use empty arrays if there are no relevant items.
"""

        print("MOCK INTERVIEW: Sending answer to Gemini")

        response = generate_with_retry(prompt)

        result_text = remove_json_fences(
            getattr(response, "text", None)
        )

        if not result_text:

            return jsonify({
                "success": False,
                "message": "The AI returned an empty response. Please try again."
            }), 502

        start = result_text.find("{")
        end = result_text.rfind("}")

        if start >= 0 and end > start:

            result_text = result_text[start:end + 1]

        feedback = json.loads(result_text)

        if not isinstance(feedback, dict):

            raise ValueError(
                "AI feedback is not a JSON object"
            )

        try:

            score = float(
                feedback.get("score", 0)
            )

        except (TypeError, ValueError):

            score = 0.0

        feedback["score"] = max(
            0,
            min(10, score)
        )

        feedback["evaluation"] = str(
            feedback.get(
                "evaluation",
                "No evaluation available."
            )
        )

        feedback["improved_answer"] = str(
            feedback.get(
                "improved_answer",
                "No sample answer available."
            )
        )

        feedback["suggestion"] = str(
            feedback.get(
                "suggestion",
                "Keep practicing."
            )
        )

        for key in ("strengths", "missing_points"):

            value = feedback.get(key, [])

            if value is None:
                value = []

            if not isinstance(value, list):
                value = [str(value)]

            feedback[key] = [
                str(item) for item in value
            ]

        print(
            "MOCK INTERVIEW: Feedback prepared successfully"
        )

        return jsonify({
            "success": True,
            "feedback": feedback
        }), 200

    except json.JSONDecodeError as e:

        print("MOCK INTERVIEW JSON ERROR:", repr(e))

        return jsonify({
            "success": False,
            "message": "The AI returned unreadable feedback. Please submit again."
        }), 502

    except Exception as e:

        print("MOCK INTERVIEW ERROR:", repr(e))

        return jsonify({
            "success": False,
            "message": "AI evaluation failed. Check the Flask terminal for the error."
        }), 500


# ==========================================
# INTERNSHIP SEARCH - JOOBLE
# ==========================================

@app.route("/internships", methods=["GET", "POST"])
def find_internships():

    if not JOOBLE_API_KEY:

        return jsonify({
            "success": False,
            "message": "Jooble API key is not configured.",
            "jobs": []
        }), 503

    data = (
        get_json_body()
        if request.method == "POST"
        else request.args
    )

    skills = str(
        data.get("skills", "")
    ).strip()

    location = str(
        data.get(
            "location",
            "Bengaluru, India"
        )
    ).strip() or "India"

    remote_value = str(
        data.get("remote", "false")
    ).strip().lower()

    remote = remote_value in (
        "true",
        "1",
        "yes",
        "on"
    )

    if not skills:

        return jsonify({
            "success": False,
            "message": "Please enter your skills.",
            "jobs": []
        }), 400

    keywords = (
        f"internship {skills}"
        + (" remote" if remote else "")
    )

    payload = {
        "keywords": keywords,
        "location": location,
        "page": "1",
        "ResultOnPage": 10,
        "companysearch": False
    }

    api_url = (
        f"{JOOBLE_API_BASE_URL}/{JOOBLE_API_KEY}"
    )

    try:

        response = requests.post(
            api_url,
            json=payload,
            headers={
                "Content-Type": "application/json"
            },
            timeout=20
        )

        print(
            "JOOBLE STATUS:",
            response.status_code
        )

        if response.status_code == 403:

            return jsonify({
                "success": False,
                "message": "Jooble rejected the API key.",
                "jobs": []
            }), 502

        if response.status_code != 200:

            print(
                "JOOBLE ERROR:",
                response.text[:500]
            )

            return jsonify({
                "success": False,
                "message": f"Jooble returned HTTP {response.status_code}.",
                "jobs": []
            }), 502

        result = response.json()

        jobs = []

        for item in result.get("jobs", []):

            title = str(
                item.get("title") or ""
            ).strip()

            link = str(
                item.get("link") or ""
            ).strip()

            if not title or not link.startswith(
                ("https://", "http://")
            ):
                continue

            jobs.append({

                "id": str(
                    item.get("id") or ""
                ),

                "title": title,

                "company": str(
                    item.get("company")
                    or "Company not specified"
                ),

                "location": str(
                    item.get("location")
                    or "Location not specified"
                ),

                "snippet": str(
                    item.get("snippet") or ""
                ),

                "salary": str(
                    item.get("salary")
                    or "Not specified"
                ),

                "type": str(
                    item.get("type")
                    or "Not specified"
                ),

                "source": str(
                    item.get("source")
                    or "Jooble"
                ),

                "link": link,

                "updated": str(
                    item.get("updated") or ""
                ),

                "deadline": None
            })

        return jsonify({

            "success": True,

            "query": keywords,

            "location": location,

            "totalCount": result.get(
                "totalCount",
                len(jobs)
            ),

            "jobs": jobs,

            "message": (
                "Open each original listing to confirm that applications are still open."
                if jobs
                else "No matching listings found. Try different skills."
            )

        }), 200

    except requests.Timeout:

        return jsonify({
            "success": False,
            "message": "Jooble timed out. Please try again.",
            "jobs": []
        }), 504

    except requests.RequestException as e:

        print(
            "JOOBLE REQUEST ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Could not connect to Jooble.",
            "jobs": []
        }), 502

    except ValueError as e:

        print(
            "JOOBLE RESPONSE ERROR:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Jooble returned an unreadable response.",
            "jobs": []
        }), 502


# ==========================================
# SEND OTP USING BREVO
# ==========================================

@app.route("/send-otp", methods=["POST"])
def send_otp():

    try:

        data = get_json_body()

        email = str(
            data.get("email", "")
        ).strip().lower()

        if not email:

            return jsonify({
                "success": False,
                "message": "Email is required"
            }), 400

        # Check whether account already exists.
        with get_db_connection() as db:

            existing = db.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if existing:

            return jsonify({
                "success": False,
                "message": "An account already exists. Please log in."
            }), 409

        otp = str(
            random.randint(100000, 999999)
        )

        url = "https://api.brevo.com/v3/smtp/email"

        headers = {
            "accept": "application/json",
            "api-key": BREVO_API_KEY,
            "content-type": "application/json"
        }

        payload = {

            "sender": {
                "name": "MindMesh",
                "email": BREVO_SENDER_EMAIL
            },

            "to": [
                {"email": email}
            ],

            "subject": "MindMesh Email Verification OTP",

            "htmlContent": (
                "<html><body>"
                "<h2>MindMesh Email Verification</h2>"
                "<p>Your OTP is:</p>"
                f"<h1>{otp}</h1>"
                "<p>Please use this code to verify your email.</p>"
                "</body></html>"
            )
        }

        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=20
        )

        if response.status_code in (
            200,
            201,
            202
        ):

            otp_storage[email] = {
                "otp": otp,
                "created_at": time.time()
            }

            return jsonify({
                "success": True,
                "message": "OTP sent successfully"
            }), 200

        print(
            "Brevo error:",
            response.status_code,
            response.text
        )

        return jsonify({
            "success": False,
            "message": "Could not send OTP email"
        }), 502

    except requests.RequestException as e:

        print(
            "OTP email request error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Email service connection failed"
        }), 502

    except Exception as e:

        print(
            "Send OTP error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "An unexpected error occurred"
        }), 500


# ==========================================
# VERIFY OTP AND CREATE ACCOUNT
# ==========================================

@app.route("/verify-otp", methods=["POST"])
def verify_otp():

    try:

        data = get_json_body()

        email = str(
            data.get("email", "")
        ).strip().lower()

        entered_otp = str(
            data.get("otp", "")
        ).strip()

        name = str(
            data.get("name", "")
        ).strip()

        password = str(
            data.get("password", "")
        )

        if not email or not entered_otp or not name or not password:

            return jsonify({
                "success": False,
                "message": "Name, email, password and OTP are required."
            }), 400

        if len(password) < 8:

            return jsonify({
                "success": False,
                "message": "Password must contain at least 8 characters."
            }), 400

        stored_data = otp_storage.get(email)

        if not stored_data:

            return jsonify({
                "success": False,
                "message": "Please request an OTP first."
            }), 400

        if time.time() - stored_data["created_at"] > 300:

            otp_storage.pop(email, None)

            return jsonify({
                "success": False,
                "message": "OTP expired. Please request a new OTP."
            }), 400

        if entered_otp != stored_data["otp"]:

            return jsonify({
                "success": False,
                "message": "Invalid OTP."
            }), 400

        with get_db_connection() as db:

            existing = db.execute(
                "SELECT id FROM users WHERE email = ?",
                (email,)
            ).fetchone()

            if existing:

                otp_storage.pop(email, None)

                return jsonify({
                    "success": False,
                    "message": "An account already exists. Please log in."
                }), 409

            # Store only the password hash, never the plain password.
            password_hash = generate_password_hash(
                password
            )

            db.execute(
                """
                INSERT INTO users (
                    name,
                    email,
                    password_hash
                )
                VALUES (?, ?, ?)
                """,
                (
                    name,
                    email,
                    password_hash
                )
            )

        otp_storage.pop(email, None)

        token = create_session(email)

        return jsonify({

            "success": True,

            "message": "Account created successfully.",

            "name": name,

            "email": email,

            "token": token

        }), 200

    except sqlite3.IntegrityError:

        return jsonify({
            "success": False,
            "message": "An account already exists with this email."
        }), 409

    except Exception as e:

        print(
            "Verify OTP error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Account creation failed. Please try again."
        }), 500


# ==========================================
# USER LOGIN
# ==========================================

@app.route("/login", methods=["POST"])
def login():

    try:

        data = get_json_body()

        email = str(
            data.get("email", "")
        ).strip().lower()

        password = str(
            data.get("password", "")
        )

        if not email or not password:

            return jsonify({
                "success": False,
                "message": "Email and password are required."
            }), 400

        with get_db_connection() as db:

            user = db.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if not user:

            return jsonify({
                "success": False,
                "message": "Account not found. Please sign up first."
            }), 401

        if not check_password_hash(
            user["password_hash"],
            password
        ):

            return jsonify({
                "success": False,
                "message": "Incorrect email or password."
            }), 401

        token = create_session(email)

        return jsonify({

            "success": True,

            "message": "Login successful.",

            "name": user["name"],

            "email": email,

            "token": token

        }), 200

    except Exception as e:

        print(
            "Login error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Login failed. Please try again."
        }), 500


# ==========================================
# VERIFY SESSION
# ==========================================

@app.route("/verify-session", methods=["POST"])
def verify_session():

    try:

        data = get_json_body()

        token = str(
            data.get("token", "")
        ).strip()

        if not token:

            return jsonify({
                "success": False,
                "message": "Session token is required."
            }), 400

        with get_db_connection() as db:

            session = db.execute(
                """
                SELECT user_sessions.email, users.name
                FROM user_sessions
                JOIN users
                ON user_sessions.email = users.email
                WHERE user_sessions.token = ?
                """,
                (token,)
            ).fetchone()

        if not session:

            return jsonify({
                "success": False,
                "message": "Invalid session. Please log in again."
            }), 401

        return jsonify({

            "success": True,

            "name": session["name"],

            "email": session["email"]

        }), 200

    except Exception as e:

        print(
            "Session verification error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Could not verify session."
        }), 500


# ==========================================
# LOGOUT
# ==========================================

@app.route("/logout", methods=["POST"])
def logout():

    try:

        data = get_json_body()

        token = str(
            data.get("token", "")
        ).strip()

        if not token:

            return jsonify({
                "success": False,
                "message": "Session token is required."
            }), 400

        with get_db_connection() as db:

            db.execute(
                "DELETE FROM user_sessions WHERE token = ?",
                (token,)
            )

        return jsonify({
            "success": True,
            "message": "Logged out successfully."
        }), 200

    except Exception as e:

        print(
            "Logout error:",
            repr(e)
        )

        return jsonify({
            "success": False,
            "message": "Logout failed."
        }), 500


# ==========================================
# HOME ROUTE
# ==========================================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message": "MindMesh backend is running"

    }), 200


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
