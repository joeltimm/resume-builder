# resume-builder/backend/app.py
# It uses the Flask framework to create a simple API.

import os
import psycopg2
import json
import requests
import io
import click
import traceback
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import nltk
# FIXED: Added Union for type hinting
from typing import Union

from scoring_logic import calculate_weighted_match_score
from llm_integration import improve_resume_bullet, find_duplicate_entries, get_available_models, analyze_job_description_with_llm
from resume_generator import generate_ats_resume_text

# Download required NLTK data for text processing
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: Could not download NLTK data: {e}")

# --- Initialization ---
app = Flask(__name__)
CORS(app)

# Load the pre-trained Sentence Transformer model.
model = SentenceTransformer('all-MiniLM-L6-v2')

# FIXED: Create a type alias for Flask's common return patterns to satisfy Pylance.
ResponseValue = Union[Response, tuple[Response, int]]


# --- Database Functions ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="postgres",
            database=os.environ.get('POSTGRES_DB'),
            user=os.environ.get('POSTGRES_USER'),
            password=os.environ.get('POSTGRES_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None


def setup_database():
    """Ensures all required tables exist in the database."""
    conn = get_db_connection()
    if not conn:
        print("Warning: Could not connect to database. Skipping table setup.")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                # --- Resume Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS resume (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL
                    );
                ''')
                cur.execute('SELECT id FROM resume WHERE id = 1;')
                if cur.fetchone() is None:
                    cur.execute('INSERT INTO resume (id, content) VALUES (1, %s);', ('{}',))

                # --- Skills Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS skills (
                        id SERIAL PRIMARY KEY,
                        skill_text TEXT NOT NULL UNIQUE,
                        embedding TEXT
                    );
                ''')

                # --- Work Experience Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS work_experience (
                        id SERIAL PRIMARY KEY,
                        job_title TEXT NOT NULL,
                        company TEXT NOT NULL,
                        location TEXT,
                        dates TEXT,
                        description TEXT,
                        embedding TEXT
                    );
                ''')

                # --- Accomplishments Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS accomplishments (
                        id SERIAL PRIMARY KEY,
                        accomplishment_text TEXT NOT NULL UNIQUE,
                        embedding TEXT,
                        work_experience_id INTEGER REFERENCES work_experience(id)
                    );
                ''')

                # Check if work_experience_id column exists, if not add it.
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = 'accomplishments' AND column_name = 'work_experience_id';
                """)
                if cur.fetchone() is None:
                    cur.execute("""
                        ALTER TABLE accomplishments
                        ADD COLUMN work_experience_id INTEGER REFERENCES work_experience(id);
                    """)

                # --- Professional Summaries Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS professional_summaries (
                        id SERIAL PRIMARY KEY,
                        summary_text TEXT NOT NULL UNIQUE,
                        embedding TEXT
                    );
                ''')

                # --- Education Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS education (
                        id SERIAL PRIMARY KEY,
                        degree TEXT NOT NULL,
                        institution TEXT NOT NULL,
                        embedding TEXT
                    );
                ''')

                # --- Technical Projects Table ---
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS technical_projects (
                        id SERIAL PRIMARY KEY,
                        project_name TEXT NOT NULL,
                        description TEXT,
                        tools TEXT,
                        embedding TEXT
                    );
                ''')

        print("Database setup completed successfully.")
    except Exception as e:
        print(f"Error during database setup: {e}")


@click.command('init-db')
def init_db_command():
    """Creates new tables in the database if they don't already exist."""
    setup_database()
    click.echo('Initialized the database.')


app.cli.add_command(init_db_command)


# --- API Routes ---

@app.route('/resume', methods=['GET', 'POST'])
def handle_resume() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                if request.method == 'POST':
                    resume_data = request.get_json()
                    if not resume_data:
                        return jsonify({"error": "No resume data provided from backend"}), 400
                    cur.execute(
                        "INSERT INTO resume (id, content) VALUES (1, %s) ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content;",
                        (json.dumps(resume_data),)
                    )
                    return jsonify({"message": "Resume saved successfully"}), 201

                if request.method == 'GET':
                    cur.execute("SELECT content FROM resume WHERE id = 1;")
                    resume_data = cur.fetchone()

                    if resume_data and resume_data[0]:
                        return jsonify(json.loads(resume_data[0]))
                    else:
                        return jsonify({"message": "No resume data found"}), 404
    except Exception as e:
        print(f"Error in /resume route: {e}")
        return jsonify({"error": "Internal server error"}), 500
    # This part is unreachable, but adding a fallback return satisfies some strict linters.
    return jsonify({"error": "An unexpected error occurred."}), 500

# --- API for Skills ---

@app.route('/api/skills', methods=['POST'])
def add_skill() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    skill_text = data.get('skill_text')
    if not skill_text:
        return jsonify({"error": "Skill text is required"}), 400

    embedding = model.encode(skill_text).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO skills (skill_text, embedding) VALUES (%s, %s) RETURNING id;',
                    (skill_text, json.dumps(embedding))
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new skill."}), 500
                new_id = result[0]
        return jsonify({"message": "Skill added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This skill already exists"}), 409
    except Exception as e:
        print(f"Error inserting skill: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/skills', methods=['GET'])
def get_skills() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, skill_text FROM skills ORDER BY skill_text;')
                skills = [{"id": row[0], "skill_text": row[1]} for row in cur.fetchall()]
        return jsonify(skills)
    except Exception as e:
        print(f"Error fetching skills: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/skills/<int:skill_id>', methods=['DELETE'])
def delete_skill(skill_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM skills WHERE id = %s;', (skill_id,))
        return jsonify({"message": "Skill deleted successfully"})
    except Exception as e:
        print(f"Error deleting skill: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- API for Accomplishments ---

@app.route('/api/accomplishments', methods=['POST'])
def add_accomplishment() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    accomplishment_text = data.get('accomplishment_text')
    work_experience_id = data.get('work_experience_id')

    if not accomplishment_text:
        return jsonify({"error": "Accomplishment text is required"}), 400

    embedding = model.encode(accomplishment_text).tolist()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO accomplishments (accomplishment_text, embedding, work_experience_id) VALUES (%s, %s, %s) RETURNING id;',
                    (accomplishment_text, json.dumps(embedding), work_experience_id)
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new accomplishment."}), 500
                new_id = result[0]
        return jsonify({"message": "Accomplishment added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This accomplishment already exists"}), 409
    except Exception as e:
        print(f"Error inserting accomplishment: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/accomplishments', methods=['GET'])
def get_accomplishments() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, accomplishment_text, work_experience_id FROM accomplishments ORDER BY accomplishment_text;')
                accomplishments = [
                    {"id": row[0], "accomplishment_text": row[1], "work_experience_id": row[2]}
                    for row in cur.fetchall()
                ]
        return jsonify(accomplishments)
    except Exception as e:
        print(f"Error fetching accomplishments: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/accomplishments/<int:accomplishment_id>', methods=['DELETE'])
def delete_accomplishment(accomplishment_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM accomplishments WHERE id = %s;', (accomplishment_id,))
        return jsonify({"message": "Accomplishment deleted successfully"})
    except Exception as e:
        print(f"Error deleting accomplishment: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- API for Professional Summaries ---

@app.route('/api/professional_summaries', methods=['POST'])
def add_professional_summary() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    summary_text = data.get('summary_text')
    if not summary_text:
        return jsonify({"error": "Summary text is required"}), 400

    embedding = model.encode(summary_text).tolist()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO professional_summaries (summary_text, embedding) VALUES (%s, %s) RETURNING id;',
                    (summary_text, json.dumps(embedding))
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new summary."}), 500
                new_id = result[0]
        return jsonify({"message": "Professional summary added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This professional summary already exists"}), 409
    except Exception as e:
        print(f"Error inserting summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/professional_summaries', methods=['GET'])
def get_professional_summaries() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, summary_text FROM professional_summaries ORDER BY summary_text;')
                summaries = [{"id": row[0], "summary_text": row[1]} for row in cur.fetchall()]
        return jsonify(summaries)
    except Exception as e:
        print(f"Error fetching summaries: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/professional_summaries/<int:summary_id>', methods=['DELETE'])
def delete_professional_summary(summary_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM professional_summaries WHERE id = %s;', (summary_id,))
        return jsonify({"message": "Professional summary deleted successfully"})
    except Exception as e:
        print(f"Error deleting summary: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- API for Work Experience ---

@app.route('/api/work_experience', methods=['POST'])
def add_work_experience() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    job_title = data.get('job_title')
    company = data.get('company')
    description = data.get('description', '')

    if not job_title or not company:
        return jsonify({"error": "Job title and company are required"}), 400

    text_to_embed = f"{job_title} {description}"
    embedding = model.encode(text_to_embed).tolist()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO work_experience (job_title, company, location, dates, description, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;',
                    (job_title, company, data.get('location'), data.get('dates'), description, json.dumps(embedding))
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new work experience."}), 500
                new_id = result[0]
        return jsonify({"message": "Work experience added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This work experience already exists"}), 409
    except Exception as e:
        print(f"Error inserting work experience: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/work_experience', methods=['GET'])
def get_work_experience() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, job_title, company, location, dates, description FROM work_experience ORDER BY id;')
                experience = [
                    {
                        "id": row[0],
                        "job_title": row[1],
                        "company": row[2],
                        "location": row[3],
                        "dates": row[4],
                        "description": row[5],
                    }
                    for row in cur.fetchall()
                ]
        return jsonify(experience)
    except Exception as e:
        print(f"Error fetching work experience: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/work_experience/<int:experience_id>', methods=['DELETE'])
def delete_work_experience(experience_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM accomplishments WHERE work_experience_id = %s;', (experience_id,))
                cur.execute('DELETE FROM work_experience WHERE id = %s;', (experience_id,))
        return jsonify({"message": "Work experience deleted successfully"})
    except Exception as e:
        print(f"Error deleting work experience: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- API for Education ---

@app.route('/api/education', methods=['POST'])
def add_education() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    degree = data.get('degree')
    institution = data.get('institution')

    if not degree or not institution:
        return jsonify({"error": "Degree and institution are required"}), 400

    text_to_embed = f"{degree} {institution}"
    embedding = model.encode(text_to_embed).tolist()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO education (degree, institution, embedding) VALUES (%s, %s, %s) RETURNING id;',
                    (degree, institution, json.dumps(embedding))
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new education entry."}), 500
                new_id = result[0]
        return jsonify({"message": "Education added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This education entry already exists"}), 409
    except Exception as e:
        print(f"Error inserting education: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/education', methods=['GET'])
def get_education() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, degree, institution FROM education ORDER BY id;')
                education = [{"id": row[0], "degree": row[1], "institution": row[2]} for row in cur.fetchall()]
        return jsonify(education)
    except Exception as e:
        print(f"Error fetching education: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/education/<int:education_id>', methods=['DELETE'])
def delete_education(education_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM education WHERE id = %s;', (education_id,))
        return jsonify({"message": "Education deleted successfully"})
    except Exception as e:
        print(f"Error deleting education: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- API for Technical Projects ---

@app.route('/api/technical_projects', methods=['POST'])
def add_technical_project() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    project_name = data.get('project_name')
    description = data.get('description', '')
    tools = data.get('tools', '')

    if not project_name:
        return jsonify({"error": "Project name is required"}), 400

    text_to_embed = f"{project_name} {description} {tools}"
    embedding = model.encode(text_to_embed).tolist()
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    'INSERT INTO technical_projects (project_name, description, tools, embedding) VALUES (%s, %s, %s, %s) RETURNING id;',
                    (project_name, description, tools, json.dumps(embedding))
                )
                # FIXED: Check if fetchone() returns None before subscripting
                result = cur.fetchone()
                if result is None:
                    return jsonify({"error": "Failed to create new technical project."}), 500
                new_id = result[0]
        return jsonify({"message": "Technical project added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({"error": "This technical project already exists"}), 409
    except Exception as e:
        print(f"Error inserting technical project: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/technical_projects', methods=['GET'])
def get_technical_projects() -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT id, project_name, description, tools FROM technical_projects ORDER BY id;')
                projects = [
                    {
                        "id": row[0],
                        "project_name": row[1],
                        "description": row[2],
                        "tools": row[3],
                    }
                    for row in cur.fetchall()
                ]
        return jsonify(projects)
    except Exception as e:
        print(f"Error fetching technical projects: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/technical_projects/<int:project_id>', methods=['DELETE'])
def delete_technical_project(project_id: int) -> ResponseValue: # FIXED: Added return type hint
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('DELETE FROM technical_projects WHERE id = %s;', (project_id,))
        return jsonify({"message": "Technical project deleted successfully"})
    except Exception as e:
        print(f"Error deleting technical project: {e}")
        return jsonify({"error": "Internal server error"}), 500


# --- MODIFIED: API for AI Matching ---

@app.route('/api/match', methods=['POST'])
def match_skills() -> ResponseValue: # FIXED: Added return type hint
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid request: No JSON body provided."}), 400

        job_description = data.get('job_description')
        model_name = data.get('model_name')

        if not job_description:
            return jsonify({"error": "Job description is required"}), 400

        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database connection failed"}), 500

        with conn:
            with conn.cursor() as cur:
                # âœ… MODIFIED: Fetch IDs along with text
                cur.execute('SELECT id, skill_text FROM skills;')
                skills = [{"id": row[0], "text": row[1]} for row in cur.fetchall()]
                cur.execute('SELECT id, accomplishment_text FROM accomplishments;')
                accomplishments = [{"id": row[0], "text": row[1]} for row in cur.fetchall()]

        user_data = {
            # âœ… MODIFIED: Pass only the text to the LLM
            "skills": [s['text'] for s in skills],
            "accomplishments": [a['text'] for a in accomplishments]
        }

        analysis_result = analyze_job_description_with_llm(job_description, user_data, model_name)

        if "error" in analysis_result:
            return jsonify(analysis_result), 500

        # âœ… MODIFIED: Create a map of text to the full object (with numeric ID)
        all_items_map = {}
        for skill in skills:
            all_items_map[skill['text']] = {"id": f"skill-{skill['id']}", "text": skill['text'], "type": "skill"}
        for acc in accomplishments:
            all_items_map[acc['text']] = {"id": f"accomplishment-{acc['id']}", "text": acc['text'], "type": "accomplishment"}

        suggestions_with_details = []
        if "suggestions" in analysis_result:
            for suggestion_text in analysis_result["suggestions"]:
                if suggestion_text in all_items_map:
                    item_details = all_items_map[suggestion_text]
                    # This score is a placeholder; you might replace with actual scoring later
                    item_details['score'] = 0.8 
                    suggestions_with_details.append(item_details)

        return jsonify({
            "suggestions": suggestions_with_details,
            "missing_keywords": analysis_result.get("missing_keywords", [])
        })
    except Exception as e:
        error_message = traceback.format_exc()
        print(f"FORCE-LOGGING ERROR: {error_message}")
        return jsonify({
            "error": "An unexpected error occurred on the backend.",
            "traceback": error_message
        }), 500


@app.route('/calculate-score', methods=['POST'])
def get_score() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    resume_text = data.get('resumeText')
    jd_text = data.get('jobDescriptionText')

    if not resume_text or not jd_text:
        return jsonify({"error": "Missing resume or job description text"}), 400

    score_data_json = calculate_weighted_match_score(resume_text, jd_text)
    return jsonify(json.loads(score_data_json))


@app.route('/api/export-pdf', methods=['POST'])
def export_pdf() -> ResponseValue: # FIXED: Added return type hint
    resume_data = request.get_json()
    if not resume_data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute('SELECT degree, institution FROM education ORDER BY id;')
                education_entries = cur.fetchall()

        # --- Format HTML (omitted for brevity, no changes needed here) ---
        skills_html = ''.join([f'<span style="background-color: #eee; padding: 2px 6px; border-radius: 4px; margin-right: 5px;">{skill}</span>' for skill in resume_data.get('skills', [])])
        experience_html = ''
        for exp in resume_data.get('experience', []):
            accomplishments_html = ''.join([
                f'<li style="margin-bottom: 5px;">{acc["accomplishment_text"]}</li>'
                for acc in resume_data.get('accomplishments', [])
                if acc.get("work_experience_id") == exp.get("id")
            ])
            description_html = ''
            if exp.get('description'):
                description_text = exp.get('description', '').replace('\n', '<br>')
                description_html = f"""
                <div style="margin-left: 20px; font-style: italic; font-size: 0.9em; color: #555;">
                     <p>{description_text}</p>
                </div>
                """
            experience_html += f"""
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: baseline;">
                    <h4 style="margin: 0; font-size: 1.1em; font-weight: bold;">{exp.get('job_title', '')} | {exp.get('company', '')} - {exp.get('location', '')}</h4>
                    <p style="margin: 0; font-style: italic;">{exp.get('dates', '')}</p>
                </div>
                {description_html}
                <ul style="margin-top: 5px; list-style-position: inside;">{accomplishments_html}</ul>
            </div>
            """
        education_html = ''.join([f"<p>{degree} - {institution}</p>" for degree, institution in education_entries])
        projects_html = ''
        for proj in resume_data.get('projects', []):
            project_desc = proj.get('description', '').replace('\n', '<br>')
            projects_html += f"""
            <div style="margin-bottom: 15px;">
                <h4 style="margin: 0; font-size: 1.1em;">{proj.get('project_name', '')}</h4>
                <p style="margin-top: 5px;">{project_desc}</p>
                <p><b>Tools:</b> {proj.get('tools', '')}</p>
            </div>
            """
        html_content = f"""
        <html><head><style>body {{ font-family: sans-serif; font-size: 11pt; }} h1, h2, h3, h4, p {{ margin: 0; padding: 0; }} hr {{ border: none; border-top: 1px solid #ccc; margin: 15px 0; }}</style></head>
        <body>
            <div style="text-align: center;"><h1 style="font-size: 2.5em;">{resume_data.get('name', 'Your Name')}</h1><p>{resume_data.get('email', '')} | {resume_data.get('phone', '')} | {resume_data.get('linkedin', '')} | {resume_data.get('github', '')} | {resume_data.get('location', '')} | {resume_data.get('portfolio', '')}</p></div><hr>
            <div><h3>Summary</h3><p>{resume_data.get('summary', '')}</p></div><hr>
            <div><h3>Skills</h3><p>{skills_html}</p></div><hr>
            <div><h3>Work Experience</h3>{experience_html}</div><hr>
            <div><h3>Technical Projects</h3>{projects_html}</div><hr>
            <div><h3>Education</h3>{education_html}</div>
        </body></html>
        """

        # --- Call Stirling-PDF ---
        stirling_url = 'http://stirling-pdf:8080/api/v1/convert/html/pdf'
        files = {'fileInput': ('resume.html', html_content, 'text/html')}

        response = requests.post(stirling_url, files=files)
        response.raise_for_status()

        return send_file(
            io.BytesIO(response.content),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='resume.pdf'
        )
    except requests.exceptions.RequestException as e:
        print(f"Error calling Stirling-PDF: {e}")
        return jsonify({"error": "Failed to generate PDF"}), 500
    except Exception as e:
        print(f"Unexpected error in /api/export-pdf: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/api/models', methods=['GET'])
def get_llm_models() -> ResponseValue: # FIXED: Added return type hint
    llm_mode = os.environ.get("LLM_MODE", "local").lower()

    if llm_mode == "local":
        try:
            models = get_available_models()
            default_model = os.environ.get("LM_STUDIO_DEFAULT_MODEL", "qwen2.5-32b-instruct")
            return jsonify({
                "mode": "local",
                "models": models,
                "default_model": default_model,
                "llama_cpp_url": os.environ.get("LLM_URL", "http://100.98.99.49:8081")
            })
        except Exception as e:
            return jsonify({"error": f"Failed to fetch models: {str(e)}"}), 500
    else:
        return jsonify({
            "mode": "production",
            "models": ["claude-3-sonnet-20240229"],
            "default_model": "claude-3-sonnet-20240229"
        })


@app.route('/improve-bullet', methods=['POST'])
def get_improved_bullet() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    bullet = data.get('bulletPoint')
    job_title = data.get('jobTitle')
    industry = data.get('industry')
    job_description = data.get('jobDescription')
    model_name = data.get('modelName')

    if not all([bullet, job_title, industry, job_description]):
        return jsonify({"error": "Missing required fields"}), 400

    improved_bullet = improve_resume_bullet(bullet, job_title, industry, job_description, model_name)

    return jsonify({
        "improved_bullet": improved_bullet,
        "model_used": model_name or "default",
        "llm_mode": os.environ.get("LLM_MODE", "production")
    })


@app.route('/check-duplicates', methods=['POST'])
def check_for_duplicates() -> ResponseValue: # FIXED: Added return type hint
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request: No JSON body provided."}), 400

    bullet_points = data.get('bulletPoints')
    if not isinstance(bullet_points, list):
        return jsonify({"error": "Expected a list of bullet points"}), 400

    duplicate_data = find_duplicate_entries(bullet_points)
    return jsonify({"duplicates": duplicate_data})


@app.route('/generate-ats-resume', methods=['POST'])
def create_ats_resume() -> ResponseValue: # FIXED: Added return type hint
    resume_data = request.get_json()
    if not resume_data:
        return jsonify({"error": "No resume data provided"}), 400

    ats_text = generate_ats_resume_text(resume_data)
    return Response(
        ats_text,
        mimetype="text/plain",
        headers={"Content-disposition": "attachment; filename=ats_resume.txt"}
    )
