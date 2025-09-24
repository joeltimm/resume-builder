#resume-builder/backend/app.py
# It uses the Flask framework to create a simple API.

import os
import psycopg2
import json # Import the json library
import numpy as np # Import numpy for array operations
import requests
import io
import click
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from sentence_transformers import SentenceTransformer, util # Import sentence-transformers
import re

# --- Initialization ---
# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
CORS(app)

# Load the pre-trained Sentence Transformer model.
# This model is optimized for semantic similarity tasks.
# The first time this runs, it will download the model, so it might take a moment.
model = SentenceTransformer('all-MiniLM-L6-v2')


# --- Database Functions ---

def get_db_connection():
    """Establishes a connection to the PostgreSQL database."""
    try:
        # Connect to the database. The 'host' is the name of the postgres service
        # in the docker-compose.yml file. Docker's internal DNS handles the resolution.
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

#resume-builder/backend/app.py

def setup_database():
    #Ensures all required tables exist in the database.
    conn = get_db_connection()
    if conn: # <-- This is the important check
        try:
            cur = conn.cursor()

            # --- Resume Table (for personal details) ---
            cur.execute('''
                CREATE TABLE IF NOT EXISTS resume (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL
                );
            ''')
            # Check if the default resume entry exists, if not, create one.
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
                    embedding TEXT
                );
            ''')
            
            # Check if work_experience_id column exists in accomplishments table, if not add it.
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

            conn.commit()
        finally:
            # Ensure the connection is closed even if errors occur
            cur.close()
            conn.close()

# --- Create a Flask CLI command to set up the database ---

@click.command('init-db')
def init_db_command():
    """Creates new tables in the database if they don't already exist."""
    setup_database()
    click.echo('Initialized the database.')

app.cli.add_command(init_db_command)

# --- API Routes ---

@app.route('/resume', methods=['GET', 'POST'])
def handle_resume():
    """
    This is the main API endpoint for personal details.
    It handles GET requests to load resume data and POST requests to save it.
    """
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()

    if request.method == 'POST':
        resume_data = request.get_json()
        cur.execute(
            "INSERT INTO resume (id, content) VALUES (1, %s) ON CONFLICT (id) DO UPDATE SET content = EXCLUDED.content;",
            (json.dumps(resume_data),)
        )
        conn.commit()

        cur.close()
        conn.close()
        return jsonify({"message": "Resume saved successfully"})

    if request.method == 'GET':
        cur.execute("SELECT content FROM resume WHERE id = 1;")
        resume_data = cur.fetchone()

        cur.close()
        conn.close()

        if resume_data and resume_data[0]:
            return jsonify(json.loads(resume_data[0]))
        else:
            return jsonify({"message": "No resume data found"}), 404

# --- API for Skills ---

@app.route('/api/skills', methods=['POST'])
def add_skill():
    """Adds a new skill and its embedding to the database."""
    data = request.json
    skill_text = data.get('skill_text')

    if not skill_text:
        return jsonify({"error": "Skill text is required"}), 400

    # Generate the embedding for the new skill text
    embedding = model.encode(skill_text).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    try:
        # Store both the text and the JSON-serialized embedding
        cur.execute(
            'INSERT INTO skills (skill_text, embedding) VALUES (%s, %s) RETURNING id;',
            (skill_text, json.dumps(embedding))
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Skill added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This skill already exists"}), 409
    finally:
        cur.close()
        conn.close()


@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Fetches all skills from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('SELECT id, skill_text FROM skills ORDER BY skill_text;')
    skills = [{"id": row[0], "skill_text": row[1]} for row in cur.fetchall()]
    cur.close()
    conn.close()

    return jsonify(skills)

@app.route('/api/skills/<int:skill_id>', methods=['DELETE'])
def delete_skill(skill_id):
    """Deletes a skill from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM skills WHERE id = %s;', (skill_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Skill deleted successfully"})

# --- API for Accomplishments ---

@app.route('/api/accomplishments', methods=['POST'])
def add_accomplishment():
    """Adds a new accomplishment and its embedding to the database."""
    data = request.json
    accomplishment_text = data.get('accomplishment_text')
    work_experience_id = data.get('work_experience_id')

    if not accomplishment_text:
        return jsonify({"error": "Accomplishment text is required"}), 400

    # Generate the embedding for the new accomplishment text
    embedding = model.encode(accomplishment_text).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    try:
        # Store both the text and the JSON-serialized embedding
        cur.execute(
            'INSERT INTO accomplishments (accomplishment_text, embedding, work_experience_id) VALUES (%s, %s, %s) RETURNING id;',
            (accomplishment_text, json.dumps(embedding), work_experience_id)
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Accomplishment added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This accomplishment already exists"}), 409
    finally:
        cur.close()
        conn.close()

@app.route('/api/accomplishments', methods=['GET'])
def get_accomplishments():
    """Fetches all accomplishments from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('SELECT id, accomplishment_text, work_experience_id FROM accomplishments ORDER BY accomplishment_text;')
    accomplishments = [{"id": row[0], "accomplishment_text": row[1], "work_experience_id": row[2]} for row in cur.fetchall()]
    cur.close()
    conn.close()

    return jsonify(accomplishments)

@app.route('/api/accomplishments/<int:accomplishment_id>', methods=['DELETE'])
def delete_accomplishment(accomplishment_id):
    """Deletes an accomplishment from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM accomplishments WHERE id = %s;', (accomplishment_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Accomplishment deleted successfully"})

# --- API for Professional Summaries ---

@app.route('/api/professional_summaries', methods=['POST'])
def add_professional_summary():
    """Adds a new professional summary and its embedding to the database."""
    data = request.json
    summary_text = data.get('summary_text')

    if not summary_text:
        return jsonify({"error": "Summary text is required"}), 400

    embedding = model.encode(summary_text).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO professional_summaries (summary_text, embedding) VALUES (%s, %s) RETURNING id;',
            (summary_text, json.dumps(embedding))
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Professional summary added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This professional summary already exists"}), 409
    finally:
        cur.close()
        conn.close()

@app.route('/api/professional_summaries', methods=['GET'])
def get_professional_summaries():
    """Fetches all professional summaries from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('SELECT id, summary_text FROM professional_summaries ORDER BY summary_text;')
    summaries = [{"id": row[0], "summary_text": row[1]} for row in cur.fetchall()]
    cur.close()
    conn.close()

    return jsonify(summaries)

@app.route('/api/professional_summaries/<int:summary_id>', methods=['DELETE'])
def delete_professional_summary(summary_id):
    """Deletes a professional summary from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM professional_summaries WHERE id = %s;', (summary_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Professional summary deleted successfully"})


# --- API for Work Experience ---

@app.route('/api/work_experience', methods=['POST'])
def add_work_experience():
    """Adds a new work experience and its embedding to the database."""
    data = request.json
    job_title = data.get('job_title')
    company = data.get('company')
    description = data.get('description', '') # Description is optional

    if not job_title or not company:
        return jsonify({"error": "Job title and company are required"}), 400

    # For work experience, we'll embed the job title and description together
    text_to_embed = f"{job_title} {description}"
    embedding = model.encode(text_to_embed).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO work_experience (job_title, company, location, dates, description, embedding) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id;',
            (job_title, company, data.get('location'), data.get('dates'), description, json.dumps(embedding))
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Work experience added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This work experience already exists"}), 409
    finally:
        cur.close()
        conn.close()

@app.route('/api/work_experience', methods=['GET'])
def get_work_experience():
    """Fetches all work experience from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
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
    cur.close()
    conn.close()

    return jsonify(experience)

@app.route('/api/work_experience/<int:experience_id>', methods=['DELETE'])
def delete_work_experience(experience_id):
    """Deletes a work experience from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM accomplishments WHERE work_experience_id = %s;', (experience_id,))
    cur.execute('DELETE FROM work_experience WHERE id = %s;', (experience_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Work experience deleted successfully"})

# --- API for Education ---

@app.route('/api/education', methods=['POST'])
def add_education():
    """Adds a new education entry and its embedding to the database."""
    data = request.json
    degree = data.get('degree')
    institution = data.get('institution')

    if not degree or not institution:
        return jsonify({"error": "Degree and institution are required"}), 400

    text_to_embed = f"{degree} {institution}"
    embedding = model.encode(text_to_embed).tolist()

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO education (degree, institution, embedding) VALUES (%s, %s, %s) RETURNING id;',
            (degree, institution, json.dumps(embedding))
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Education added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This education entry already exists"}), 409
    finally:
        cur.close()
        conn.close()

@app.route('/api/education', methods=['GET'])
def get_education():
    """Fetches all education entries from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('SELECT id, degree, institution FROM education ORDER BY id;')
    education = [{"id": row[0], "degree": row[1], "institution": row[2]} for row in cur.fetchall()]
    cur.close()
    conn.close()

    return jsonify(education)

@app.route('/api/education/<int:education_id>', methods=['DELETE'])
def delete_education(education_id):
    """Deletes an education entry from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM education WHERE id = %s;', (education_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Education deleted successfully"})


# --- API for Technical Projects ---

@app.route('/api/technical_projects', methods=['POST'])
def add_technical_project():
    """Adds a new technical project and its embedding to the database."""
    data = request.json
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

    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO technical_projects (project_name, description, tools, embedding) VALUES (%s, %s, %s, %s) RETURNING id;',
            (project_name, description, tools, json.dumps(embedding))
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Technical project added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "This technical project already exists"}), 409
    finally:
        cur.close()
        conn.close()

@app.route('/api/technical_projects', methods=['GET'])
def get_technical_projects():
    """Fetches all technical projects from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
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
    cur.close()
    conn.close()

    return jsonify(projects)

@app.route('/api/technical_projects/<int:project_id>', methods=['DELETE'])
def delete_technical_project(project_id):
    """Deletes a technical project from the database."""
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500

    cur = conn.cursor()
    cur.execute('DELETE FROM technical_projects WHERE id = %s;', (project_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Technical project deleted successfully"})


# --- NEW: Keyword Extraction Function ---
def extract_keywords(text):
    """Extracts potentially important keywords from text."""
    # This is a simple implementation. More sophisticated methods exist.
    stop_words = set([
        "a", "an", "the", "and", "but", "or", "for", "in", "on", "at", "to", "with", 
        "about", "of", "is", "are", "was", "were", "be", "been", "being", "have", 
        "has", "had", "do", "does", "did", "will", "would", "should", "can", "could",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"
    ])
    
    words = re.findall(r'\b\w+\b', text.lower())
    # Consider words that are longer than 2 characters and not stop words
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    # Simple frequency count
    freq_dist = {kw: keywords.count(kw) for kw in set(keywords)}
    # Return keywords that appear more than once, sorted by frequency
    return sorted([kw for kw, freq in freq_dist.items() if freq > 1], key=lambda kw: freq_dist[kw], reverse=True)


# --- API for AI Matching ---

@app.route('/api/match', methods=['POST'])
def match_skills():
    """
    Finds the most relevant skills and accomplishments for a given job description
    and provides an analysis of missing keywords.
    """
    data = request.json
    job_description = data.get('job_description')
    top_n = data.get('limit', 20) 

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    # 1. Generate embedding for the job description
    job_embedding = model.encode(job_description, convert_to_tensor=True)

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    cur = conn.cursor()

    # 2. Fetch all items from the database
    cur.execute('SELECT id, skill_text, embedding FROM skills;')
    skills = cur.fetchall()
    cur.execute('SELECT id, accomplishment_text, embedding FROM accomplishments;')
    accomplishments = cur.fetchall()
    cur.execute('SELECT id, summary_text, embedding FROM professional_summaries;')
    summaries = cur.fetchall()
    cur.execute('SELECT id, job_title, company, description, embedding FROM work_experience;')
    experiences = cur.fetchall()
    cur.execute('SELECT id, degree, institution, embedding FROM education;')
    educations = cur.fetchall()
    cur.execute('SELECT id, project_name, description, tools, embedding FROM technical_projects;')
    projects = cur.fetchall()
    cur.close()
    conn.close()

    all_items = []

    def process_items(items, item_type, get_text, get_embedding):
        if not items:
            return
        
        embeddings = np.array([json.loads(get_embedding(item)) for item in items if get_embedding(item)]).astype(np.float32)
        if embeddings.size == 0:
            return

        scores = util.cos_sim(job_embedding, embeddings)[0]
        for i, item in enumerate(items):
            if get_embedding(item):
                all_items.append({
                    "id": f"{item_type}-{item[0]}",
                    "text": get_text(item),
                    "score": float(scores[i]),
                    "type": item_type
                })

    process_items(skills, 'skill', lambda item: item[1], lambda item: item[2])
    process_items(accomplishments, 'accomplishment', lambda item: item[1], lambda item: item[2])
    process_items(summaries, 'summary', lambda item: item[1], lambda item: item[2])
    process_items(experiences, 'experience', lambda item: f"{item[1]} at {item[2]}", lambda item: item[4])
    process_items(educations, 'education', lambda item: f"{item[1]}, {item[2]}", lambda item: item[3])
    process_items(projects, 'project', lambda item: item[1], lambda item: item[4])

    # Sort all items by score in descending order
    sorted_items = sorted(all_items, key=lambda x: x['score'], reverse=True)
    top_results = sorted_items[:top_n]

    # 3. Analyze for missing keywords
    job_keywords = set(extract_keywords(job_description))
    resume_text = " ".join([item['text'] for item in top_results])
    resume_words = set(re.findall(r'\b\w+\b', resume_text.lower()))
    
    missing_keywords = list(job_keywords - resume_words)
    
    # Return the top results and the analysis
    return jsonify({
        "results": top_results,
        "analysis": {
            "missing_keywords": missing_keywords[:10] # Limit to top 10 missing
        }
    })

@app.route('/api/export-pdf', methods=['POST'])
def export_pdf():
    """
    Receives resume data, formats it as HTML, sends it to Stirling-PDF,
    and returns the resulting PDF file.
    """
    resume_data = request.json

    # --- 1. Format the data into an HTML string ---
    # This is a simple template. You can make this as complex and stylish as you want.
    skills_html = ''.join([f'<span style="background-color: #eee; padding: 2px 6px; border-radius: 4px; margin-right: 5px;">{skill}</span>' for skill in resume_data.get('skills', [])])

    experience_html = ''
    for exp in resume_data.get('experience', []):
        accomplishments_html = ''.join([f'<li style="margin-bottom: 5px;">{acc["accomplishment_text"]}</li>' for acc in resume_data.get('accomplishments', []) if acc.get("work_experience_id") == exp.get("id")])
        experience_html += f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between;">
                <h4 style="margin: 0; font-size: 1.1em;">{exp.get('job_title', '')}</h4>
                <p style="margin: 0;">{exp.get('dates', '')}</p>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <p style="margin: 0; font-style: italic;">{exp.get('company', '')}</p>
                <p style="margin: 0;">{exp.get('location', '')}</p>
            </div>
            <p style="margin-top: 5px;">{exp.get('description', '').replace('/n', '<br/>')}</p>
            <ul>{accomplishments_html}</ul>
        </div>
        """

    education_html = ''
    for edu in resume_data.get('education', []):
        education_html += f"<p>{edu.get('degree', '')} - {edu.get('institution', '')}</p>"

    projects_html = ''
    for proj in resume_data.get('projects', []):
        projects_html += f"""
        <div style="margin-bottom: 15px;">
            <h4 style="margin: 0; font-size: 1.1em;">{proj.get('project_name', '')}</h4>
            <p style="margin-top: 5px;">{proj.get('description', '').replace('/n', '<br/>')}</p>
            <p><b>Tools:</b> {proj.get('tools', '')}</p>
        </div>
        """


    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; font-size: 11pt; }}
            h1, h2, h3, h4, p {{ margin: 0; padding: 0; }}
            hr {{ border: none; border-top: 1px solid #ccc; margin: 15px 0; }}
        </style>
    </head>
    <body>
        <div style="text-align: center;">
            <h1 style="font-size: 2.5em;">{resume_data.get('name', 'Your Name')}</h1>
            <p>{resume_data.get('email', '')} | {resume_data.get('phone', '')} | {resume_data.get('linkedin', '')} | {resume_data.get('github', '')} | {resume_data.get('location', '')} | {resume_data.get('portfolio', '')}</p>
        </div>
        <hr>
        <div>
            <h3>Summary</h3>
            <p>{resume_data.get('summary', '')}</p>
        </div>
        <hr>
        <div>
            <h3>Skills</h3>
            <p>{skills_html}</p>
        </div>
        <hr>
        <div>
            <h3>Work Experience</h3>
            {experience_html}
        </div>
        <hr>
        <div>
            <h3>Technical Projects</h3>
            {projects_html}
        </div>
        <hr>
        <div>
            <h3>Education</h3>
            {education_html}
        </div>
    </body>
    </html>
    """

    # --- 2. Make API call to Stirling-PDF ---
    # The URL uses the service name 'stirling-pdf' which Docker resolves to the container's IP.
    stirling_url = 'http://stirling-pdf:8080/api/v1/convert/html/pdf'
    files = {'fileInput': ('resume.html', html_content, 'text/html')}

    try:
        response = requests.post(stirling_url, files=files)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # --- 3. Return the PDF to the user ---
        return send_file(
            io.BytesIO(response.content),
            mimetype='application/pdf',
            as_attachment=True,
            download_name='resume.pdf'
        )
    except requests.exceptions.RequestException as e:
        print(f"Error calling Stirling-PDF: {e}")
        return jsonify({"error": "Failed to generate PDF"}), 500

if __name__ == '__main__':
    app.run(debug=True)