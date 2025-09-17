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
            
            # --- Resume Table ---
            # FIX: Changed column name from 'data' to 'content' to match the GET request
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

            # --- New Skills Table ---
            cur.execute('''
                CREATE TABLE IF NOT EXISTS skills (
                    id SERIAL PRIMARY KEY,
                    skill_text TEXT NOT NULL UNIQUE,
                    embedding TEXT 
                );
            ''')

            # --- New Accomplishments Table ---
            cur.execute('''
                CREATE TABLE IF NOT EXISTS accomplishments (
                    id SERIAL PRIMARY KEY,
                    accomplishment_text TEXT NOT NULL UNIQUE,
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
    This is the main API endpoint.
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
            'INSERT INTO accomplishments (accomplishment_text, embedding) VALUES (%s, %s) RETURNING id;', 
            (accomplishment_text, json.dumps(embedding))
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
    cur.execute('SELECT id, accomplishment_text FROM accomplishments ORDER BY accomplishment_text;')
    accomplishments = [{"id": row[0], "accomplishment_text": row[1]} for row in cur.fetchall()]
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

# --- NEW: API for AI Matching ---

@app.route('/api/match', methods=['POST'])
def match_skills():
    """
    Finds the most relevant skills and accomplishments for a given job description.
    """
    data = request.json
    job_description = data.get('job_description')
    top_n = data.get('limit', 10) # Allow frontend to specify how many results to return

    if not job_description:
        return jsonify({"error": "Job description is required"}), 400

    # 1. Generate embedding for the job description
    job_embedding = model.encode(job_description)

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    cur = conn.cursor()

    # 2. Fetch all skills and their embeddings
    cur.execute('SELECT id, skill_text, embedding FROM skills;')
    skills = cur.fetchall()
    
    # 3. Fetch all accomplishments and their embeddings
    cur.execute('SELECT id, accomplishment_text, embedding FROM accomplishments;')
    accomplishments = cur.fetchall()
    
    cur.close()
    conn.close()

    all_items = []
    if skills:
        skill_embeddings = np.array([json.loads(s[2]) for s in skills])
        # Compute cosine similarity between the job description and all skills
        skill_scores = util.cos_sim(job_embedding, skill_embeddings)[0]
        for i, skill in enumerate(skills):
            all_items.append({"id": f"skill-{skill[0]}", "text": skill[1], "score": float(skill_scores[i]), "type": "skill"})
            
    if accomplishments:
        accomplishment_embeddings = np.array([json.loads(a[2]) for a in accomplishments])
        # Compute cosine similarity for accomplishments
        accomplishment_scores = util.cos_sim(job_embedding, accomplishment_embeddings)[0]
        for i, acc in enumerate(accomplishments):
            all_items.append({"id": f"acc-{acc[0]}", "text": acc[1], "score": float(accomplishment_scores[i]), "type": "accomplishment"})

    # 4. Sort all items by score in descending order
    sorted_items = sorted(all_items, key=lambda x: x['score'], reverse=True)
    
    # 5. Return the top N results
    return jsonify(sorted_items[:top_n])

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
    
    accomplishments_html = ''.join([f'<li style="margin-bottom: 5px;">{acc}</li>' for acc in resume_data.get('accomplishments', [])])

    experience_html = ''
    for exp in resume_data.get('experience', []):
        experience_html += f"""
        <div style="margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between;">
                <h4 style="margin: 0; font-size: 1.1em;">{exp.get('jobTitle', '')}</h4>
                <p style="margin: 0;">{exp.get('dates', '')}</p>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <p style="margin: 0; font-style: italic;">{exp.get('company', '')}</p>
                <p style="margin: 0;">{exp.get('location', '')}</p>
            </div>
            <p style="margin-top: 5px;">{exp.get('description', '').replace('/n', '<br/>')}</p>
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
            <p>{resume_data.get('email', '')} | {resume_data.get('phone', '')} | {resume_data.get('linkedin', '')} | {resume_data.get('github', '')}</p>
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
            <h3>Key Accomplishments</h3>
            <ul>{accomplishments_html}</ul>
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
        