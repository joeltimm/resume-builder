#resume-builder/backend/app.py
# It uses the Flask framework to create a simple API.

import os
import psycopg2
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Initialization ---
# Initialize the Flask application
app = Flask(__name__)

# Enable Cross-Origin Resource Sharing (CORS)
# This allows the frontend (on port 8080) to make requests to the backend (on port 5031)
CORS(app)

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

// Path: joeltimm/resume-builder/backend/app.py
def setup_database():
    """Ensures all required tables exist in the database."""
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        
        # --- Resume Table ---
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
                skill_text TEXT NOT NULL UNIQUE
            );
        ''')

        # --- New Accomplishments Table ---
        cur.execute('''
            CREATE TABLE IF NOT EXISTS accomplishments (
                id SERIAL PRIMARY KEY,
                accomplishment_text TEXT NOT NULL UNIQUE
            );
        ''')
        
        conn.commit()
        cur.close()
        conn.close()

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
        # Get the JSON data sent from the frontend
        resume_data = request.get_json()
        
        # Use an "upsert" operation: Update the row with id=1, or insert it if it doesn't exist.
        # This is safer than a simple UPDATE.
        # We are storing the entire JSON payload in the 'data' column.
        cur.execute(
            "INSERT INTO resume (id, data) VALUES (1, %s) ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data;",
            (jsonify(resume_data).get_data(as_text=True),)
        )
        conn.commit()
        
        cur.close()
        conn.close()
        return jsonify({"message": "Resume saved successfully"})

    if request.method == 'GET':
        # Fetch the resume data from the row with id=1.
        cur.execute("SELECT data FROM resume WHERE id = 1;")
        resume_data = cur.fetchone()
        
        cur.close()
        conn.close()

        if resume_data and resume_data[0]:
            # If data exists, return it.
            return jsonify(resume_data[0])
        else:
            # If no data is found, return a 404.
            return jsonify({"message": "No resume data found"}), 404

# --- API for Skills ---

@app.route('/api/skills', methods=['POST'])
def add_skill():
    """Adds a new skill to the database."""
    data = request.json
    skill_text = data.get('skill_text')

    if not skill_text:
        return jsonify({"error": "Skill text is required"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    cur = conn.cursor()
    try:
        cur.execute('INSERT INTO skills (skill_text) VALUES (%s) RETURNING id;', (skill_text,))
        new_id = cur.fetchone()[0]
        conn.commit()
        return jsonify({"message": "Skill added successfully", "id": new_id}), 201
    except psycopg2.IntegrityError:
        conn.rollback() # Roll back the transaction on error
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

# --- API for Skills ---

@app.route('/api/accomplishments', methods=['POST'])
def add_accomplishment():
    """Adds a new accomplishment to the database."""
    data = request.json
    # FIX: Changed variable to be specific to accomplishments
    accomplishment_text = data.get('accomplishment_text')

    if not accomplishment_text:
        return jsonify({"error": "Accomplishment text is required"}), 400

    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    cur = conn.cursor()
    try:
        # FIX: Correctly reference the accomplishment_text variable
        cur.execute('INSERT INTO accomplishments (accomplishment_text) VALUES (%s) RETURNING id;', (accomplishment_text,))
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
    # FIX: Corrected column name from 'accomplishments_text' to 'accomplishment_text'
    cur.execute('SELECT id, accomplishment_text FROM accomplishments ORDER BY accomplishment_text;')
    # FIX: Corrected key in the dictionary to match the column name
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
    # FIX: Corrected table name from 'accomplishmentss' to 'accomplishments'
    cur.execute('DELETE FROM accomplishments WHERE id = %s;', (accomplishment_id,))
    conn.commit()
    cur.close()
    conn.close()

    return jsonify({"message": "Accomplishment deleted successfully"})


# --- Main Execution ---
if __name__ == '__main__':
    # This check ensures that setup_database() is called once when the app starts.
    setup_database()
    # Run the Flask development server.
    # host='0.0.0.0' makes it accessible from outside the container.
    app.run(host='0.0.0.0', port=5001)
