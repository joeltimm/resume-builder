#resume-builder/backend/app.py
# This is our main backend application file.
# It uses the Flask framework to create a simple API.

import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import psycopg2
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
# Enable Cross-Origin Resource Sharing (CORS) to allow our frontend
# to make requests to this API, even if they are on different domains/ports.
CORS(app)

def get_db_connection():
    #Establishes a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            host="postgres", # The hostname 'postgres' matches the service name in docker-compose.yml
            database=os.getenv('POSTGRES_DB'),
            user=os.getenv('POSTGRES_USER'),
            password=os.getenv('POSTGRES_PASSWORD')
        )
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def setup_database():
    #Ensures the resume table exists in the database.
    conn = get_db_connection()
    if conn:
        cur = conn.cursor()
        # Use TEXT type for simplicity. JSONB would be a more advanced option.
        cur.execute('''
            CREATE TABLE IF NOT EXISTS resume (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL
            );
        ''')
        # Check if a default resume entry exists, if not, create one.
        cur.execute('SELECT id FROM resume WHERE id = 1;')
        if cur.fetchone() is None:
            cur.execute('INSERT INTO resume (id, content) VALUES (1, %s);', ('{}',))
        conn.commit()
        cur.close()
        conn.close()

# Define the API endpoint for getting the resume data.
# It listens for GET requests at the URL /api/resume
@app.route('/api/resume', methods=['GET'])
def get_resume():
    #Fetches the resume data from the database.
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
        
    cur = conn.cursor()
    # We will only ever have one resume, so we fetch the one with id=1.
    cur.execute('SELECT content FROM resume WHERE id = 1;')
    resume_data = cur.fetchone()
    cur.close()
    conn.close()
    
    if resume_data:
        # The data is stored as a string, so we return it.
        # The frontend will parse it as JSON.
        return jsonify(resume_data[0])
    else:
        # If for some reason there's no resume, return an empty object.
        return jsonify({})

# Define the API endpoint for saving the resume data.
# It listens for POST requests at the URL /api/resume
@app.route('/api/resume', methods=['POST'])
def save_resume():
    #"""Saves or updates the resume data in the database."""
    # Get the JSON data sent from the frontend in the request body.
    new_data = request.json
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
        
    cur = conn.cursor()
    # Update the content of the resume with id=1.
    # The %s is a placeholder to prevent SQL injection.
    cur.execute('UPDATE resume SET content = %s WHERE id = 1;', (str(new_data),))
    conn.commit()
    cur.close()
    conn.close()
    
    return jsonify({"message": "Resume saved successfully"})

# This block ensures that the database table is created when the app starts.
if __name__ == '__main__':
    setup_database()
    # This starts the Flask development server.
    app.run(host='0.0.0.0', port=5001)
