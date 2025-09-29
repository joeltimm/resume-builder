"""
Pytest configuration and fixtures for resume-builder tests.
"""
import pytest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from flask import Flask
import sys
import tempfile

# Add the backend directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import after path modification
from app import app as flask_app


@pytest.fixture
def app():
    """Create and configure a test Flask app."""
    flask_app.config['TESTING'] = True
    flask_app.config['WTF_CSRF_ENABLED'] = False

    # Mock environment variables for testing
    with patch.dict(os.environ, {
        'POSTGRES_DB': 'test_resume_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_password',
        'ANTHROPIC_API_KEY': 'test_api_key_123'
    }):
        yield flask_app


@pytest.fixture
def client(app):
    """Create a test client for the Flask app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """Create a test runner for the Flask app."""
    return app.test_cli_runner()


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    return mock_conn, mock_cursor


@pytest.fixture
def sample_resume_data():
    """Sample resume data for testing."""
    return {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "555-123-4567",
        "linkedin": "linkedin.com/in/johndoe",
        "location": "New York, NY",
        "summary": "Experienced software engineer with 5+ years of experience",
        "skills": ["Python", "JavaScript", "SQL", "React", "AWS"],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Tech Corp",
                "location": "San Francisco, CA",
                "dates": "2020-2023",
                "bullets": [
                    "Developed scalable web applications using Python and React",
                    "Led a team of 5 engineers to deliver critical features",
                    "Improved system performance by 40% through optimization"
                ]
            }
        ],
        "education": [
            {
                "degree": "BS Computer Science",
                "school": "Stanford University",
                "location": "Stanford, CA",
                "dates": "2016-2020"
            }
        ]
    }


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return """
    We are seeking a Senior Software Engineer to join our team. The ideal candidate will have:
    - 5+ years of experience in Python development
    - Strong knowledge of JavaScript and React
    - Experience with AWS cloud services
    - Leadership experience managing engineering teams
    - Bachelor's degree in Computer Science or related field
    - Excellent problem-solving and communication skills
    """


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for LLM testing."""
    with patch('anthropic.Anthropic') as mock_anthropic:
        mock_client = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Improved bullet point with better metrics and action verbs"
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_sentence_transformer():
    """Mock SentenceTransformer for embedding tests."""
    with patch('sentence_transformers.SentenceTransformer') as mock_st:
        mock_model = Mock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st.return_value = mock_model
        yield mock_model


@pytest.fixture
def sample_skills_data():
    """Sample skills data for database testing."""
    return [
        {"id": 1, "skill_text": "Python", "embedding": "[0.1, 0.2, 0.3]"},
        {"id": 2, "skill_text": "JavaScript", "embedding": "[0.4, 0.5, 0.6]"},
        {"id": 3, "skill_text": "React", "embedding": "[0.7, 0.8, 0.9]"}
    ]


@pytest.fixture
def sample_accomplishments_data():
    """Sample accomplishments data for database testing."""
    return [
        {
            "id": 1,
            "accomplishment_text": "Increased system performance by 40%",
            "work_experience_id": 1,
            "embedding": "[0.1, 0.2, 0.3]"
        },
        {
            "id": 2,
            "accomplishment_text": "Led team of 5 engineers to deliver critical features",
            "work_experience_id": 1,
            "embedding": "[0.4, 0.5, 0.6]"
        }
    ]


@pytest.fixture
def mock_requests_post():
    """Mock requests.post for PDF generation testing."""
    with patch('requests.post') as mock_post:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"Mock PDF content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture
def mock_yake_extractor():
    """Mock YAKE keyword extractor."""
    with patch('yake.KeywordExtractor') as mock_yake:
        mock_extractor = Mock()
        mock_extractor.extract_keywords.return_value = [
            (0.1, "python"), (0.2, "javascript"), (0.3, "react")
        ]
        mock_yake.return_value = mock_extractor
        yield mock_extractor


@pytest.fixture
def sample_duplicate_entries():
    """Sample entries for duplicate detection testing."""
    return [
        "Developed web applications using Python and Django",
        "Built scalable web apps with Python and Django framework",
        "Created mobile applications using React Native",
        "Managed team of 5 software engineers",
        "Led a development team of 5 engineers"
    ]


@pytest.fixture
def temp_db_file():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        yield f.name
    os.unlink(f.name)