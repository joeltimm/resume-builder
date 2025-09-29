"""
Tests for Flask API endpoints in app.py
"""
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestResumeEndpoint:
    """Tests for /resume endpoint"""

    @patch('app.get_db_connection')
    def test_get_resume_success(self, mock_get_db, client):
        """Test GET /resume returns resume data successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = ('{"name": "John Doe", "email": "john@example.com"}',)

        response = client.get('/resume')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['name'] == 'John Doe'
        assert data['email'] == 'john@example.com'

    @patch('app.get_db_connection')
    def test_get_resume_no_data(self, mock_get_db, client):
        """Test GET /resume when no resume data exists"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchone.return_value = ('{}',)

        response = client.get('/resume')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data == {}

    @patch('app.get_db_connection')
    def test_post_resume_success(self, mock_get_db, client, sample_resume_data):
        """Test POST /resume saves resume data successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        response = client.post('/resume',
                             data=json.dumps(sample_resume_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Resume saved successfully.'

    @patch('app.get_db_connection')
    def test_resume_db_connection_fail(self, mock_get_db, client):
        """Test resume endpoint when database connection fails"""
        mock_get_db.return_value = None

        response = client.get('/resume')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'Database connection failed' in data['error']


class TestSkillsEndpoint:
    """Tests for /api/skills endpoints"""

    @patch('app.get_db_connection')
    @patch('app.model')
    def test_add_skill_success(self, mock_model, mock_get_db, client):
        """Test POST /api/skills adds skill successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        skill_data = {"skill_text": "Python"}

        response = client.post('/api/skills',
                             data=json.dumps(skill_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Skill added successfully.'

    @patch('app.get_db_connection')
    def test_get_skills_success(self, mock_get_db, client, sample_skills_data):
        """Test GET /api/skills returns skills list"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Convert sample data to tuple format that fetchall returns
        mock_cursor.fetchall.return_value = [
            (item['id'], item['skill_text'], item['embedding'])
            for item in sample_skills_data
        ]

        response = client.get('/api/skills')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 3
        assert data[0]['skill_text'] == 'Python'

    @patch('app.get_db_connection')
    def test_delete_skill_success(self, mock_get_db, client):
        """Test DELETE /api/skills/<id> deletes skill successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        response = client.delete('/api/skills/1')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Skill deleted successfully.'


class TestAccomplishmentsEndpoint:
    """Tests for /api/accomplishments endpoints"""

    @patch('app.get_db_connection')
    @patch('app.model')
    def test_add_accomplishment_success(self, mock_model, mock_get_db, client):
        """Test POST /api/accomplishments adds accomplishment successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        accomplishment_data = {
            "accomplishment_text": "Increased system performance by 40%",
            "work_experience_id": 1
        }

        response = client.post('/api/accomplishments',
                             data=json.dumps(accomplishment_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Accomplishment added successfully.'

    @patch('app.get_db_connection')
    def test_get_accomplishments_success(self, mock_get_db, client):
        """Test GET /api/accomplishments returns accomplishments list"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            (1, "Increased performance by 40%", 1, "[0.1,0.2,0.3]"),
            (2, "Led team of 5 engineers", 1, "[0.4,0.5,0.6]")
        ]

        response = client.get('/api/accomplishments')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 2
        assert "performance" in data[0]['accomplishment_text']


class TestMatchEndpoint:
    """Tests for /api/match endpoint"""

    @patch('app.get_db_connection')
    @patch('app.model')
    @patch('app.util.semantic_search')
    def test_match_skills_success(self, mock_search, mock_model, mock_get_db,
                                client, sample_job_description):
        """Test POST /api/match returns relevant skills and accomplishments"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock database responses
        mock_cursor.fetchall.side_effect = [
            [(1, "Python", "[0.1,0.2,0.3]")],  # skills
            [(1, "Built scalable apps", 1, "[0.4,0.5,0.6]")]  # accomplishments
        ]

        mock_model.encode.return_value = [0.7, 0.8, 0.9]
        mock_search.return_value = [{'corpus_id': 0, 'score': 0.95}]

        match_data = {"job_description": sample_job_description, "limit": 10}

        response = client.post('/api/match',
                             data=json.dumps(match_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'relevant_skills' in data
        assert 'relevant_accomplishments' in data

    def test_match_missing_job_description(self, client):
        """Test /api/match with missing job description"""
        match_data = {"limit": 10}

        response = client.post('/api/match',
                             data=json.dumps(match_data),
                             content_type='application/json')

        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'Job description is required' in data['error']


class TestScoringEndpoint:
    """Tests for /calculate-score endpoint"""

    @patch('app.calculate_weighted_match_score')
    def test_calculate_score_success(self, mock_score_func, client):
        """Test POST /calculate-score returns score successfully"""
        mock_score_func.return_value = json.dumps({
            "overallScore": 85,
            "breakdown": {
                "hard_skills": {"score": 90, "weight": 50, "contribution": 45}
            }
        })

        score_data = {
            "resumeText": "Python developer with 5 years experience",
            "jobDescription": "Looking for Python developer"
        }

        response = client.post('/calculate-score',
                             data=json.dumps(score_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['overallScore'] == 85

    def test_calculate_score_missing_data(self, client):
        """Test /calculate-score with missing required data"""
        score_data = {"resumeText": "Python developer"}

        response = client.post('/calculate-score',
                             data=json.dumps(score_data),
                             content_type='application/json')

        assert response.status_code == 400


class TestImproveBulletEndpoint:
    """Tests for /improve-bullet endpoint"""

    @patch('app.improve_resume_bullet')
    def test_improve_bullet_success(self, mock_improve_func, client):
        """Test POST /improve-bullet returns improved bullet"""
        mock_improve_func.return_value = "Engineered scalable Python applications, increasing performance by 40%"

        bullet_data = {
            "bulletPoint": "Worked on Python apps",
            "jobTitle": "Software Engineer",
            "industry": "Technology"
        }

        response = client.post('/improve-bullet',
                             data=json.dumps(bullet_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'improved_bullet' in data
        assert "Engineered" in data['improved_bullet']

    def test_improve_bullet_missing_data(self, client):
        """Test /improve-bullet with missing required data"""
        bullet_data = {"bulletPoint": "Worked on apps"}

        response = client.post('/improve-bullet',
                             data=json.dumps(bullet_data),
                             content_type='application/json')

        assert response.status_code == 400


class TestCheckDuplicatesEndpoint:
    """Tests for /check-duplicates endpoint"""

    @patch('app.find_duplicate_entries')
    def test_check_duplicates_success(self, mock_duplicates_func, client):
        """Test POST /check-duplicates returns duplicate analysis"""
        mock_duplicates_func.return_value = [
            {
                "indices": [0, 1],
                "entries": ["Built web apps", "Developed web applications"],
                "score": 85.5
            }
        ]

        duplicates_data = {
            "bulletPoints": [
                "Built web apps using Python",
                "Developed web applications with Python",
                "Managed team of engineers"
            ]
        }

        response = client.post('/check-duplicates',
                             data=json.dumps(duplicates_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'duplicates' in data
        assert len(data['duplicates']) == 1


class TestATSResumeEndpoint:
    """Tests for /generate-ats-resume endpoint"""

    @patch('app.generate_ats_resume_text')
    def test_generate_ats_resume_success(self, mock_generate_func, client, sample_resume_data):
        """Test POST /generate-ats-resume returns formatted resume text"""
        mock_generate_func.return_value = "JOHN DOE\njohn.doe@email.com | 555-123-4567"

        response = client.post('/generate-ats-resume',
                             data=json.dumps(sample_resume_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'ats_resume_text' in data
        assert "JOHN DOE" in data['ats_resume_text']


class TestPDFExportEndpoint:
    """Tests for /api/export-pdf endpoint"""

    @patch('app.requests.post')
    @patch('app.generate_ats_resume_text')
    def test_export_pdf_success(self, mock_generate, mock_requests, client, sample_resume_data):
        """Test POST /api/export-pdf returns PDF file"""
        mock_generate.return_value = "JOHN DOE\nSoftware Engineer"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"Mock PDF content"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_requests.return_value = mock_response

        response = client.post('/api/export-pdf',
                             data=json.dumps(sample_resume_data),
                             content_type='application/json')

        assert response.status_code == 200
        assert response.content_type == 'application/pdf'
        assert response.data == b"Mock PDF content"

    @patch('app.requests.post')
    def test_export_pdf_stirling_error(self, mock_requests, client, sample_resume_data):
        """Test /api/export-pdf when Stirling PDF service fails"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_requests.return_value = mock_response

        response = client.post('/api/export-pdf',
                             data=json.dumps(sample_resume_data),
                             content_type='application/json')

        assert response.status_code == 500
        data = json.loads(response.data)
        assert 'error' in data


class TestWorkExperienceEndpoint:
    """Tests for /api/work_experience endpoints"""

    @patch('app.get_db_connection')
    @patch('app.model')
    def test_add_work_experience_success(self, mock_model, mock_get_db, client):
        """Test POST /api/work_experience adds work experience successfully"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_model.encode.return_value = [0.1, 0.2, 0.3]

        work_data = {
            "job_title": "Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "dates": "2020-2023",
            "description": "Developed scalable applications"
        }

        response = client.post('/api/work_experience',
                             data=json.dumps(work_data),
                             content_type='application/json')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['message'] == 'Work experience added successfully.'

    @patch('app.get_db_connection')
    def test_get_work_experience_success(self, mock_get_db, client):
        """Test GET /api/work_experience returns work experience list"""
        mock_conn, mock_cursor = Mock(), Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        mock_cursor.fetchall.return_value = [
            (1, "Software Engineer", "Tech Corp", "SF", "2020-2023", "Developed apps", "[0.1,0.2]")
        ]

        response = client.get('/api/work_experience')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert len(data) == 1
        assert data[0]['job_title'] == 'Software Engineer'