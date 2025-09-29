"""
Tests for resume generation functions in resume_generator.py
"""
import pytest
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from resume_generator import generate_ats_resume_text


class TestGenerateATSResumeText:
    """Tests for generate_ats_resume_text function"""

    def test_generate_complete_resume(self, sample_resume_data):
        """Test generate_ats_resume_text with complete resume data"""
        result = generate_ats_resume_text(sample_resume_data)

        # Check that all sections are present
        assert "John Doe" in result
        assert "john.doe@email.com" in result
        assert "555-123-4567" in result
        assert "linkedin.com/in/johndoe" in result
        assert "SUMMARY" in result
        assert "SKILLS" in result
        assert "WORK EXPERIENCE" in result
        assert "EDUCATION" in result

        # Check content formatting
        assert "Python, JavaScript, SQL, React, AWS" in result
        assert "Senior Software Engineer" in result
        assert "Tech Corp" in result
        assert "Stanford University" in result

    def test_generate_resume_minimal_data(self):
        """Test generate_ats_resume_text with minimal resume data"""
        minimal_data = {
            "name": "Jane Smith",
            "email": "jane@example.com"
        }

        result = generate_ats_resume_text(minimal_data)

        assert "Jane Smith" in result
        assert "jane@example.com" in result
        assert "SUMMARY" not in result  # No summary provided
        assert "SKILLS" not in result   # No skills provided

    def test_generate_resume_empty_data(self):
        """Test generate_ats_resume_text with empty data"""
        result = generate_ats_resume_text({})

        # Should handle empty data gracefully
        assert isinstance(result, str)
        assert len(result) > 0
        # Should contain basic structure even with no data
        assert "=" in result  # Header separator

    def test_generate_resume_contact_info_formatting(self):
        """Test contact information formatting"""
        contact_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "555-1234",
            "linkedin": "linkedin.com/in/john"
        }

        result = generate_ats_resume_text(contact_data)

        # Check that contact info is formatted with separators
        assert "john@example.com | 555-1234 | linkedin.com/in/john" in result

    def test_generate_resume_partial_contact_info(self):
        """Test contact information with missing fields"""
        contact_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "linkedin": "linkedin.com/in/john"
            # Missing phone
        }

        result = generate_ats_resume_text(contact_data)

        # Should handle missing phone gracefully
        assert "john@example.com | linkedin.com/in/john" in result
        assert "| |" not in result  # No double separators

    def test_generate_resume_skills_section(self):
        """Test skills section formatting"""
        skills_data = {
            "name": "John Doe",
            "skills": ["Python", "JavaScript", "React", "AWS", "Docker"]
        }

        result = generate_ats_resume_text(skills_data)

        assert "SKILLS" in result
        assert "Python, JavaScript, React, AWS, Docker" in result

    def test_generate_resume_empty_skills(self):
        """Test with empty skills list"""
        skills_data = {
            "name": "John Doe",
            "skills": []
        }

        result = generate_ats_resume_text(skills_data)

        # Should not include skills section if empty
        assert "SKILLS" not in result

    def test_generate_resume_work_experience_section(self):
        """Test work experience section formatting"""
        experience_data = {
            "name": "John Doe",
            "experience": [
                {
                    "title": "Senior Software Engineer",
                    "company": "Tech Corp",
                    "dates": "2020-2023",
                    "bullets": [
                        "Developed scalable web applications",
                        "Led team of 5 engineers",
                        "Improved performance by 40%"
                    ]
                },
                {
                    "title": "Software Engineer",
                    "company": "StartupCo",
                    "dates": "2018-2020",
                    "bullets": [
                        "Built REST APIs using Python"
                    ]
                }
            ]
        }

        result = generate_ats_resume_text(experience_data)

        assert "WORK EXPERIENCE" in result
        assert "Senior Software Engineer" in result
        assert "Tech Corp | 2020-2023" in result
        assert "- Developed scalable web applications" in result
        assert "- Led team of 5 engineers" in result
        assert "Software Engineer" in result
        assert "StartupCo | 2018-2020" in result

    def test_generate_resume_experience_missing_fields(self):
        """Test work experience with missing fields"""
        experience_data = {
            "name": "John Doe",
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    # Missing dates
                    "bullets": ["Developed applications"]
                }
            ]
        }

        result = generate_ats_resume_text(experience_data)

        assert "Software Engineer" in result
        assert "Tech Corp" in result
        assert "- Developed applications" in result

    def test_generate_resume_experience_empty_bullets(self):
        """Test work experience with empty bullets list"""
        experience_data = {
            "name": "John Doe",
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "Tech Corp",
                    "dates": "2020-2023",
                    "bullets": []
                }
            ]
        }

        result = generate_ats_resume_text(experience_data)

        assert "Software Engineer" in result
        assert "Tech Corp" in result
        # Should handle empty bullets gracefully

    def test_generate_resume_education_section(self):
        """Test education section formatting"""
        education_data = {
            "name": "John Doe",
            "education": [
                {
                    "degree": "BS Computer Science",
                    "school": "Stanford University",
                    "dates": "2016-2020"
                },
                {
                    "degree": "MS Data Science",
                    "school": "MIT",
                    "dates": "2020-2022"
                }
            ]
        }

        result = generate_ats_resume_text(education_data)

        assert "EDUCATION" in result
        assert "BS Computer Science" in result
        assert "Stanford University | 2016-2020" in result
        assert "MS Data Science" in result
        assert "MIT | 2020-2022" in result

    def test_generate_resume_education_missing_fields(self):
        """Test education with missing fields"""
        education_data = {
            "name": "John Doe",
            "education": [
                {
                    "degree": "BS Computer Science",
                    "school": "Stanford University"
                    # Missing dates
                }
            ]
        }

        result = generate_ats_resume_text(education_data)

        assert "BS Computer Science" in result
        assert "Stanford University" in result

    def test_generate_resume_summary_section(self):
        """Test summary section formatting"""
        summary_data = {
            "name": "John Doe",
            "summary": "Experienced software engineer with 5+ years developing scalable applications"
        }

        result = generate_ats_resume_text(summary_data)

        assert "SUMMARY" in result
        assert "Experienced software engineer with 5+ years developing scalable applications" in result

    def test_generate_resume_empty_summary(self):
        """Test with empty summary"""
        summary_data = {
            "name": "John Doe",
            "summary": ""
        }

        result = generate_ats_resume_text(summary_data)

        # Should not include summary section if empty
        assert "SUMMARY" not in result

    def test_generate_resume_section_order(self):
        """Test that sections appear in correct order"""
        complete_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "summary": "Software engineer",
            "skills": ["Python"],
            "experience": [{"title": "Engineer", "company": "Corp", "bullets": ["Worked"]}],
            "education": [{"degree": "BS", "school": "University"}]
        }

        result = generate_ats_resume_text(complete_data)

        # Find positions of each section
        summary_pos = result.find("SUMMARY")
        skills_pos = result.find("SKILLS")
        experience_pos = result.find("WORK EXPERIENCE")
        education_pos = result.find("EDUCATION")

        # Verify order: Summary -> Skills -> Experience -> Education
        assert summary_pos < skills_pos
        assert skills_pos < experience_pos
        assert experience_pos < education_pos

    def test_generate_resume_header_separator(self):
        """Test that header separator is properly formatted"""
        data = {"name": "John Doe", "email": "john@example.com"}
        result = generate_ats_resume_text(data)

        # Should contain separator line
        assert "=" * 80 in result

    def test_generate_resume_line_breaks(self):
        """Test proper line break formatting"""
        data = {
            "name": "John Doe",
            "summary": "Software engineer",
            "skills": ["Python", "JavaScript"]
        }

        result = generate_ats_resume_text(data)
        lines = result.split('\n')

        # Should have proper spacing between sections
        assert len(lines) > 5
        assert any(line.strip() == "" for line in lines)  # Should have empty lines

    def test_generate_resume_special_characters(self):
        """Test handling of special characters in resume data"""
        special_data = {
            "name": "José García-Smith",
            "email": "jose.garcia@example.com",
            "summary": "Engineer with 5+ years experience & expertise in C++/Python",
            "skills": ["C++", "Python", ".NET", "SQL Server"],
            "experience": [{
                "title": "Sr. Software Engineer",
                "company": "Tech & Innovation Corp",
                "bullets": ["Increased performance by 25%", "Reduced costs by $50K"]
            }]
        }

        result = generate_ats_resume_text(special_data)

        # Should handle special characters properly
        assert "José García-Smith" in result
        assert "5+ years experience & expertise" in result
        assert "C++, Python, .NET, SQL Server" in result
        assert "25%" in result
        assert "$50K" in result

    def test_generate_resume_return_type(self):
        """Test that function returns a string"""
        result = generate_ats_resume_text({"name": "Test"})
        assert isinstance(result, str)

    def test_generate_resume_not_empty_output(self):
        """Test that function never returns empty string"""
        # Test with various inputs
        test_cases = [
            {},
            {"name": ""},
            {"name": "John"},
            {"random_field": "value"}
        ]

        for test_data in test_cases:
            result = generate_ats_resume_text(test_data)
            assert len(result) > 0
            assert isinstance(result, str)

    @pytest.mark.parametrize("field_name,field_value", [
        ("name", "John Doe"),
        ("email", "john@example.com"),
        ("phone", "555-1234"),
        ("linkedin", "linkedin.com/in/john"),
        ("location", "New York, NY"),
        ("summary", "Experienced engineer")
    ])
    def test_generate_resume_individual_fields(self, field_name, field_value):
        """Test individual fields are properly included"""
        data = {field_name: field_value}
        result = generate_ats_resume_text(data)

        if field_name in ["name", "email", "phone", "linkedin", "location"]:
            # These should appear in contact section
            assert field_value in result
        elif field_name == "summary":
            assert "SUMMARY" in result
            assert field_value in result