"""
Tests for scoring logic functions in scoring_logic.py
"""
import pytest
import json
from unittest.mock import Mock, patch
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from scoring_logic import (
    calculate_weighted_match_score,
    _preprocess_text,
    _extract_keywords_by_category,
    _calculate_category_score,
    _calculate_quantifiable_score,
    HARD_SKILLS,
    TOOLS,
    SOFT_SKILLS,
    ACTION_VERBS,
    CATEGORY_WEIGHTS
)


class TestPreprocessText:
    """Tests for _preprocess_text function"""

    def test_preprocess_text_lowercase(self):
        """Test that text is converted to lowercase"""
        result = _preprocess_text("HELLO WORLD Python")
        assert result == "hello world python"

    def test_preprocess_text_remove_punctuation(self):
        """Test that punctuation is removed"""
        result = _preprocess_text("Hello, World! How are you?")
        assert result == "hello world how"

    def test_preprocess_text_remove_numbers(self):
        """Test that numbers are removed"""
        result = _preprocess_text("Python 3.9 version 2023")
        assert result == "python version"

    @patch('scoring_logic.stopwords.words')
    def test_preprocess_text_remove_stopwords(self, mock_stopwords):
        """Test that stopwords are removed"""
        mock_stopwords.return_value = ['is', 'the', 'and', 'a']
        result = _preprocess_text("Python is the best programming language and a great tool")
        assert result == "python best programming language great tool"

    def test_preprocess_text_empty_string(self):
        """Test preprocessing empty string"""
        result = _preprocess_text("")
        assert result == ""

    def test_preprocess_text_whitespace_handling(self):
        """Test that multiple whitespaces are handled correctly"""
        result = _preprocess_text("  hello    world   ")
        expected_words = result.split()
        assert "hello" in expected_words
        assert "world" in expected_words


class TestExtractKeywordsByCategory:
    """Tests for _extract_keywords_by_category function"""

    def test_extract_keywords_found(self):
        """Test extracting keywords that exist in text"""
        text = "I have experience with Python, JavaScript, and React development"
        keywords = ["python", "javascript", "react", "java"]

        result = _extract_keywords_by_category(text, keywords)

        assert "python" in result
        assert "javascript" in result
        assert "react" in result
        assert "java" not in result

    def test_extract_keywords_none_found(self):
        """Test when no keywords are found"""
        text = "I like cooking and gardening"
        keywords = ["python", "javascript", "react"]

        result = _extract_keywords_by_category(text, keywords)

        assert result == ""

    def test_extract_keywords_case_insensitive(self):
        """Test that keyword extraction is case insensitive"""
        text = "I work with PYTHON and JavaScript"
        keywords = ["python", "javascript"]

        result = _extract_keywords_by_category(text, keywords)

        assert "python" in result
        assert "javascript" in result

    def test_extract_keywords_empty_text(self):
        """Test with empty text"""
        result = _extract_keywords_by_category("", ["python"])
        assert result == ""

    def test_extract_keywords_empty_list(self):
        """Test with empty keyword list"""
        result = _extract_keywords_by_category("Python development", [])
        assert result == ""


class TestCalculateCategoryScore:
    """Tests for _calculate_category_score function"""

    @patch('scoring_logic.cosine_similarity')
    @patch('scoring_logic.TfidfVectorizer')
    def test_calculate_category_score_with_matches(self, mock_vectorizer, mock_cosine):
        """Test category score calculation when keywords match"""
        # Mock TF-IDF vectorizer
        mock_tfidf = Mock()
        mock_vectorizer.return_value = mock_tfidf
        mock_tfidf.fit_transform.return_value = [[0.1, 0.2], [0.3, 0.4]]

        # Mock cosine similarity
        mock_cosine.return_value = [[0.85]]

        resume_text = "Python developer with JavaScript experience"
        jd_text = "Looking for Python and JavaScript developer"
        keywords = ["python", "javascript"]

        result = _calculate_category_score(resume_text, jd_text, keywords)

        assert result == 0.85

    def test_calculate_category_score_no_resume_keywords(self):
        """Test when no keywords found in resume"""
        resume_text = "Marketing specialist"
        jd_text = "Python developer needed"
        keywords = ["python", "javascript"]

        result = _calculate_category_score(resume_text, jd_text, keywords)

        assert result == 0.0

    def test_calculate_category_score_no_jd_keywords(self):
        """Test when no keywords found in job description"""
        resume_text = "Python developer"
        jd_text = "Marketing manager position"
        keywords = ["python", "javascript"]

        result = _calculate_category_score(resume_text, jd_text, keywords)

        assert result == 0.0

    @patch('scoring_logic.TfidfVectorizer')
    def test_calculate_category_score_vectorizer_error(self, mock_vectorizer):
        """Test handling of vectorizer errors"""
        mock_tfidf = Mock()
        mock_vectorizer.return_value = mock_tfidf
        mock_tfidf.fit_transform.side_effect = ValueError("Vectorizer error")

        result = _calculate_category_score("python", "python", ["python"])

        assert result == 0.0


class TestCalculateQuantifiableScore:
    """Tests for _calculate_quantifiable_score function"""

    def test_quantifiable_score_with_percentages(self):
        """Test scoring with percentage metrics"""
        text = "Increased performance by 40% and improved efficiency by 25%"
        result = _calculate_quantifiable_score(text)
        assert result == 1.0

    def test_quantifiable_score_with_money(self):
        """Test scoring with monetary metrics"""
        text = "Saved company $50000 annually and generated $100K in revenue"
        result = _calculate_quantifiable_score(text)
        assert result == 1.0

    def test_quantifiable_score_with_numbers(self):
        """Test scoring with numeric metrics"""
        text = "Managed team of 15 people and completed 50 projects"
        result = _calculate_quantifiable_score(text)
        assert result == 1.0

    def test_quantifiable_score_no_metrics(self):
        """Test scoring without quantifiable metrics"""
        text = "Developed applications and managed projects effectively"
        result = _calculate_quantifiable_score(text)
        assert result == 0.0

    def test_quantifiable_score_mixed_content(self):
        """Test with mixed content (some metrics, some not)"""
        text = "Developed web applications and increased performance by 30%"
        result = _calculate_quantifiable_score(text)
        assert result == 1.0

    def test_quantifiable_score_decimal_percentages(self):
        """Test with decimal percentages"""
        text = "Improved accuracy by 12.5% over baseline"
        result = _calculate_quantifiable_score(text)
        assert result == 1.0


class TestCalculateWeightedMatchScore:
    """Tests for calculate_weighted_match_score function"""

    @patch('scoring_logic._calculate_category_score')
    @patch('scoring_logic._calculate_quantifiable_score')
    def test_calculate_weighted_match_score_perfect_match(self, mock_quant_score, mock_cat_score):
        """Test weighted score calculation with perfect matches"""
        # Mock all category scores as perfect (1.0)
        mock_cat_score.return_value = 1.0
        mock_quant_score.return_value = 1.0

        resume_text = "Python JavaScript developer with AWS experience"
        job_description = "Python JavaScript developer with AWS required"

        result_json = calculate_weighted_match_score(resume_text, job_description)
        result = json.loads(result_json)

        assert result['overallScore'] == 100
        assert 'breakdown' in result
        assert len(result['breakdown']) == 5  # All categories

    @patch('scoring_logic._calculate_category_score')
    @patch('scoring_logic._calculate_quantifiable_score')
    def test_calculate_weighted_match_score_no_match(self, mock_quant_score, mock_cat_score):
        """Test weighted score calculation with no matches"""
        # Mock all scores as zero
        mock_cat_score.return_value = 0.0
        mock_quant_score.return_value = 0.0

        result_json = calculate_weighted_match_score("Marketing", "Engineering")
        result = json.loads(result_json)

        assert result['overallScore'] == 0
        assert result['breakdown']['hard_skills']['score'] == 0

    @patch('scoring_logic._calculate_category_score')
    @patch('scoring_logic._calculate_quantifiable_score')
    def test_calculate_weighted_match_score_partial_match(self, mock_quant_score, mock_cat_score):
        """Test weighted score calculation with partial matches"""
        # Mock different scores for different categories
        def mock_category_side_effect(resume, jd, keywords):
            if keywords == HARD_SKILLS:
                return 0.8
            elif keywords == TOOLS:
                return 0.6
            elif keywords == SOFT_SKILLS:
                return 0.4
            elif keywords == ACTION_VERBS:
                return 0.7
            return 0.0

        mock_cat_score.side_effect = mock_category_side_effect
        mock_quant_score.return_value = 0.5

        result_json = calculate_weighted_match_score("Resume text", "Job description")
        result = json.loads(result_json)

        # Should calculate weighted average based on CATEGORY_WEIGHTS
        expected_score = (0.8 * 0.50) + (0.6 * 0.20) + (0.4 * 0.15) + (0.7 * 0.10) + (0.5 * 0.05)
        expected_percentage = int(round(expected_score * 100))

        assert result['overallScore'] == expected_percentage

    def test_calculate_weighted_match_score_return_format(self):
        """Test that return format is valid JSON with required fields"""
        result_json = calculate_weighted_match_score("Python developer", "Python required")

        # Should be valid JSON
        result = json.loads(result_json)

        # Check required fields
        assert 'overallScore' in result
        assert 'breakdown' in result
        assert isinstance(result['overallScore'], int)
        assert isinstance(result['breakdown'], dict)

        # Check breakdown structure
        expected_categories = ['hard_skills', 'tools', 'soft_skills', 'action_verbs', 'quantifiable_metrics']
        for category in expected_categories:
            assert category in result['breakdown']
            assert 'score' in result['breakdown'][category]
            assert 'weight' in result['breakdown'][category]
            assert 'contribution' in result['breakdown'][category]

    def test_calculate_weighted_match_score_weights_sum_to_100(self):
        """Test that category weights sum to 100%"""
        total_weight = sum(CATEGORY_WEIGHTS.values())
        assert abs(total_weight - 1.0) < 0.01  # Should sum to 1.0 (100%)

    def test_calculate_weighted_match_score_score_range(self):
        """Test that scores are within valid range (0-100)"""
        test_cases = [
            ("", ""),
            ("Python", "JavaScript"),
            ("Python developer", "Python developer"),
            ("Python React AWS leadership", "Python React AWS leadership required")
        ]

        for resume_text, jd_text in test_cases:
            result_json = calculate_weighted_match_score(resume_text, jd_text)
            result = json.loads(result_json)

            assert 0 <= result['overallScore'] <= 100

            for category_data in result['breakdown'].values():
                assert 0 <= category_data['score'] <= 100
                assert 0 <= category_data['weight'] <= 100
                assert 0 <= category_data['contribution'] <= 100

    @patch('scoring_logic._preprocess_text')
    def test_calculate_weighted_match_score_preprocessing_called(self, mock_preprocess):
        """Test that text preprocessing is called"""
        mock_preprocess.side_effect = lambda x: x.lower()

        calculate_weighted_match_score("RESUME TEXT", "JOB DESCRIPTION")

        assert mock_preprocess.call_count == 2

    def test_calculate_weighted_match_score_real_data(self):
        """Test with realistic resume and job description data"""
        resume_text = """
        Senior Software Engineer with 5+ years of experience in Python, JavaScript, React, and AWS.
        Led team of 10 engineers and increased system performance by 40%.
        Strong communication and problem-solving skills.
        Experience with Git, Jira, and Agile development.
        """

        job_description = """
        We are looking for a Senior Software Engineer with:
        - 5+ years of Python and JavaScript experience
        - React and AWS expertise
        - Leadership and communication skills
        - Experience with agile development and Git
        - Proven track record of performance improvements
        """

        result_json = calculate_weighted_match_score(resume_text, job_description)
        result = json.loads(result_json)

        # Should have a reasonable match score
        assert result['overallScore'] > 50  # Should be a decent match
        assert result['breakdown']['hard_skills']['score'] > 0
        assert result['breakdown']['quantifiable_metrics']['score'] == 100  # Has "40%" metric

    def test_calculate_weighted_match_score_empty_inputs(self):
        """Test with empty inputs"""
        result_json = calculate_weighted_match_score("", "")
        result = json.loads(result_json)

        assert result['overallScore'] == 0
        assert all(cat['score'] == 0 for cat in result['breakdown'].values())

    @pytest.mark.parametrize("resume_text,jd_text,expected_min_score", [
        ("Python developer", "Python required", 40),  # Should match hard skills
        ("Led team with communication", "Leadership and communication", 10),  # Soft skills
        ("Used AWS and Git", "AWS and Git experience", 15),  # Tools
        ("Developed and managed", "Development and management", 5),  # Action verbs
        ("Improved by 25%", "Performance improvements", 2),  # Quantifiable
    ])
    def test_calculate_weighted_match_score_category_matching(self, resume_text, jd_text, expected_min_score):
        """Test that different categories contribute to score appropriately"""
        result_json = calculate_weighted_match_score(resume_text, jd_text)
        result = json.loads(result_json)

        assert result['overallScore'] >= expected_min_score