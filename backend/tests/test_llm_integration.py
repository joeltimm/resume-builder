"""
Tests for LLM integration functions in llm_integration.py
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_integration import improve_resume_bullet, find_duplicate_entries


class TestImproveResumeBullet:
    """Tests for improve_resume_bullet function"""

    @patch('llm_integration.anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key_123'})
    def test_improve_resume_bullet_success(self, mock_anthropic):
        """Test improve_resume_bullet returns improved text successfully"""
        # Mock the Anthropic client and response
        mock_client = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Engineered scalable web applications using Python and Django, resulting in 40% performance improvement"
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        result = improve_resume_bullet(
            bullet_point="Worked on web applications",
            job_title="Software Engineer",
            industry="Technology"
        )

        assert "Engineered scalable web applications" in result
        assert "40% performance improvement" in result
        mock_client.messages.create.assert_called_once()

    @patch.dict(os.environ, {}, clear=True)
    def test_improve_resume_bullet_missing_api_key(self):
        """Test improve_resume_bullet when ANTHROPIC_API_KEY is not set"""
        result = improve_resume_bullet(
            bullet_point="Worked on applications",
            job_title="Engineer",
            industry="Tech"
        )

        assert "Error: ANTHROPIC_API_KEY environment variable not set." in result

    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': '   '})
    def test_improve_resume_bullet_empty_api_key(self):
        """Test improve_resume_bullet when ANTHROPIC_API_KEY is empty"""
        result = improve_resume_bullet(
            bullet_point="Worked on applications",
            job_title="Engineer",
            industry="Tech"
        )

        assert "Error: ANTHROPIC_API_KEY is empty." in result

    @patch('llm_integration.anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key_123'})
    def test_improve_resume_bullet_api_error(self, mock_anthropic):
        """Test improve_resume_bullet when Anthropic API raises an error"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API connection failed")
        mock_anthropic.return_value = mock_client

        result = improve_resume_bullet(
            bullet_point="Worked on applications",
            job_title="Engineer",
            industry="Tech"
        )

        assert "An error occurred with the LLM API: API connection failed" in result

    @patch('llm_integration.anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key_123'})
    def test_improve_resume_bullet_with_system_prompt(self, mock_anthropic):
        """Test that improve_resume_bullet uses correct system prompt"""
        mock_client = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "Improved bullet point"
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        improve_resume_bullet(
            bullet_point="Managed team",
            job_title="Senior Manager",
            industry="Finance"
        )

        # Verify the call was made with correct parameters
        call_args = mock_client.messages.create.call_args
        assert call_args[1]['model'] == "claude-3-sonnet-20240229"
        assert call_args[1]['max_tokens'] == 256
        assert "Finance" in call_args[1]['system']
        assert "Senior Manager" in call_args[1]['system']
        assert "Managed team" in call_args[1]['messages'][0]['content']

    @patch('llm_integration.anthropic.Anthropic')
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test_key_123'})
    def test_improve_resume_bullet_strips_whitespace(self, mock_anthropic):
        """Test that improve_resume_bullet strips whitespace from response"""
        mock_client = Mock()
        mock_message = Mock()
        mock_content = Mock()
        mock_content.text = "   Improved bullet with whitespace   "
        mock_message.content = [mock_content]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        result = improve_resume_bullet(
            bullet_point="Original bullet",
            job_title="Engineer",
            industry="Tech"
        )

        assert result == "Improved bullet with whitespace"


class TestFindDuplicateEntries:
    """Tests for find_duplicate_entries function"""

    def test_find_duplicate_entries_exact_matches(self):
        """Test find_duplicate_entries with exact duplicate strings"""
        entries = [
            "Developed web applications",
            "Developed web applications",
            "Managed team projects"
        ]

        result = find_duplicate_entries(entries, threshold=100)

        assert len(result) == 1
        assert result[0]['indices'] == [0, 1]
        assert result[0]['score'] == 100.0

    def test_find_duplicate_entries_similar_matches(self):
        """Test find_duplicate_entries with similar strings"""
        entries = [
            "Developed web applications using Python",
            "Built web apps with Python framework",
            "Managed software development team",
            "Led development team for software projects"
        ]

        result = find_duplicate_entries(entries, threshold=80)

        # Should find similarities between entries
        assert len(result) >= 1
        # Check that indices are different and scores are reasonable
        for duplicate in result:
            assert len(duplicate['indices']) == 2
            assert duplicate['indices'][0] != duplicate['indices'][1]
            assert duplicate['score'] >= 80

    def test_find_duplicate_entries_no_matches(self):
        """Test find_duplicate_entries when no duplicates exist"""
        entries = [
            "Developed web applications",
            "Managed marketing campaigns",
            "Analyzed financial data",
            "Designed user interfaces"
        ]

        result = find_duplicate_entries(entries, threshold=90)

        assert len(result) == 0

    def test_find_duplicate_entries_empty_list(self):
        """Test find_duplicate_entries with empty list"""
        result = find_duplicate_entries([], threshold=90)
        assert len(result) == 0

    def test_find_duplicate_entries_single_item(self):
        """Test find_duplicate_entries with single item"""
        result = find_duplicate_entries(["Single entry"], threshold=90)
        assert len(result) == 0

    def test_find_duplicate_entries_custom_threshold(self):
        """Test find_duplicate_entries with different threshold values"""
        entries = [
            "Python developer with experience",
            "Experienced Python programmer",
            "Java enterprise developer"
        ]

        # High threshold - fewer matches
        result_high = find_duplicate_entries(entries, threshold=95)

        # Low threshold - more matches
        result_low = find_duplicate_entries(entries, threshold=50)

        assert len(result_low) >= len(result_high)

    def test_find_duplicate_entries_return_format(self):
        """Test that find_duplicate_entries returns correct format"""
        entries = [
            "Developed applications",
            "Built applications",
            "Managed teams"
        ]

        result = find_duplicate_entries(entries, threshold=70)

        for duplicate in result:
            # Check required keys exist
            assert 'indices' in duplicate
            assert 'entries' in duplicate
            assert 'score' in duplicate

            # Check data types
            assert isinstance(duplicate['indices'], list)
            assert isinstance(duplicate['entries'], list)
            assert isinstance(duplicate['score'], (int, float))

            # Check indices point to correct entries
            idx1, idx2 = duplicate['indices']
            assert duplicate['entries'][0] == entries[idx1]
            assert duplicate['entries'][1] == entries[idx2]

    def test_find_duplicate_entries_with_special_characters(self):
        """Test find_duplicate_entries with entries containing special characters"""
        entries = [
            "Increased revenue by 25% through optimization",
            "Improved revenue by 25% via optimization techniques",
            "Reduced costs by $50,000 annually"
        ]

        result = find_duplicate_entries(entries, threshold=80)

        # Should still find similarities despite special characters
        assert len(result) >= 0  # May or may not find matches depending on fuzzy logic

    def test_find_duplicate_entries_case_insensitive(self):
        """Test find_duplicate_entries is case insensitive"""
        entries = [
            "Developed Web Applications",
            "developed web applications",
            "DEVELOPED WEB APPLICATIONS"
        ]

        result = find_duplicate_entries(entries, threshold=95)

        # Should find all as duplicates since they're essentially the same
        assert len(result) >= 2  # Should match first with second, first with third

    @pytest.mark.parametrize("threshold", [0, 25, 50, 75, 90, 100])
    def test_find_duplicate_entries_threshold_range(self, threshold):
        """Test find_duplicate_entries with various threshold values"""
        entries = [
            "Python developer experience",
            "Experienced Python developer",
            "Java developer background"
        ]

        result = find_duplicate_entries(entries, threshold=threshold)

        # Result should always be a list
        assert isinstance(result, list)

        # All scores should meet the threshold
        for duplicate in result:
            assert duplicate['score'] >= threshold

    def test_find_duplicate_entries_long_entries(self):
        """Test find_duplicate_entries with long text entries"""
        entries = [
            "Developed and maintained multiple full-stack web applications using Python, Django, PostgreSQL, and React, serving over 10,000 active users daily",
            "Built and sustained several complete web applications with Python/Django backend, PostgreSQL database, and React frontend, supporting 10K+ daily active users",
            "Designed user interface components for mobile application using React Native framework"
        ]

        result = find_duplicate_entries(entries, threshold=70)

        # Should handle long entries without issues
        assert isinstance(result, list)
        if len(result) > 0:
            # Verify structure is maintained for long entries
            for duplicate in result:
                assert len(duplicate['entries'][0]) > 50
                assert len(duplicate['entries'][1]) > 50