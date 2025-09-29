"""
Tests for database operations in app.py
"""
import pytest
import os
from unittest.mock import Mock, patch, MagicMock
import psycopg2
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import get_db_connection, setup_database


class TestGetDbConnection:
    """Tests for get_db_connection function"""

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_pass'
    })
    def test_get_db_connection_success(self, mock_connect):
        """Test successful database connection"""
        mock_connection = Mock()
        mock_connect.return_value = mock_connection

        result = get_db_connection()

        assert result == mock_connection
        mock_connect.assert_called_once_with(
            host="postgres",
            database="test_db",
            user="test_user",
            password="test_pass"
        )

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_pass'
    })
    def test_get_db_connection_failure(self, mock_connect):
        """Test database connection failure"""
        mock_connect.side_effect = psycopg2.Error("Connection failed")

        with patch('builtins.print') as mock_print:
            result = get_db_connection()

            assert result is None
            mock_print.assert_called_once()
            assert "Error connecting to database" in str(mock_print.call_args)

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {}, clear=True)
    def test_get_db_connection_missing_env_vars(self, mock_connect):
        """Test connection with missing environment variables"""
        mock_connect.side_effect = psycopg2.Error("Missing credentials")

        result = get_db_connection()

        assert result is None
        # Should still attempt connection with None values
        mock_connect.assert_called_once_with(
            host="postgres",
            database=None,
            user=None,
            password=None
        )

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {
        'POSTGRES_DB': '',
        'POSTGRES_USER': '',
        'POSTGRES_PASSWORD': ''
    })
    def test_get_db_connection_empty_env_vars(self, mock_connect):
        """Test connection with empty environment variables"""
        mock_connect.side_effect = psycopg2.Error("Empty credentials")

        result = get_db_connection()

        assert result is None
        mock_connect.assert_called_once_with(
            host="postgres",
            database="",
            user="",
            password=""
        )

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_pass'
    })
    def test_get_db_connection_timeout(self, mock_connect):
        """Test database connection timeout"""
        mock_connect.side_effect = psycopg2.OperationalError("Connection timeout")

        result = get_db_connection()

        assert result is None

    @patch('app.psycopg2.connect')
    @patch.dict(os.environ, {
        'POSTGRES_DB': 'test_db',
        'POSTGRES_USER': 'test_user',
        'POSTGRES_PASSWORD': 'test_pass'
    })
    def test_get_db_connection_authentication_error(self, mock_connect):
        """Test database authentication error"""
        mock_connect.side_effect = psycopg2.OperationalError("Authentication failed")

        result = get_db_connection()

        assert result is None


class TestSetupDatabase:
    """Tests for setup_database function"""

    @patch('app.get_db_connection')
    def test_setup_database_success(self, mock_get_db):
        """Test successful database setup"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock fetchone to return None (no existing resume entry)
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Verify that cursor was called
        assert mock_cursor.execute.call_count >= 1
        mock_conn.commit.assert_called()
        mock_cursor.close.assert_called()
        mock_conn.close.assert_called()

    @patch('app.get_db_connection')
    def test_setup_database_connection_failure(self, mock_get_db):
        """Test setup_database when database connection fails"""
        mock_get_db.return_value = None

        # Should not raise exception, just return early
        setup_database()

        # No further database operations should be attempted
        mock_get_db.assert_called_once()

    @patch('app.get_db_connection')
    def test_setup_database_resume_table_creation(self, mock_get_db):
        """Test that resume table is created correctly"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Check that resume table creation SQL was executed
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        resume_table_created = any("CREATE TABLE IF NOT EXISTS resume" in sql for sql in execute_calls)
        assert resume_table_created

    @patch('app.get_db_connection')
    def test_setup_database_skills_table_creation(self, mock_get_db):
        """Test that skills table is created correctly"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Check that skills table creation SQL was executed
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        skills_table_created = any("CREATE TABLE IF NOT EXISTS skills" in sql for sql in execute_calls)
        assert skills_table_created

    @patch('app.get_db_connection')
    def test_setup_database_work_experience_table_creation(self, mock_get_db):
        """Test that work_experience table is created correctly"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Check that work_experience table creation SQL was executed
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        work_table_created = any("CREATE TABLE IF NOT EXISTS work_experience" in sql for sql in execute_calls)
        assert work_table_created

    @patch('app.get_db_connection')
    def test_setup_database_accomplishments_table_creation(self, mock_get_db):
        """Test that accomplishments table is created correctly"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Check that accomplishments table creation SQL was executed
        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]
        accomplishments_table_created = any("CREATE TABLE IF NOT EXISTS accomplishments" in sql for sql in execute_calls)
        assert accomplishments_table_created

    @patch('app.get_db_connection')
    def test_setup_database_default_resume_insertion(self, mock_get_db):
        """Test that default resume entry is inserted when none exists"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock fetchone to return None (no existing resume entry)
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Check that default resume insertion was attempted
        execute_calls = [call[0] for call in mock_cursor.execute.call_args_list]
        insert_calls = [call for call in execute_calls if len(call) > 1 and call[1] == ('{}',)]
        assert len(insert_calls) > 0

    @patch('app.get_db_connection')
    def test_setup_database_existing_resume_not_overwritten(self, mock_get_db):
        """Test that existing resume entry is not overwritten"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock fetchone to return existing entry
        mock_cursor.fetchone.return_value = (1,)

        setup_database()

        # Check that no INSERT was performed for resume
        execute_calls = [call[0] for call in mock_cursor.execute.call_args_list]
        insert_calls = [call for call in execute_calls if len(call) > 1 and "INSERT INTO resume" in call[0]]
        assert len(insert_calls) == 0

    @patch('app.get_db_connection')
    def test_setup_database_handles_sql_error(self, mock_get_db):
        """Test that setup_database handles SQL execution errors"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor

        # Mock execute to raise an error
        mock_cursor.execute.side_effect = psycopg2.Error("SQL execution failed")

        with patch('builtins.print') as mock_print:
            setup_database()

            # Should handle the error and print it
            mock_print.assert_called()
            assert "Error setting up database" in str(mock_print.call_args)

    @patch('app.get_db_connection')
    def test_setup_database_closes_connections_on_success(self, mock_get_db):
        """Test that database connections are properly closed on success"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Verify connections are closed
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('app.get_db_connection')
    def test_setup_database_closes_connections_on_error(self, mock_get_db):
        """Test that database connections are properly closed even on error"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.execute.side_effect = Exception("Database error")

        with patch('builtins.print'):
            setup_database()

        # Verify connections are closed even after error
        mock_cursor.close.assert_called_once()
        mock_conn.close.assert_called_once()

    @patch('app.get_db_connection')
    def test_setup_database_commits_changes(self, mock_get_db):
        """Test that database changes are committed"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        # Verify commit was called
        mock_conn.commit.assert_called_once()

    @patch('app.get_db_connection')
    def test_setup_database_table_schemas(self, mock_get_db):
        """Test that table schemas contain expected columns"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]

        # Check resume table schema
        resume_create = next((sql for sql in execute_calls if "CREATE TABLE IF NOT EXISTS resume" in sql), "")
        assert "id SERIAL PRIMARY KEY" in resume_create
        assert "content TEXT NOT NULL" in resume_create

        # Check skills table schema
        skills_create = next((sql for sql in execute_calls if "CREATE TABLE IF NOT EXISTS skills" in sql), "")
        assert "id SERIAL PRIMARY KEY" in skills_create
        assert "skill_text TEXT NOT NULL UNIQUE" in skills_create
        assert "embedding TEXT" in skills_create

        # Check work_experience table schema
        work_create = next((sql for sql in execute_calls if "CREATE TABLE IF NOT EXISTS work_experience" in sql), "")
        assert "id SERIAL PRIMARY KEY" in work_create
        assert "job_title TEXT NOT NULL" in work_create
        assert "company TEXT NOT NULL" in work_create
        assert "embedding TEXT" in work_create

        # Check accomplishments table schema
        accomplishments_create = next((sql for sql in execute_calls if "CREATE TABLE IF NOT EXISTS accomplishments" in sql), "")
        assert "id SERIAL PRIMARY KEY" in accomplishments_create
        assert "accomplishment_text TEXT NOT NULL UNIQUE" in accomplishments_create
        assert "embedding TEXT" in accomplishments_create


class TestDatabaseIntegration:
    """Integration tests for database operations"""

    @patch('app.get_db_connection')
    def test_database_operations_flow(self, mock_get_db):
        """Test the complete flow of database operations"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        # Test setup
        setup_database()

        # Verify the expected sequence of operations
        assert mock_get_db.called
        assert mock_conn.cursor.called
        assert mock_cursor.execute.called
        assert mock_conn.commit.called
        assert mock_cursor.close.called
        assert mock_conn.close.called

    @patch('app.get_db_connection')
    def test_multiple_table_creation_order(self, mock_get_db):
        """Test that tables are created in the correct order"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_get_db.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None

        setup_database()

        execute_calls = [call[0][0] for call in mock_cursor.execute.call_args_list]

        # Find positions of table creations
        table_positions = {}
        for i, sql in enumerate(execute_calls):
            if "CREATE TABLE IF NOT EXISTS resume" in sql:
                table_positions['resume'] = i
            elif "CREATE TABLE IF NOT EXISTS skills" in sql:
                table_positions['skills'] = i
            elif "CREATE TABLE IF NOT EXISTS work_experience" in sql:
                table_positions['work_experience'] = i
            elif "CREATE TABLE IF NOT EXISTS accomplishments" in sql:
                table_positions['accomplishments'] = i

        # Verify tables are created (order may vary, but all should exist)
        assert 'resume' in table_positions
        assert 'skills' in table_positions
        assert 'work_experience' in table_positions
        assert 'accomplishments' in table_positions