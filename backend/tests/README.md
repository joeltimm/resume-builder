# Resume-Builder Test Suite

This directory contains comprehensive unit tests for the resume-builder backend application.

## Test Structure

``` File Structure

tests/
├── __init__.py              # Makes tests a Python package
├── conftest.py              # Pytest configuration and fixtures
├── test_app.py              # Flask API endpoints tests
├── test_llm_integration.py  # LLM functions tests
├── test_resume_generator.py # Resume generation tests
├── test_scoring_logic.py    # Scoring algorithm tests
├── test_database.py         # Database operation tests
└── README.md               # This file
```

## Prerequisites

1. **Install test dependencies:**

   ```bash
   pip install -r requirements-test.txt
   ```

2. **Install NLTK data (required for scoring logic):**

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

## Running Tests

### Run All Tests

```bash
# From the backend directory
pytest tests/

# Or with verbose output
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test only API endpoints
pytest tests/test_app.py

# Test only LLM integration
pytest tests/test_llm_integration.py

# Test only resume generation
pytest tests/test_resume_generator.py

# Test only scoring logic
pytest tests/test_scoring_logic.py

# Test only database operations
pytest tests/test_database.py
```

### Run Specific Test Classes or Functions

```bash
# Run specific test class
pytest tests/test_app.py::TestResumeEndpoint

# Run specific test function
pytest tests/test_llm_integration.py::TestImproveResumeBullet::test_improve_resume_bullet_success
```

### Run Tests with Coverage

```bash
# Generate coverage report
pytest --cov=. tests/

# Generate HTML coverage report
pytest --cov=. --cov-report=html tests/

# Coverage report will be in htmlcov/index.html
```

### Run Tests in Parallel

```bash
# Run tests in parallel (faster)
pytest tests/ -n auto
```

## Test Categories

### 1. API Endpoint Tests (`test_app.py`)

Tests all Flask API endpoints including:

- `/resume` (GET/POST) - Resume data handling
- `/api/skills` (GET/POST/DELETE) - Skills management
- `/api/accomplishments` (GET/POST/DELETE) - Accomplishments
- `/api/work_experience` (GET/POST/DELETE) - Work experience
- `/api/education` (GET/POST/DELETE) - Education
- `/api/technical_projects` (GET/POST/DELETE) - Projects
- `/api/professional_summaries` (GET/POST/DELETE) - Summaries
- `/api/match` (POST) - Job matching algorithm
- `/calculate-score` (POST) - Resume scoring
- `/improve-bullet` (POST) - LLM bullet improvement
- `/check-duplicates` (POST) - Duplicate detection
- `/generate-ats-resume` (POST) - ATS resume generation
- `/api/export-pdf` (POST) - PDF export functionality

**Key Features:**

- Mocks database connections and external APIs
- Tests both success and failure scenarios
- Validates request/response formats
- Tests error handling

### 2. LLM Integration Tests (`test_llm_integration.py`)

Tests LLM-related functionality:

- `improve_resume_bullet()` - Bullet point improvement
- `find_duplicate_entries()` - Duplicate detection algorithm
- API key validation and error handling
- Anthropic API mocking

**Key Features:**

- Mocks Anthropic API calls
- Tests various edge cases
- Validates fuzzy matching algorithms
- Tests error scenarios (missing API keys, API failures)

### 3. Resume Generation Tests (`test_resume_generator.py`)

Tests resume text generation:

- `generate_ats_resume_text()` - ATS resume formatting
- Section formatting (contact, summary, skills, experience, education)
- Handling of missing or incomplete data
- Output format validation

**Key Features:**

- Tests with various data completeness levels
- Validates ATS-friendly formatting
- Tests edge cases (empty data, missing fields)
- Verifies section ordering

### 4. Scoring Logic Tests (`test_scoring_logic.py`)

Tests resume scoring algorithms:

- `calculate_weighted_match_score()` - Main scoring function
- Text preprocessing functions
- Category-based scoring (skills, tools, soft skills, etc.)
- Quantifiable metrics detection
- JSON output validation

**Key Features:**

- Tests all scoring categories
- Validates weighted scoring algorithm
- Tests text preprocessing
- Mocks external dependencies (TF-IDF, cosine similarity)

### 5. Database Tests (`test_database.py`)

Tests database operations:

- `get_db_connection()` - Database connection
- `setup_database()` - Database initialization
- Table creation and schema validation
- Error handling for connection failures

**Key Features:**

- Mocks PostgreSQL connections
- Tests connection error scenarios
- Validates table schemas
- Tests environment variable handling

## Test Configuration (`conftest.py`)

The configuration file provides shared fixtures for all tests:

### Available Fixtures

- `app` - Configured Flask test app
- `client` - Flask test client
- `mock_db_connection` - Mock database connection
- `sample_resume_data` - Sample resume data
- `sample_job_description` - Sample job description
- `mock_anthropic_client` - Mock Anthropic API client
- `mock_sentence_transformer` - Mock SentenceTransformer
- `sample_skills_data` - Sample skills database data
- `sample_accomplishments_data` - Sample accomplishments data
- `mock_requests_post` - Mock HTTP requests
- `sample_duplicate_entries` - Sample data for duplicate testing

## Mocking Strategy

The tests extensively use mocking to isolate units and avoid dependencies:

1. **Database Operations**: All database calls are mocked to avoid requiring a real database
2. **External APIs**: Anthropic API calls are mocked with predictable responses
3. **ML Models**: SentenceTransformer and other ML models are mocked
4. **HTTP Requests**: PDF generation service calls are mocked
5. **Environment Variables**: Test-specific environment variables are set

## Common Test Patterns

### Testing API Endpoints

```python

def test_endpoint_success(self, client, mock_get_db):
    # Setup mocks
    mock_conn, mock_cursor = Mock(), Mock()
    mock_get_db.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cursor

    # Make request
    response = client.post('/api/endpoint',
                          data=json.dumps(test_data),
                          content_type='application/json')

    # Assertions
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['expected_field'] == 'expected_value'
```

### Testing Functions with Mocks

```python

@patch('module.external_dependency')
def test_function_success(self, mock_dependency):
    # Setup mock
    mock_dependency.return_value = expected_result

    # Call function
    result = function_under_test(input_data)

    # Assertions
    assert result == expected_result
    mock_dependency.assert_called_once_with(expected_args)
```

## Test Data

Test data is defined in fixtures to ensure consistency:

- Resume data includes all possible fields
- Job descriptions are realistic and comprehensive
- Mock responses mirror actual API responses
- Edge cases are covered (empty data, missing fields)

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running tests from the backend directory
2. **NLTK Data**: Download required NLTK data for text processing tests
3. **Environment Variables**: Tests mock environment variables, but some may leak through
4. **Database Mocking**: Ensure all database calls are properly mocked

### Debug Test Failures

```bash

# Run with verbose output and stop on first failure
pytest tests/ -v -x

# Run specific failing test with full output
pytest tests/test_file.py::test_function -v -s

# Print variables during test execution
pytest tests/ -v -s --capture=no
```

## Continuous Integration

The test suite is designed to run in CI/CD environments:

```yaml
# Example GitHub Actions workflow
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: pytest tests/ --cov=. --cov-report=xml
```

## Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain test coverage above 90%
4. Update this README if adding new test categories

## Coverage Goals

Target coverage levels:

- **Overall**: 90%+
- **API endpoints**: 95%+
- **Core business logic**: 95%+
- **Database operations**: 90%+
- **Error handling**: 85%+
