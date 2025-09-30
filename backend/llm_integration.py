# resume-builder/backend/llm_integration.py

import requests
import os
import json
import time
from typing import List, Dict, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Configuration ---
LLM_URL = os.environ.get("LLM_URL", "http://100.98.99.49:8081")
LLM_API_PATH = "/v1/chat/completions"
DEFAULT_MODEL = os.environ.get("LM_STUDIO_DEFAULT_MODEL", "qwen2.5-32b-instruct")

# Timeout configuration (in seconds)
LLM_CONNECT_TIMEOUT = int(os.environ.get("LLM_CONNECT_TIMEOUT", "10"))
LLM_READ_TIMEOUT = int(os.environ.get("LLM_READ_TIMEOUT", "60"))

# Cache file
CACHE_DIR = "/app/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "models.json")
CACHE_TIMEOUT = 300  # 5 minutes

# --- Ensure cache directory exists ---
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_retry_session(retries=3, backoff_factor=0.3):
    """
    Create a requests session with retry logic for transient failures.
    """
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=["GET", "POST"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _parse_llm_json_response(content: str) -> List[str]:
    """
    Robust JSON parsing for LLM responses with multiple fallback strategies.
    """
    # Strategy 1: Direct JSON parse
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return [str(item) for item in data]
        elif isinstance(data, dict) and "suggestions" in data:
            return data["suggestions"]
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Extract JSON from markdown code blocks
    if "```json" in content:
        try:
            json_str = content.split("```json")[1].split("```")[0].strip()
            data = json.loads(json_str)
            if isinstance(data, list):
                return [str(item) for item in data]
        except (IndexError, json.JSONDecodeError):
            pass
    
    # Strategy 3: Extract array-like content
    if content.strip().startswith('[') and content.strip().endswith(']'):
        try:
            data = json.loads(content.strip())
            if isinstance(data, list):
                return [str(item) for item in data]
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Line-by-line extraction (last resort)
    lines = content.split('\n')
    items = []
    for line in lines:
        line = line.strip()
        # Remove common prefixes
        line = line.lstrip('- ').lstrip('* ').lstrip('• ')
        # Remove quotes and numbering
        line = line.strip('"').strip("'").lstrip('0123456789. ')
        if line and len(line) > 3:  # Ignore very short lines
            items.append(line)
    
    return items if items else []


# --- Helper: Get Available Models from llama.cpp ---
def get_available_models() -> List[Dict[str, str]]:
    """
    Returns a list of available models from llama.cpp, cached for 5 minutes.
    """
    # Check cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                if time.time() - cache_data.get("timestamp", 0) < CACHE_TIMEOUT:
                    return cache_data.get("models", [])
        except Exception as e:
            print(f"Cache read failed: {e}")

    # Fetch fresh models with retry logic
    session = _get_retry_session()
    try:
        response = session.get(
            f"{LLM_URL}/v1/models",
            timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
        )
        if response.status_code == 200:
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model["id"]
                size_gb = "N/A"
                if "7b" in model_id.lower():
                    size_gb = "7"
                elif "13b" in model_id.lower():
                    size_gb = "13"
                elif "32b" in model_id.lower():
                    size_gb = "32"
                models.append({
                    "id": model_id,
                    "size_gb": size_gb
                })

            # Save to cache
            cache_data = {
                "timestamp": time.time(),
                "models": models
            }
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            return models
        else:
            print(f"LLM API error: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print(f"Timeout fetching models from {LLM_URL}")
    except requests.exceptions.ConnectionError:
        print(f"Connection error to {LLM_URL}")
    except Exception as e:
        print(f"Error fetching models: {e}")
    finally:
        session.close()

    return []


# --- LLM Function: Analyze Job Description ---
def analyze_job_description_with_llm(job_description: str, user_data: Dict, model_name: Optional[str]) -> Dict:
    """
    Uses llama.cpp to analyze job description and find relevant skills/accomplishments.
    Returns: {'suggestions': [...], 'missing_keywords': [...]}
    """
    prompt = f"""
You are an AI assistant helping a job seeker match their resume to a job description.

Job Description:
{job_description.strip()}

Your Task:
1. Identify the most relevant skills and accomplishments from the user's data that match this job.
2. Return only the exact text (no explanation) of the top 8-12 most relevant items.
3. Do not include duplicates.
4. If no relevant items found, return an empty list.

User Data:
- Skills: {', '.join(user_data.get('skills', []))}
- Accomplishments: {', '.join(user_data.get('accomplishments', []))}

Output:
Return a JSON array of strings, e.g., ["Python", "Docker", "Led team of 5"].
IMPORTANT: Return ONLY the JSON array, no other text.
"""

    session = _get_retry_session()
    try:
        response = session.post(
            f"{LLM_URL}{LLM_API_PATH}",
            json={
                "model": model_name or DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful resume matcher. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 512,
                "stream": False
            },
            timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
        )

        if response.status_code != 200:
            return {"error": f"LLM API error: {response.status_code} - {response.text}"}

        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        # Use robust parsing
        suggestions = _parse_llm_json_response(content)

        # Find missing keywords (simplified)
        job_keywords = set(word.lower() for word in job_description.split() if len(word) > 3)
        user_texts = user_data.get('skills', []) + user_data.get('accomplishments', [])
        user_keywords = set()
        for text in user_texts:
            user_keywords.update(word.lower() for word in text.split() if len(word) > 3)

        missing_keywords = list(job_keywords - user_keywords)[:20]  # Limit to 20

        return {
            "suggestions": suggestions,
            "missing_keywords": missing_keywords
        }

    except requests.exceptions.Timeout:
        return {"error": "LLM request timed out. Try again or check your LLM server."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to LLM server. Check if it's running."}
    except Exception as e:
        return {"error": f"LLM analysis failed: {str(e)}"}
    finally:
        session.close()


# --- LLM Function: Improve Resume Bullet ---
def improve_resume_bullet(
    bullet_point: str,
    job_title: str,
    industry: str,
    job_description: str,
    model_name: Optional[str]
) -> str:
    """
    Uses llama.cpp to rephrase a resume bullet point to better match the job.
    Returns: improved bullet point as string
    """
    prompt = f"""
Rewrite the following resume bullet point to better match the job description and industry.

Job Title: {job_title}
Industry: {industry}
Job Description:
{job_description.strip()}

Original bullet:
"{bullet_point}"

Rewrite to make it more relevant, concise, and ATS-friendly. Keep the meaning but use keywords from the job description. Return only the rewritten bullet point.
"""

    session = _get_retry_session()
    try:
        response = session.post(
            f"{LLM_URL}{LLM_API_PATH}",
            json={
                "model": model_name or DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a professional resume writer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.5,
                "max_tokens": 150,
                "stream": False
            },
            timeout=(LLM_CONNECT_TIMEOUT, LLM_READ_TIMEOUT)
        )

        if response.status_code != 200:
            return f"Error: LLM API failed with status {response.status_code}"

        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return content

    except requests.exceptions.Timeout:
        return "Error: Request timed out. The LLM took too long to respond."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to LLM server."
    except Exception as e:
        return f"Error: Failed to improve bullet: {str(e)}"
    finally:
        session.close()


# --- LLM Function: Check for Duplicates ---
def find_duplicate_entries(bullet_points: List[str]) -> List[str]:
    """
    Uses LLM to detect if any bullet points are duplicates or very similar.
    Returns: list of duplicate/very similar items.
    """
    if len(bullet_points) < 2:
        return []

    prompt = f"""
Determine which of the following bullet points are duplicates or very similar in meaning.

Bullet points:
{chr(10).join(bullet_points)}

Return only a JSON array of strings containing the duplicate or very similar ones. If none, return an empty array [].
"""

    session = _get_retry_session(retries=2)
    try:
        response = session.post(
            f"{LLM_URL}{LLM_API_PATH}",
            json={
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a resume consistency checker. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 200,
                "stream": False
            },
            timeout=(LLM_CONNECT_TIMEOUT, 30)  # Shorter timeout for this function
        )

        if response.status_code != 200:
            return []

        content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        duplicates = _parse_llm_json_response(content)
        return duplicates

    except Exception as e:
        print(f"Duplicate check failed: {e}")
        return []
    finally:
        session.close()