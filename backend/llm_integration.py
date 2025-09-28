import os
import requests
import json
import anthropic
from rapidfuzz import fuzz
from itertools import combinations

def get_available_models():
    """
    Fetches available models from LM Studio.

    Returns:
        list: List of available model names, or empty list if error.
    """
    try:
        lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://100.98.99.49:6969")
        response = requests.get(f"{lm_studio_url}/v1/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            return [model['id'] for model in models_data.get('data', [])]
        return []
    except Exception as e:
        print(f"Error fetching LM Studio models: {e}")
        return []

def improve_resume_bullet(bullet_point, job_title, industry, model_name=None):
    """
    Uses either Anthropic's Claude or local LM Studio to improve a resume bullet point.

    Args:
        bullet_point (str): The original bullet point from the user's resume.
        job_title (str): The target job title to tailor the bullet point for.
        industry (str): The target industry.
        model_name (str, optional): Specific model to use (for LM Studio mode).

    Returns:
        str: The improved bullet point, or an error message.
    """
    llm_mode = os.environ.get("LLM_MODE", "production").lower()

    if llm_mode == "local":
        return _improve_with_lm_studio(bullet_point, job_title, industry, model_name)
    else:
        return _improve_with_anthropic(bullet_point, job_title, industry)

def _improve_with_lm_studio(bullet_point, job_title, industry, model_name=None):
    """Uses LM Studio local LLM to improve resume bullet point."""
    try:
        lm_studio_url = os.environ.get("LM_STUDIO_URL", "http://100.98.99.49:6969")
        default_model = os.environ.get("LM_STUDIO_DEFAULT_MODEL", "qwen2.5-32b-instruct")
        selected_model = model_name or default_model

        system_prompt = f"""You are an expert resume writer and career coach specializing in the {industry} industry.
Your task is to rewrite a single resume bullet point to make it more impactful for a '{job_title}' position.

Follow these rules strictly:
1. Rephrase using the STAR (Situation, Task, Action, Result) method.
2. Use strong, professional action verbs.
3. Quantify the result with specific metrics if possible.
4. The output must be a single, revised bullet point and nothing else.
5. Do not include any explanations or additional text."""

        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Original bullet point: '{bullet_point}'"
                }
            ],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": False
        }

        response = requests.post(
            f"{lm_studio_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            improved_text = result['choices'][0]['message']['content']
            return improved_text.strip()
        else:
            return f"Error: LM Studio API returned status {response.status_code}"

    except requests.exceptions.Timeout:
        return "Error: LM Studio request timed out. Check if LM Studio is running."
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to LM Studio. Check if it's running at the configured URL."
    except Exception as e:
        return f"Error with LM Studio API: {str(e)}"

def _improve_with_anthropic(bullet_point, job_title, industry):
    """Uses Anthropic Claude to improve resume bullet point."""
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return "Error: ANTHROPIC_API_KEY environment variable not set."

        if not api_key.strip():
            return "Error: ANTHROPIC_API_KEY is empty."

        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = f"""
        You are an expert resume writer and career coach specializing in the {industry} industry.
        Your task is to rewrite a single resume bullet point to make it more impactful for a '{job_title}' position.
        Follow these rules strictly:
        1. Rephrase using the STAR (Situation, Task, Action, Result) method.
        2. Use strong, professional action verbs.
        3. Quantify the result with specific metrics if possible.
        4. The output must be a single, revised bullet point and nothing else.
        """

        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=256,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": f"Original bullet point: '{bullet_point}'"
                }
            ]
        )

        improved_text = message.content[0].text
        return improved_text.strip()

    except Exception as e:
        return f"An error occurred with the Anthropic API: {str(e)}"
    
def find_duplicate_entries(entries, threshold=90):
    """
    Identifies duplicate or highly similar strings within a list using fuzzy matching.

    Args:
        entries (list of str): A list of strings to check (e.g., resume bullet points).
        threshold (int): The similarity score (0-100) to consider items duplicates.

    Returns:
        list of dict: A list detailing the pairs of duplicate entries found.
    """
    duplicates = []
    indexed_entries = list(enumerate(entries))

    for (idx1, entry1), (idx2, entry2) in combinations(indexed_entries, 2):
        # token_set_ratio is robust against word order and partial matches 
        similarity_score = fuzz.token_set_ratio(entry1, entry2)

        if similarity_score >= threshold:
            duplicates.append({
                "indices": [idx1, idx2],
                "entries": [entry1, entry2],
                "score": round(similarity_score, 2)
            })
    
    return duplicates