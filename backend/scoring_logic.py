import re
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# --- Predefined Keyword Lists & Weights ---
HARD_SKILLS = ['python', 'django', 'react', 'sql', 'agile', 'database management', 'financial modeling']
TOOLS = ['aws', 'jira', 'git', 'salesforce', 'figma', 'tableau']
SOFT_SKILLS = ['leadership', 'communication', 'teamwork', 'problem-solving']
ACTION_VERBS = ['developed', 'managed', 'engineered', 'proven', 'working', 'orchestrated', 'quantified', 'streamlined']
QUANTIFIABLE_REGEX = r'\b\d+(\.\d+)?%?\b|\$\d+'

CATEGORY_WEIGHTS = {
    "hard_skills": 0.50,
    "tools": 0.20,
    "soft_skills": 0.15,
    "action_verbs": 0.10,
    "quantifiable_metrics": 0.05
}

def _preprocess_text(text):
    """Internal helper function to clean and preprocess text."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def _extract_keywords_by_category(text, keyword_list):
    """Internal helper to extract words from text that are in a keyword list."""
    tokens = word_tokenize(text.lower())
    found_keywords = [word for word in tokens if word in keyword_list]
    return " ".join(found_keywords)

def _calculate_category_score(resume_text, jd_text, keyword_list):
    """Internal helper to calculate cosine similarity for a specific keyword category."""
    resume_keywords = _extract_keywords_by_category(resume_text, keyword_list)
    jd_keywords = _extract_keywords_by_category(jd_text, keyword_list)

    if not resume_keywords.strip() or not jd_keywords.strip():
        return 0.0

    corpus = [resume_keywords, jd_keywords]
    vectorizer = TfidfVectorizer()

    try:
        tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Get individual row vectors safely using index 0 and 1
    # This avoids slicing and is more clearly supported by Pylance
    
        doc1_vector = tfidf_matrix[0]  # type:ignore
        doc2_vector = tfidf_matrix[1]  #type:ignore
    
    # Convert to dense arrays if needed for cosine_similarity
    # But cosine_similarity accepts sparse matrices directly!
        similarity = cosine_similarity(doc1_vector, doc2_vector)[0][0]
        return float(similarity)
    except ValueError as e:
        if "empty vocabulary" in str(e):
            return 0.0
        else:
            print(f"Unexpected ValueError in TfidfVectorizer: {e}")
        return 0.0
    except Exception as e:
        print(f"Unexpected error in category score calculation: {e}")
        return 0.0

def _calculate_quantifiable_score(resume_text):
    """Internal helper to score based on the presence of quantifiable metrics."""
    resume_metrics = re.findall(QUANTIFIABLE_REGEX, resume_text)
    return 1.0 if len(resume_metrics) > 0 else 0.0

def calculate_weighted_match_score(resume_text, job_description_text):
    """
    Calculates a weighted match score and provides a detailed breakdown.
    This is the main function to be called from your application.

    Returns:
        str: A JSON string containing the overall score and a detailed breakdown.
    """
    scores = {}
    
    processed_resume = _preprocess_text(resume_text)
    processed_jd = _preprocess_text(job_description_text)

    scores["hard_skills"] = _calculate_category_score(processed_resume, processed_jd, HARD_SKILLS)
    scores["tools"] = _calculate_category_score(processed_resume, processed_jd, TOOLS)
    scores["soft_skills"] = _calculate_category_score(processed_resume, processed_jd, SOFT_SKILLS)
    scores["action_verbs"] = _calculate_category_score(processed_resume, processed_jd, ACTION_VERBS)
    scores["quantifiable_metrics"] = _calculate_quantifiable_score(resume_text)

    overall_score = 0.0
    breakdown = {}
    for category, weight in CATEGORY_WEIGHTS.items():
        score = scores[category]
        weighted_score = score * weight
        overall_score += weighted_score
        breakdown[category] = {
            "score": int(round(score * 100)),
            "weight": int(weight * 100),
            "contribution": int(round(weighted_score * 100))
        }

    final_score_percentage = int(round(overall_score * 100))

    result = {
        "overallScore": final_score_percentage,
        "breakdown": breakdown
    }

    return json.dumps(result, indent=4)
