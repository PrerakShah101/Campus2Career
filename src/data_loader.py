"""
Data Loader Module for Campus2Career AI
Loads preprocessed job postings, calculates skill frequencies, and handles role configurations.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

logger = logging.getLogger("data_loader")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
JOBS_CSV = os.path.join(DATA_DIR, "jobs.csv")
METADATA_JSON = os.path.join(DATA_DIR, "dataset_metadata.json")

# Role definitions & metadata with pastel accent colors
ROLE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "Machine Learning Intern": {
        "title": "Machine Learning Intern",
        "badge": "AI & Deep Learning",
        "icon": "🤖",
        "description": "Develop, train, and test intelligent algorithms, deep learning models, and ML pipelines.",
        "bg_color": "#E9E2FF",  # Soft lavender
        "border_color": "#D1C4E9",
        "core_skills": [
            "Python", "Machine Learning", "Deep Learning", "NumPy", 
            "Pandas", "Scikit-learn", "Git", "Data Structures and Algorithms"
        ]
    },
    "Data Scientist": {
        "title": "Data Scientist",
        "badge": "Modeling & Statistics",
        "icon": "🧪",
        "description": "Extract predictive insights, design A/B experiments, and build statistical models from complex data.",
        "bg_color": "#FFE4D6",  # Pale peach
        "border_color": "#FFCCBC",
        "core_skills": [
            "Python", "SQL", "Statistics", "Machine Learning", 
            "Pandas", "NumPy", "Data Visualization", "Scikit-learn"
        ]
    },
    "Data Analyst": {
        "title": "Data Analyst",
        "badge": "Insights & Dashboards",
        "icon": "📊",
        "description": "Translate business data into intuitive dashboards, clear metrics, and actionable decisions.",
        "bg_color": "#DDF3E4",  # Pastel mint
        "border_color": "#C8E6C9",
        "core_skills": [
            "SQL", "Python", "Data Visualization", "Statistics", 
            "Pandas", "Git"
        ]
    },
    "Software Engineer": {
        "title": "Software Engineer",
        "badge": "Systems & APIs",
        "icon": "💻",
        "description": "Architect scalable backend services, clean APIs, and maintainable software systems.",
        "bg_color": "#FFF1B8",  # Butter yellow
        "border_color": "#FFE082",
        "core_skills": [
            "Python", "Java", "C++", "Git", 
            "APIs", "Data Structures and Algorithms", "SQL"
        ]
    }
}

# Skill Categories for the student picker
SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Programming": ["Python", "Java", "C++", "SQL"],
    "Data": ["Pandas", "NumPy", "Data Visualization", "Statistics"],
    "AI and ML": ["Machine Learning", "Deep Learning", "Scikit-learn"],
    "Development": ["Git", "APIs", "Data Structures and Algorithms"]
}


def load_jobs_dataset() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load preprocessed job data from data/jobs.csv.
    If unavailable, automatically trigger fallback generation.
    """
    metadata: Dict[str, Any] = {}
    if os.path.exists(METADATA_JSON):
        try:
            with open(METADATA_JSON, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata json: {e}")

    if os.path.exists(JOBS_CSV):
        try:
            df = pd.read_csv(JOBS_CSV)
            required_columns = {"job_title", "target_role", "company", "description", "skills", "location"}
            if not df.empty and required_columns.issubset(df.columns):
                df = df.dropna(subset=["job_title", "target_role", "description", "skills"])
                for column in required_columns:
                    df[column] = df[column].astype(str).str.strip()
                df = df[(df["job_title"] != "") & (df["description"] != "") & (df["skills"] != "")]
                if not df.empty:
                    return df.reset_index(drop=True), metadata
            logger.warning("jobs.csv is empty or missing required columns; attempting fallback data.")
        except Exception as e:
            logger.warning(f"Failed to read existing jobs.csv: {e}")

    # Fallback to generating dataset
    try:
        from scripts.preprocess_jobs import generate_curated_fallback_dataset
        df, metadata = generate_curated_fallback_dataset()
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(JOBS_CSV, index=False)
        with open(METADATA_JSON, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return df, metadata
    except Exception as e:
        logger.error(f"Failed to generate fallback dataset: {e}")
        # Minimal empty frame
        return pd.DataFrame(columns=["job_title", "target_role", "company", "description", "skills", "location"]), {
            "source": "Empty fallback", "is_fallback": True
        }


def get_role_postings(df: pd.DataFrame, role_name: str) -> pd.DataFrame:
    """Filter job postings by role."""
    if df.empty or "target_role" not in df.columns:
        return pd.DataFrame()
    return df[df["target_role"] == role_name].copy()


def calculate_role_skill_frequencies(df: pd.DataFrame, role_name: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Calculate empirical skill frequencies for the target role from sample job postings.
    Returns list of dicts: [{'skill': 'Python', 'count': 180, 'percentage': 85.0}, ...]
    """
    role_df = get_role_postings(df, role_name)
    total_postings = len(role_df)

    if total_postings == 0:
        # Do not invent counts when the selected role has no observations.
        return []

    skill_counts: Dict[str, int] = {}
    for skills_str in role_df["skills"].dropna():
        if not isinstance(skills_str, str):
            continue
        for s in skills_str.split(","):
            cleaned = s.strip()
            if cleaned:
                skill_counts[cleaned] = skill_counts.get(cleaned, 0) + 1

    sorted_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for skill, count in sorted_skills[:top_n]:
        pct = round((count / total_postings) * 100, 1)
        results.append({
            "skill": skill,
            "count": count,
            "percentage": pct
        })
    return results


def get_all_target_skills(role_name: str) -> List[str]:
    """Retrieve canonical required skill list for a given role."""
    return ROLE_CONFIGS.get(role_name, {}).get("core_skills", [])
