"""
Skill Matcher Module for Campus2Career AI
Performs exact matching, alias mapping, and semantic vector similarity analysis
between student skills and target role requirements.
"""

import os
import re
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import ROLE_CONFIGS, get_all_target_skills

logger = logging.getLogger("skill_matcher")

# Comprehensive alias normalization dictionary
SKILL_ALIASES: Dict[str, str] = {
    # Programming
    "py": "Python",
    "python3": "Python",
    "python 3": "Python",
    "core java": "Java",
    "c plus plus": "C++",
    "cpp": "C++",
    "c/c++": "C++",
    "sql queries": "SQL",
    "relational database": "SQL",
    "rdbms": "SQL",
    "postgresql": "SQL",
    "postgres": "SQL",
    "mysql": "SQL",
    
    # Data & Stats
    "pandas library": "Pandas",
    "numpy library": "NumPy",
    "data viz": "Data Visualization",
    "visualization": "Data Visualization",
    "matplotlib": "Data Visualization",
    "seaborn": "Data Visualization",
    "tableau": "Data Visualization",
    "power bi": "Data Visualization",
    "powerbi": "Data Visualization",
    "stats": "Statistics",
    "statistical modeling": "Statistics",
    "probability and statistics": "Statistics",
    "hypothesis testing": "Statistics",
    
    # AI & ML
    "ml": "Machine Learning",
    "machine-learning": "Machine Learning",
    "applied ml": "Machine Learning",
    "deep-learning": "Deep Learning",
    "dl": "Deep Learning",
    "neural networks": "Deep Learning",
    "artificial neural networks": "Deep Learning",
    "pytorch": "Deep Learning",
    "tensorflow": "Deep Learning",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "scikitlearn": "Scikit-learn",
    
    # Development & Systems
    "github": "Git",
    "gitlab": "Git",
    "version control": "Git",
    "rest apis": "APIs",
    "rest api": "APIs",
    "restful": "APIs",
    "fastapi": "APIs",
    "flask api": "APIs",
    "dsa": "Data Structures and Algorithms",
    "data structures": "Data Structures and Algorithms",
    "algorithms": "Data Structures and Algorithms",
    "ds & algo": "Data Structures and Algorithms",
    "ds and algo": "Data Structures and Algorithms",
    "data structures & algorithms": "Data Structures and Algorithms",
}

# Optional global SentenceTransformer cache
_EMBEDDING_MODEL = None
_SENTENCE_TRANSFORMERS_CHECKED = False


def get_sentence_transformer():
    """Lazily load SentenceTransformer model if installed, otherwise return None."""
    global _EMBEDDING_MODEL, _SENTENCE_TRANSFORMERS_CHECKED
    if _SENTENCE_TRANSFORMERS_CHECKED:
        return _EMBEDDING_MODEL

    _SENTENCE_TRANSFORMERS_CHECKED = True
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        _EMBEDDING_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("SentenceTransformer model loaded successfully.")
    except Exception as e:
        logger.info(f"SentenceTransformers not loaded: {e}. Using TF-IDF semantic vector fallback.")
        _EMBEDDING_MODEL = None
    return _EMBEDDING_MODEL


def canonicalize_skill(skill: str) -> str:
    """Normalize skill string using alias dictionary."""
    if not isinstance(skill, str):
        return ""
    cleaned = skill.strip().lower()
    
    # Check exact alias
    if cleaned in SKILL_ALIASES:
        return SKILL_ALIASES[cleaned]
    
    # Case-insensitive title lookup
    for alias, canonical in SKILL_ALIASES.items():
        if alias.lower() == cleaned:
            return canonical
        
    return skill.strip()


def calculate_semantic_similarity(
    source_skill: str, 
    target_skill: str, 
    use_neural: bool = True
) -> float:
    """
    Calculate semantic similarity score between two skill terms.
    Uses SentenceTransformers if available, otherwise high-precision TF-IDF n-gram vectors.
    """
    s1 = canonicalize_skill(source_skill).lower()
    s2 = canonicalize_skill(target_skill).lower()

    if s1 == s2:
        return 1.0

    # Try neural embedding if requested
    if use_neural:
        st_model = get_sentence_transformer()
        if st_model is not None:
            try:
                emb1 = st_model.encode([source_skill])[0]
                emb2 = st_model.encode([target_skill])[0]
                cos_sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
                return max(0.0, min(1.0, cos_sim))
            except Exception as e:
                logger.warning(f"Error computing neural embedding similarity: {e}")

    # High-precision TF-IDF character + word n-gram cosine similarity
    try:
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform([s1, s2])
        sim = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.0


def analyze_skill_gaps(
    target_role: str,
    student_skills: List[str],
    similarity_threshold: float = 0.68,
    use_semantic: bool = True
) -> Dict[str, Any]:
    """
    Perform multi-tier matching between student skills and role requirements:
    Tier 1: Exact matches (canonical normalized)
    Tier 2: Alias dictionary matches
    Tier 3: Semantic similarity (embeddings or TF-IDF vectors)
    """
    core_requirements = get_all_target_skills(target_role)
    if not core_requirements:
        core_requirements = ["Python", "Git", "SQL", "Data Structures and Algorithms"]

    # Canonicalize student skills
    normalized_student_skills = [canonicalize_skill(s) for s in student_skills if s and s != "None yet, I'm starting fresh"]

    matched_requirements: List[Dict[str, Any]] = []
    gap_requirements: List[str] = []
    matched_req_names: Set[str] = set()

    for req in core_requirements:
        req_canonical = canonicalize_skill(req)
        best_match: Optional[str] = None
        best_score = 0.0
        match_type = "unmatched"

        # Tier 1 & 2: Direct or alias match
        for st_skill in normalized_student_skills:
            if st_skill.lower() == req_canonical.lower():
                best_match = st_skill
                best_score = 1.0
                match_type = "exact"
                break
        
        # Tier 3: Semantic vector similarity
        if not best_match and use_semantic and normalized_student_skills:
            for st_skill in normalized_student_skills:
                score = calculate_semantic_similarity(st_skill, req_canonical, use_neural=True)
                if score > best_score:
                    best_score = score
                    if score >= similarity_threshold:
                        best_match = st_skill
                        match_type = "semantic"

        if best_match and (best_score >= similarity_threshold or match_type == "exact"):
            matched_req_names.add(req_canonical)
            matched_requirements.append({
                "required_skill": req,
                "student_skill": best_match,
                "score": round(best_score, 2),
                "match_type": match_type
            })
        else:
            gap_requirements.append(req)

    # Calculate Skill Coverage %
    total_reqs = len(core_requirements)
    matched_count = len(matched_requirements)
    skill_coverage = round((matched_count / total_reqs) * 100) if total_reqs > 0 else 0

    st_model = get_sentence_transformer() if use_semantic else None
    matcher_engine = "SentenceTransformers (all-MiniLM-L6-v2)" if st_model is not None else "Exact + TF-IDF Vector Cosine Matching"

    return {
        "target_role": target_role,
        "total_requirements": total_reqs,
        "matched_count": matched_count,
        "gap_count": len(gap_requirements),
        "skill_coverage": skill_coverage,
        "coverage_explanation": (
            f"Calculated as {matched_count} matched out of {total_reqs} core employer-requested skills "
            f"({skill_coverage}%) based on sample job postings for {target_role}."
        ),
        "matched_skills": [m["required_skill"] for m in matched_requirements],
        "matched_details": matched_requirements,
        "gap_skills": gap_requirements,
        "student_skills_submitted": normalized_student_skills,
        "matcher_engine": matcher_engine,
        "similarity_threshold": similarity_threshold
    }
