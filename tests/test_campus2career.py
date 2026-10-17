"""
Unit and Integration Tests for Campus2Career AI
Validates data loading, role distributions, skill matching, LLM fallback, and chart generation.
"""

import pytest
import pandas as pd
from src.data_loader import (
    load_jobs_dataset,
    calculate_role_skill_frequencies,
    get_all_target_skills,
    ROLE_CONFIGS,
    SKILL_CATEGORIES
)
from src.skill_matcher import (
    canonicalize_skill,
    calculate_semantic_similarity,
    analyze_skill_gaps
)
from src.llm_service import (
    validate_gemma_endpoint,
    generate_rule_based_roadmap,
    get_personalized_recommendations
)
from src.analytics import create_skill_frequency_chart


def test_data_loader_returns_valid_dataset():
    """Ensure dataset loads and contains required columns and roles."""
    df, meta = load_jobs_dataset()
    assert not df.empty, "Dataset should not be empty"
    required_cols = ["job_title", "target_role", "company", "description", "skills", "location"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Check that all 4 target roles exist
    present_roles = set(df["target_role"].unique())
    for role in ROLE_CONFIGS.keys():
        assert role in present_roles, f"Role '{role}' should be present in dataset"


def test_skill_frequencies_calculation():
    """Ensure skill frequency calculation returns structured results."""
    df, _ = load_jobs_dataset()
    freq = calculate_role_skill_frequencies(df, "Software Engineer", top_n=5)
    assert len(freq) > 0
    first = freq[0]
    assert "skill" in first
    assert "count" in first
    assert "percentage" in first
    assert 0 <= first["percentage"] <= 100


def test_skill_aliases_normalization():
    """Ensure common skill aliases resolve to canonical names."""
    assert canonicalize_skill("sklearn") == "Scikit-learn"
    assert canonicalize_skill("py") == "Python"
    assert canonicalize_skill("cpp") == "C++"
    assert canonicalize_skill("dsa") == "Data Structures and Algorithms"
    assert canonicalize_skill("postgres") == "SQL"
    assert canonicalize_skill("neural networks") == "Deep Learning"


def test_skill_matcher_exact_and_gap_analysis():
    """Ensure accurate matching, coverage score calculation, and gap detection."""
    role = "Machine Learning Intern"
    student_skills = ["Python", "Git", "NumPy"]
    
    res = analyze_skill_gaps(target_role=role, student_skills=student_skills)
    
    assert res["target_role"] == role
    assert res["matched_count"] >= 3
    assert "Python" in res["matched_skills"]
    assert "Git" in res["matched_skills"]
    assert "NumPy" in res["matched_skills"]
    assert len(res["gap_skills"]) > 0
    assert 0 <= res["skill_coverage"] <= 100
    assert "coverage_explanation" in res


def test_fresh_start_handling():
    """Ensure students with no prior skills get a 0% coverage and full gap list."""
    role = "Data Analyst"
    res = analyze_skill_gaps(target_role=role, student_skills=["None yet, I'm starting fresh"])
    
    assert res["matched_count"] == 0
    assert res["skill_coverage"] == 0
    assert len(res["gap_skills"]) == len(get_all_target_skills(role))


def test_llm_service_fallback_roadmap():
    """Ensure rule-based fallback generates valid 4-week roadmap and 2 portfolio projects."""
    recs = get_personalized_recommendations(
        target_role="Software Engineer",
        matched_skills=["Python", "Git"],
        gap_skills=["APIs", "Data Structures and Algorithms"],
        weekly_hours=8
    )
    
    assert "roadmap" in recs
    assert "portfolio_projects" in recs
    assert len(recs["roadmap"]) == 4, "Roadmap should have exactly 4 weeks"
    
    for idx, wk in enumerate(recs["roadmap"]):
        assert "week" in wk
        assert "title" in wk
        assert "objectives" in wk
        assert len(wk["objectives"]) >= 1
        assert "practical_task" in wk
        assert "completion_outcome" in wk
        
    assert len(recs["portfolio_projects"]) == 2, "Should recommend exactly 2 projects"
    for proj in recs["portfolio_projects"]:
        assert "title" in proj
        assert "what_it_does" in proj
        assert "skills_demonstrated" in proj
        assert "first_milestone" in proj


def test_analytics_chart_generation():
    """Ensure Plotly figure is generated with valid structure."""
    dummy_freq = [
        {"skill": "Python", "count": 50, "percentage": 90.0},
        {"skill": "SQL", "count": 40, "percentage": 75.0}
    ]
    fig = create_skill_frequency_chart(dummy_freq, "Data Scientist", 55)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].orientation == "h"


def test_empty_role_frequency_does_not_invent_counts():
    """Missing role data must not be represented as fabricated observations."""
    empty = pd.DataFrame(columns=["job_title", "target_role", "company", "description", "skills", "location"])
    assert calculate_role_skill_frequencies(empty, "Data Scientist") == []


def test_gemma_config_defaults_to_google_ai_studio(monkeypatch):
    """The configured default should use Google's official Gemma generateContent endpoint."""
    from src.llm_service import get_gemma_config
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_API_URL", raising=False)
    monkeypatch.delenv("GEMMA_ENDPOINT", raising=False)
    monkeypatch.delenv("GEMMA_MODEL", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    config = get_gemma_config()
    assert config["model"] == "gemma-4-26b-a4b-it"
    assert config["endpoint"] == "https://generativelanguage.googleapis.com/v1beta/models/gemma-4-26b-a4b-it:generateContent"
    assert config["api_key"] == "test-google-key"


def test_gemma_payload_validation_rejects_malformed_response():
    from src.llm_service import _validate_recommendation_payload
    assert _validate_recommendation_payload({"roadmap": [], "portfolio_projects": []}) is None


def test_gemma_success_is_attributed_only_after_valid_response(monkeypatch):
    import json
    from src import llm_service

    roadmap = [
        {
            "week": i,
            "title": f"Week {i}",
            "objectives": ["Learn one relevant concept"],
            "practical_task": "Build a small exercise",
            "completion_outcome": "Commit the exercise",
        }
        for i in range(1, 5)
    ]
    projects = [
        {
            "title": f"Project {i}",
            "what_it_does": "A small role-relevant portfolio project.",
            "skills_demonstrated": ["Python"],
            "first_milestone": "Create the repository.",
        }
        for i in range(1, 3)
    ]

    class FakeResponse:
        ok = True
        status_code = 200
        text = ""

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": json.dumps({"roadmap": roadmap, "portfolio_projects": projects})}]}}]}

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMMA_API_KEY", raising=False)
    monkeypatch.setenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    monkeypatch.setattr(llm_service.requests, "post", lambda *args, **kwargs: FakeResponse())

    result = llm_service.get_personalized_recommendations(
        target_role="Data Analyst",
        matched_skills=["SQL"],
        gap_skills=["Statistics"],
        weekly_hours=6,
    )
    assert result["is_ai_generated"] is True
    assert result["engine_name"].startswith("Google AI Studio Gemma")
    assert len(result["roadmap"]) == 4
    assert len(result["portfolio_projects"]) == 2
