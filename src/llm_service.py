"""
LLM Service Module for Campus2Career AI
Integrates with Google AI Studio's Gemma 4 generateContent API and provides a
pedagogical rule-based fallback when Gemma is unconfigured or unavailable.
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
import requests

logger = logging.getLogger("llm_service")


def get_gemma_config(override_key: Optional[str] = None, override_url: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Read Google AI Studio settings for hosted Gemma inference.

    GOOGLE_API_KEY is preferred. GEMMA_API_KEY remains supported for compatibility
    with existing local .env files, but its value must be a Google AI Studio key.
    override_url is accepted for backwards compatibility and intentionally ignored:
    Google AI Studio uses the official model-specific generateContent endpoint.
    """
    api_key = (
        override_key
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GEMMA_API_KEY")
        or ""
    ).strip()
    model = (os.getenv("GEMMA_MODEL") or "gemma-4-26b-a4b-it").strip()
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    return {"api_key": api_key or None, "endpoint": endpoint, "model": model}


def validate_gemma_endpoint(
    override_key: Optional[str] = None,
    override_url: Optional[str] = None,
) -> Tuple[bool, str]:
    """Send a small real request to verify Google AI Studio Gemma access."""
    config = get_gemma_config(override_key=override_key)
    if not config["api_key"]:
        return False, "No Google AI Studio API key configured. Add GOOGLE_API_KEY to .env."

    try:
        response = requests.post(
            config["endpoint"],
            headers={
                "x-goog-api-key": config["api_key"],
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": "Reply with the word ready."}]}],
                "generationConfig": {"maxOutputTokens": 8, "temperature": 0},
            },
            timeout=12,
        )
        if response.ok:
            return True, f"Connected to Google AI Studio model {config['model']}"
        if response.status_code in (401, 403):
            return False, "Google API key rejected or this model is unavailable to the project."
        if response.status_code == 404:
            return False, f"Model not found or unavailable: {config['model']}. Check GEMMA_MODEL."
        if response.status_code == 429:
            return False, "Google AI Studio rate-limited this request. Check quota and try again."
        return False, f"Google AI Studio returned HTTP {response.status_code}: {response.text[:180]}"
    except requests.RequestException as exc:
        return False, f"Could not reach Google AI Studio: {exc}"


def _validate_recommendation_payload(payload: Any) -> Optional[Dict[str, Any]]:
    """Reject malformed model output before it reaches the UI."""
    if not isinstance(payload, dict):
        return None
    roadmap = payload.get("roadmap")
    projects = payload.get("portfolio_projects")
    if not isinstance(roadmap, list) or len(roadmap) != 4:
        return None
    if not isinstance(projects, list) or len(projects) != 2:
        return None

    cleaned_weeks = []
    for index, week in enumerate(roadmap, start=1):
        if not isinstance(week, dict):
            return None
        objectives = week.get("objectives")
        if not isinstance(objectives, list) or not objectives:
            return None
        required_text = ("title", "practical_task", "completion_outcome")
        if any(not isinstance(week.get(key), str) or not week[key].strip() for key in required_text):
            return None
        cleaned_weeks.append({
            "week": index,
            "title": week["title"].strip(),
            "objectives": [str(item).strip() for item in objectives if str(item).strip()][:3],
            "practical_task": week["practical_task"].strip(),
            "completion_outcome": week["completion_outcome"].strip(),
        })
        if not cleaned_weeks[-1]["objectives"]:
            return None

    cleaned_projects = []
    for project in projects:
        if not isinstance(project, dict):
            return None
        required_text = ("title", "what_it_does", "first_milestone")
        if any(not isinstance(project.get(key), str) or not project[key].strip() for key in required_text):
            return None
        skills = project.get("skills_demonstrated")
        if not isinstance(skills, list):
            return None
        cleaned_projects.append({
            "title": project["title"].strip(),
            "what_it_does": project["what_it_does"].strip(),
            "skills_demonstrated": [str(item).strip() for item in skills if str(item).strip()][:5],
            "first_milestone": project["first_milestone"].strip(),
        })

    return {"roadmap": cleaned_weeks, "portfolio_projects": cleaned_projects}


def call_gemma_inference(
    target_role: str,
    matched_skills: List[str],
    gap_skills: List[str],
    weekly_hours: int,
    override_key: Optional[str] = None,
    override_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Call Gemma through Google's Gemini API and validate its JSON output."""
    config = get_gemma_config(override_key=override_key)
    if not config["api_key"]:
        return None

    prompt = f"""You are Campus2Career, a practical career planner for college students.
Create a realistic, specific career plan. Return ONLY valid JSON, with no markdown fences.
Target role: {target_role}
Skills already selected by the student: {", ".join(matched_skills) if matched_skills else "None yet"}
Skills to prioritize: {", ".join(gap_skills) if gap_skills else "Review and deepen existing skills"}
Study availability: {weekly_hours} hours per week.
Fit the workload to the stated weekly hours. Prioritize the missing skills. Do not make unsupported job-market claims.

Return an object with exactly these top-level keys:
{{
  "roadmap": [
    {{"week": 1, "title": "...", "objectives": ["..."], "practical_task": "...", "completion_outcome": "..."}},
    {{"week": 2, "title": "...", "objectives": ["..."], "practical_task": "...", "completion_outcome": "..."}},
    {{"week": 3, "title": "...", "objectives": ["..."], "practical_task": "...", "completion_outcome": "..."}},
    {{"week": 4, "title": "...", "objectives": ["..."], "practical_task": "...", "completion_outcome": "..."}}
  ],
  "portfolio_projects": [
    {{"title": "...", "what_it_does": "...", "skills_demonstrated": ["..."], "first_milestone": "..."}},
    {{"title": "...", "what_it_does": "...", "skills_demonstrated": ["..."], "first_milestone": "..."}}
  ]
}}"""

    try:
        response = requests.post(
            config["endpoint"],
            headers={
                "x-goog-api-key": config["api_key"],
                "Content-Type": "application/json",
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.3,
                    "maxOutputTokens": 2200,
                },
            },
            timeout=45,
        )
        if not response.ok:
            logger.warning(
                "Google AI Studio returned HTTP %s: %s",
                response.status_code,
                response.text[:300],
            )
            return None
        body = response.json()
        parts = body["candidates"][0]["content"]["parts"]
        content = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
        if not content.strip():
            return None
        content = content.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\\s*|\\s*```$", "", content, flags=re.IGNORECASE)
        parsed = json.loads(content)
        return _validate_recommendation_payload(parsed)
    except (requests.RequestException, ValueError, KeyError, TypeError, IndexError) as exc:
        logger.warning("Google Gemma inference failed or returned invalid JSON: %s", exc)
        return None


def generate_rule_based_roadmap(
    target_role: str,
    matched_skills: List[str],
    gap_skills: List[str],
    weekly_hours: int
) -> Dict[str, Any]:
    """
    High-quality, pedagogical rule-based career roadmap generator.
    Customizes weekly milestones and portfolio projects according to
    the student's target role, matched skills, missing skills, and study hours.
    """
    primary_focus = gap_skills[:2] if gap_skills else ["Advanced System Architecture", "Production Deployment"]
    secondary_focus = gap_skills[2:4] if len(gap_skills) > 2 else ["Portfolio Polish", "Technical Interview Prep"]

    # Role-specific project templates
    role_project_templates = {
        "Machine Learning Intern": [
            {
                "title": "Campus Sentiment & Course Review Classifier",
                "what_it_does": "An end-to-end NLP pipeline that ingests student feedback and course reviews, categorizes topics, and predicts sentiment ratings using Scikit-learn and PyTorch.",
                "skills_demonstrated": [s for s in ["Python", "Machine Learning", "Scikit-learn", "Pandas"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Scrape or assemble a 200-row CSV of student reviews and train a simple baseline Naive Bayes or Logistic Regression model in a Jupyter Notebook."
            },
            {
                "title": "Real-Time Campus Foot-Traffic Vision Monitor",
                "what_it_does": "A computer vision model that analyzes cafeteria or library video feeds to count open study desks and compute peak occupancy trends.",
                "skills_demonstrated": [s for s in ["Python", "Deep Learning", "NumPy", "Data Visualization"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Download an open webcam dataset, annotate 50 frames, and run a pretrained YOLO or OpenCV desk detector."
            }
        ],
        "Data Scientist": [
            {
                "title": "Student Loan & Tuition ROI Predictive Engine",
                "what_it_does": "A predictive machine learning service that analyzes historical university tuition, major choices, and median post-grad earnings to estimate 10-year loan repayment timelines.",
                "skills_demonstrated": [s for s in ["Python", "SQL", "Statistics", "Machine Learning"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Query US College Scorecard data using SQL, clean nulls in Pandas, and run exploratory correlation plots."
            },
            {
                "title": "A/B Testing Simulator for EdTech Landing Pages",
                "what_it_does": "A statistical simulation tool that generates synthetic user conversion funnels, computes p-values, and performs sequential hypothesis testing for growth teams.",
                "skills_demonstrated": [s for s in ["Statistics", "Python", "Data Visualization", "NumPy"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Write a Python script that simulates 1,000 Bernoulli trials for control vs. treatment and plots statistical power curves."
            }
        ],
        "Data Analyst": [
            {
                "title": "University Internship Placement & Salary Dashboard",
                "what_it_does": "An interactive business intelligence reporting suite analyzing college placement rates, company hiring hotspots, and compensation bands across graduation years.",
                "skills_demonstrated": [s for s in ["SQL", "Data Visualization", "Pandas", "Statistics"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Design a 3-table normalized SQL schema for Students, Applications, and Offers; write 5 analytical aggregations with window functions."
            },
            {
                "title": "Campus Dining Hall Food Waste Analytics",
                "what_it_does": "An operational analytics report that tracks daily kitchen food prep vs. student waste, uncovering cost-saving opportunities through weekly trend analysis.",
                "skills_demonstrated": [s for s in ["SQL", "Data Visualization", "Statistics", "Python"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Synthesize a sample dining hall dataset and build 4 clean horizontal summary charts highlighting peak waste weekdays."
            }
        ],
        "Software Engineer": [
            {
                "title": "Peer-to-Peer Study Group Matching Microservice",
                "what_it_does": "A containerized REST API backend that pairs students based on courses, available study hours, and skill levels, complete with authentication and automated tests.",
                "skills_demonstrated": [s for s in ["APIs", "Git", "SQL", "Data Structures and Algorithms"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Implement 3 REST endpoints (`/students`, `/courses`, `/match`) using FastAPI or Express with an in-memory test suite."
            },
            {
                "title": "Distributed Task Scheduler & Job Queue",
                "what_it_does": "A lightweight task queue that executes background jobs with exponential backoff retries, priority ordering, and worker status monitoring.",
                "skills_demonstrated": [s for s in ["Data Structures and Algorithms", "APIs", "Git", "Python"] if s in gap_skills or s in matched_skills][:3],
                "first_milestone": "Build a thread-safe Min-Heap priority queue in Python or Java that pops tasks based on scheduled execution timestamps."
            }
        ]
    }

    # Format 4 weeks
    week_1_skill = primary_focus[0] if primary_focus else "Core Foundations"
    week_2_skill = primary_focus[1] if len(primary_focus) > 1 else (secondary_focus[0] if secondary_focus else "Data Processing")
    week_3_skill = secondary_focus[0] if secondary_focus else "Integration & Testing"
    week_4_skill = "Project Polish & Career Launch"

    hours_note = f"Paced for {weekly_hours} hours/week of focused study."

    roadmap = [
        {
            "week": 1,
            "title": f"Week 1: Core Foundations in {week_1_skill}",
            "objectives": [
                f"Master core syntax, data models, and fundamental patterns of {week_1_skill}.",
                f"Set up a clean GitHub repository and modern development environment ({hours_note})."
            ],
            "practical_task": f"Build 3 small practical scripts using {week_1_skill} addressing real-world problem scenarios.",
            "completion_outcome": f"Push your verified code and README documentation to GitHub."
        },
        {
            "week": 2,
            "title": f"Week 2: Deep Dive into {week_2_skill}",
            "objectives": [
                f"Apply {week_2_skill} to structured problem-solving and algorithmic data workflows.",
                "Bridge your existing toolkit with newly learned methodologies."
            ],
            "practical_task": f"Implement a standalone mini-pipeline integrating {week_1_skill} and {week_2_skill}.",
            "completion_outcome": "Working pipeline script with unit tests demonstrating edge case handling."
        },
        {
            "week": 3,
            "title": f"Week 3: Practical Integration & Portfolio Milestone",
            "objectives": [
                f"Integrate {week_3_skill} into your first portfolio project.",
                "Refactor code for readability, performance, and clean documentation."
            ],
            "practical_task": "Build the first functional prototype of your chosen portfolio project.",
            "completion_outcome": "Runnable prototype with sample input data and clear usage instructions."
        },
        {
            "week": 4,
            "title": "Week 4: Real-World Polish & Showcase",
            "objectives": [
                "Deploy your project to a public URL or write an engaging engineering case study.",
                f"Prepare 3 concise talking points for interviews regarding how you solved {target_role} problems."
            ],
            "practical_task": "Record a 60-second video demo or write a compelling GitHub README with architecture diagrams.",
            "completion_outcome": "Complete portfolio-ready project linked on your resume and LinkedIn."
        }
    ]

    selected_projects = role_project_templates.get(target_role, role_project_templates["Software Engineer"])

    return {
        "roadmap": roadmap,
        "portfolio_projects": selected_projects
    }


def get_personalized_recommendations(
    target_role: str,
    matched_skills: List[str],
    gap_skills: List[str],
    weekly_hours: int = 6,
    override_key: Optional[str] = None,
    override_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Try one live Gemma 4 generation, then fall back transparently."""
    config = get_gemma_config(override_key=override_key, override_url=override_url)
    if config["api_key"]:
        llm_result = call_gemma_inference(
            target_role,
            matched_skills,
            gap_skills,
            weekly_hours,
            override_key=override_key,
            override_url=override_url,
        )
        if llm_result:
            return {
                **llm_result,
                "is_ai_generated": True,
                "engine_name": f"Google AI Studio Gemma ({config['model']})",
                "engine_badge": "AI-generated via Google AI Studio",
                "notes": f"Generated using Gemma model {config['model']} through Google AI Studio.",
            }
        fallback_note = "Google AI Studio was configured but the request failed or returned invalid output."
    else:
        fallback_note = "No Google AI Studio API key configured."

    fallback_result = generate_rule_based_roadmap(target_role, matched_skills, gap_skills, weekly_hours)
    return {
        "roadmap": fallback_result["roadmap"],
        "portfolio_projects": fallback_result["portfolio_projects"],
        "is_ai_generated": False,
        "engine_name": "Rule-Based Career Guidance Engine",
        "engine_badge": "Curated Rule-Based Guidance",
        "notes": f"Generated using deterministic rules. {fallback_note}",
    }
