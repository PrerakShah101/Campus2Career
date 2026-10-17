"""
Campus2Career AI - A Student-Friendly Career Roadmap Web App
Designed with pastel, playful, modern aesthetic inspired by study planners & Notion.
"""

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load local environment variables from .env if present
load_dotenv()

from src.data_loader import (
    ROLE_CONFIGS,
    SKILL_CATEGORIES,
    load_jobs_dataset,
    calculate_role_skill_frequencies,
    get_all_target_skills
)
from src.skill_matcher import analyze_skill_gaps
from src.llm_service import (
    get_personalized_recommendations,
    validate_gemma_endpoint,
    get_gemma_config
)
from src.analytics import create_skill_frequency_chart

# Page configuration
st.set_page_config(
    page_title="Campus2Career AI | Student Career Roadmap",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Pastel & Modern Study Planner CSS Theme
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* Color palette */
:root {
    --bg: #F7F3E8;
    --surface: #FFFFFF;
    --ink: #171717;
    --muted: #454545;
    --pink: #FF759D;
    --lime: #C9F45A;
    --purple: #7654F6;
    --border: #171717;
}

/* Base page */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Inter', Arial, sans-serif !important;
}

[data-testid="stHeader"] {
    background: var(--bg) !important;
}

.block-container {
    max-width: 1180px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
}

/* Readable text */
h1, h2, h3, h4, h5, h6,
p, label, li, span, div {
    text-shadow: none;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--ink) !important;
    font-family: 'Inter', Arial, sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: -0.035em;
}

p, li, label {
    color: var(--muted) !important;
}

/* Grid background */
[data-testid="stAppViewContainer"] .main {
    background-image:
        linear-gradient(#1717170D 1px, transparent 1px),
        linear-gradient(90deg, #1717170D 1px, transparent 1px);
    background-size: 22px 22px;
}

/* Cards and containers */
.c2c-card,
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1.5px solid var(--border) !important;
    border-radius: 4px !important;
    box-shadow: 3px 3px 0 var(--border) !important;
}

/* Keep the app's existing card classes, but sharpen their appearance */
.c2c-card {
    background: var(--surface) !important;
    color: var(--ink) !important;
}

/* Buttons: high-contrast text */
div.stButton > button,
div.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    background: var(--purple) !important;
    color: #FFFFFF !important;
    border: 2px solid var(--ink) !important;
    border-radius: 3px !important;
    font-weight: 700 !important;
    box-shadow: 3px 3px 0 var(--ink) !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

div.stButton > button *,
div.stDownloadButton > button *,
[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: var(--ink) !important;
    color: #FFFFFF !important;
    transform: translate(2px, 2px);
    box-shadow: 1px 1px 0 var(--ink) !important;
}

/* Inputs */
input, textarea,
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border-color: var(--ink) !important;
    border-radius: 3px !important;
}

/* Select menu and dropdown text */
[data-baseweb="select"] *,
[role="option"] {
    color: var(--ink) !important;
}

/* Selected controls */
input:checked + div {
    background-color: var(--purple) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 2px solid var(--ink) !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: var(--ink) !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: var(--ink) !important;
    font-weight: 700 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--ink) !important;
    border-bottom: 3px solid var(--purple) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1.5px solid var(--ink) !important;
    padding: 14px !important;
    box-shadow: 3px 3px 0 var(--ink) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricValue"] {
    color: var(--ink) !important;
}

/* Alerts and captions */
[data-testid="stAlert"] {
    border: 1.5px solid var(--ink) !important;
    border-radius: 3px !important;
}

.stCaption, [data-testid="stCaptionContainer"] {
    color: #454545 !important;
}

/* Code blocks should not spill beyond their containers */
[data-testid="stCode"],
[data-testid="stCodeBlock"] {
    max-width: 100% !important;
    overflow-x: auto !important;
}

/* Keep charts and images within the layout */
[data-testid="stImage"],
[data-testid="stPlotlyChart"] {
    max-width: 100% !important;
}

/* Mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        box-shadow: 2px 2px 0 var(--ink) !important;
    }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Initialize Session State
if "selected_role" not in st.session_state:
    st.session_state.selected_role = "Machine Learning Intern"
if "selected_skills" not in st.session_state:
    st.session_state.selected_skills = ["Python", "Git", "NumPy"]
if "weekly_hours" not in st.session_state:
    st.session_state.weekly_hours = 6
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "roadmap_data" not in st.session_state:
    st.session_state.roadmap_data = None


# Load Data (Cached by Streamlit)
@st.cache_data(show_spinner=False)
def get_cached_jobs_data():
    return load_jobs_dataset()

jobs_df, dataset_meta = get_cached_jobs_data()


# Top Navigation / Friendly Brand Header
col_header_1, col_header_2 = st.columns([3, 1])
with col_header_1:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.8rem;">🎓</span>
            <span style="font-size: 1.45rem; font-weight: 700; color: #2D2746; letter-spacing: -0.02em;">Campus2Career AI</span>
            <span class="c2c-badge badge-lavender">Student Companion</span>
        </div>
        """,
        unsafe_allow_html=True
    )
with col_header_2:
    with st.popover("⚙️ Settings & AI Status"):
        st.markdown("#### Inference & Matching Setup")
        current_key = (
            st.session_state.get("gemma_api_key")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GEMMA_API_KEY")
            or ""
        )
        current_model = os.getenv("GEMMA_MODEL") or "gemma-4-26b-a4b-it"
        st.caption(f"Provider: Google AI Studio · Model: `{current_model}`")
        if current_key:
            st.info("A Google AI Studio key is configured. Test access before generating a roadmap.")
        else:
            st.info("Google AI Studio is not configured. The app will use the clearly labeled rule-based fallback.")

        entered_key = st.text_input(
            "Google AI Studio API key",
            value=current_key,
            type="password",
            placeholder="Paste your Google AI Studio key",
            help="Create a key at aistudio.google.com/apikey. The key is kept in session state when entered here.",
            key="gemma_key_input",
        )

        save_col, test_col = st.columns(2)
        with save_col:
            if st.button("Save for this session", use_container_width=True):
                st.session_state.gemma_api_key = entered_key.strip()
                st.success("Google AI Studio key saved for this session only. It is not written to disk.")
        with test_col:
            if st.button("Test connection", use_container_width=True):
                with st.spinner("Testing Gemma connection..."):
                    ok, message = validate_gemma_endpoint(
                        override_key=entered_key.strip(),
                    )
                if ok:
                    st.success(message)
                else:
                    st.warning(message)

        similarity_threshold = st.slider(
            "Semantic Similarity Threshold",
            min_value=0.50,
            max_value=0.90,
            value=float(st.session_state.get("similarity_threshold", 0.68)),
            step=0.02,
            help="Threshold for vector similarity when matching student skills to job expectations.",
        )
        st.session_state.similarity_threshold = similarity_threshold

# SECTION A: Welcome Section (Friendly & Compact)
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
welcome_col1, welcome_col2 = st.columns([2.8, 1.2])

with welcome_col1:
    st.markdown(
        """
        <div class="c2c-card" style="background: linear-gradient(135deg, #FFFFFF 0%, #FAF7FF 100%);">
            <span class="c2c-badge badge-peach" style="margin-bottom: 0.6rem;">Designed for Students</span>
            <h1 style="font-size: 2.1rem; line-height: 1.25; margin-bottom: 0.4rem;">
                Your next career move, made clearer.
            </h1>
            <p style="font-size: 1.05rem; color: #525062; margin-bottom: 1.1rem;">
                Pick your dream role, tell us what you know, and discover what to work on next. 
                Simple, actionable, and built to take you from coursework to portfolio-ready.
            </p>
            <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
                <a href="#career-setup" style="text-decoration: none;">
                    <div style="background-color: #4E4376; color: #FFF; font-weight: 600; padding: 0.65rem 1.4rem; border-radius: 14px; display: inline-flex; align-items: center; gap: 8px;">
                        <span>Build my roadmap</span> <span>↓</span>
                    </div>
                </a>
                <span style="font-size: 0.88rem; color: #757388;">
                    🌱 4 roles supported &bull; 4-week roadmap &bull; Zero fluff
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with welcome_col2:
    # Friendly stationery-style progress element
    st.markdown(
        """
        <div class="c2c-card" style="background-color: #FFFDF9; text-align: center; padding: 1.2rem;">
            <div style="font-size: 1.8rem; margin-bottom: 4px;">🗺️</div>
            <div style="font-weight: 600; font-size: 0.96rem; color: #343344; margin-bottom: 4px;">Student Study Plan</div>
            <div style="font-size: 0.82rem; color: #6D6B80; margin-bottom: 12px;">Step-by-step career path</div>
            <div style="background: #F4EFEB; border-radius: 12px; padding: 8px 12px; text-align: left; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #2E7D32; font-weight: 600;">✓ Step 1:</span> Choose Role
            </div>
            <div style="background: #F4EFEB; border-radius: 12px; padding: 8px 12px; text-align: left; font-size: 0.82rem; margin-bottom: 6px;">
                <span style="color: #2E7D32; font-weight: 600;">✓ Step 2:</span> Select Skills
            </div>
            <div style="background: #E9E2FF; border-radius: 12px; padding: 8px 12px; text-align: left; font-size: 0.82rem; font-weight: 600; color: #4B3A80;">
                <span>✦ Step 3:</span> Your Roadmap
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Anchor for smooth scroll
st.markdown("<div id='career-setup'></div>", unsafe_allow_html=True)


# SECTION B: Career Setup Form
st.markdown("### Career Setup")
st.markdown("<p style='margin-top: -8px; color: #666479;'>Tell us what career direction you're exploring and what you have learned so far.</p>", unsafe_allow_html=True)

# Step 1: Target Role Cards
st.markdown("##### Step 1: Choose your target role")
role_cols = st.columns(4)
role_keys = list(ROLE_CONFIGS.keys())

for idx, role_name in enumerate(role_keys):
    cfg = ROLE_CONFIGS[role_name]
    is_selected = (st.session_state.selected_role == role_name)
    with role_cols[idx]:
        card_html = f"""
        <div class="role-card-container {'selected' if is_selected else ''}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                <span style="font-size: 1.6rem;">{cfg['icon']}</span>
                {'<span class="c2c-badge badge-lavender" style="font-size: 0.72rem;">Selected</span>' if is_selected else ''}
            </div>
            <div style="font-weight: 600; font-size: 0.98rem; color: #2D2746; margin-bottom: 4px;">{cfg['title']}</div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        if st.button(f"Select {cfg['title']}", key=f"btn_role_{idx}", use_container_width=True):
            st.session_state.selected_role = role_name
            st.rerun()

st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)


# Step 2: Pick Your Current Skills
st.markdown("##### Step 2: Pick your current skills")
st.markdown("<p style='font-size: 0.9rem; color: #727084; margin-top: -6px;'>Check any skills you've encountered in coursework, personal exploration, or past projects.</p>", unsafe_allow_html=True)

fresh_start = st.checkbox("🌱 None yet, I'm starting completely fresh", value=("None yet, I'm starting fresh" in st.session_state.selected_skills))

selected_skills_list = []
if fresh_start:
    selected_skills_list = ["None yet, I'm starting fresh"]
    st.info("No worries at all! Campus2Career will construct a foundational roadmap starting from day one basics.")
else:
    # Render categories in a clean 4-column layout
    cat_cols = st.columns(4)
    cat_names = list(SKILL_CATEGORIES.keys())
    
    for i, cat_name in enumerate(cat_names):
        with cat_cols[i]:
            badge_class = ["badge-lavender", "badge-mint", "badge-peach", "badge-yellow"][i % 4]
            st.markdown(f"<span class='c2c-badge {badge_class}' style='margin-bottom: 10px;'>{cat_name}</span>", unsafe_allow_html=True)
            for skill in SKILL_CATEGORIES[cat_name]:
                # Default checked state from session
                is_checked = (skill in st.session_state.selected_skills and "None yet, I'm starting fresh" not in st.session_state.selected_skills)
                checked = st.checkbox(skill, value=is_checked, key=f"chk_{skill}")
                if checked:
                    selected_skills_list.append(skill)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# Study hours & Action Row
hours_col1, hours_col2 = st.columns([1.5, 2.5])
with hours_col1:
    study_hours = st.slider(
        "Weekly study time available",
        min_value=2,
        max_value=20,
        value=st.session_state.weekly_hours,
        step=1,
        help="How many hours per week can you comfortably dedicate to career learning?",
        format="%d hrs / week"
    )
    st.session_state.weekly_hours = study_hours

with hours_col2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button("✨ Analyze my skills & build roadmap", use_container_width=True)

if analyze_clicked:
    st.session_state.selected_skills = selected_skills_list
    with st.spinner("Analyzing employer expectations from sample job postings..."):
        sim_thresh = getattr(st.session_state, "similarity_threshold", 0.68)
        
        # Skill gap analysis
        gap_res = analyze_skill_gaps(
            target_role=st.session_state.selected_role,
            student_skills=selected_skills_list,
            similarity_threshold=sim_thresh,
            use_semantic=os.getenv("USE_SEMANTIC_EMBEDDINGS", "false").strip().lower() in {"1", "true", "yes"}
        )
        
        # Personalized Roadmap & Portfolio Projects
        roadmap_res = get_personalized_recommendations(
            target_role=st.session_state.selected_role,
            matched_skills=gap_res["matched_skills"],
            gap_skills=gap_res["gap_skills"],
            weekly_hours=study_hours,
            override_key=st.session_state.get("gemma_api_key"),
        )
        
        st.session_state.analysis_data = gap_res
        st.session_state.roadmap_data = roadmap_res
        st.session_state.analyzed = True
    st.rerun()


# SECTION C: Career Results Dashboard
if st.session_state.analyzed and st.session_state.analysis_data and st.session_state.roadmap_data:
    st.markdown("<hr style='border: none; border-top: 1px solid #ECE7DE; margin: 2.5rem 0;'>", unsafe_allow_html=True)
    st.markdown(f"## Your Career Roadmap: {st.session_state.selected_role}")
    
    analysis = st.session_state.analysis_data
    roadmap = st.session_state.roadmap_data
    
    # 1. Your Skill Snapshot
    st.markdown("#### 1. Your Skill Snapshot")
    
    # Compute postings count
    role_postings_df = jobs_df[jobs_df["target_role"] == st.session_state.selected_role] if not jobs_df.empty else pd.DataFrame()
    num_postings = len(role_postings_df)
    
    snap_col1, snap_col2, snap_col3, snap_col4 = st.columns(4)
    with snap_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #4E4376;">{analysis['skill_coverage']}%</div>
                <div class="metric-label">Skill Coverage</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with snap_col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #2E7D32;">{analysis['matched_count']}</div>
                <div class="metric-label">Requirements Matched</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with snap_col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #D97706;">{analysis['gap_count']}</div>
                <div class="metric-label">Skills to Develop</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with snap_col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-value" style="color: #4A495A;">{num_postings}</div>
                <div class="metric-label">Sample Jobs Analyzed</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        f"""
        <div class="c2c-alert-note">
            💡 <b>How Skill Coverage is calculated:</b> {analysis['coverage_explanation']}
            <br><span style="font-size: 0.82rem; color: #6D6B80;">Note: This metric reflects curriculum alignment with sample postings, not a test of verified proficiency.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)


    # 2. Skills Employers Ask For (Plotly Chart)
    chart_col1, chart_col2 = st.columns([2.8, 1.2])
    with chart_col1:
        st.markdown("#### 2. Skills Employers Ask For")
        freq_data = calculate_role_skill_frequencies(jobs_df, st.session_state.selected_role, top_n=8)
        source_note = dataset_meta.get("source", "Curated Offline Demonstration Sample")
        if dataset_meta.get("is_fallback"):
            source_note = "Curated Offline Demonstration Sample"
        
        chart_fig = create_skill_frequency_chart(
            frequency_data=freq_data,
            target_role=st.session_state.selected_role,
            total_postings=num_postings,
            source_label=source_note
        )
        st.plotly_chart(chart_fig, use_container_width=True)
        st.caption(
            f"📊 Based on an analysis of {num_postings} sample postings from {source_note}. "
            "Data represents observations from this dataset, not definitive industry-wide market statistics."
        )

    with chart_col2:
        st.markdown("#### About This Dataset")
        st.markdown(
            f"""
            <div class="c2c-card" style="font-size: 0.85rem; padding: 1.1rem;">
                <div style="font-weight: 600; color: #2D2746; margin-bottom: 6px;">Data Transparency</div>
                <p style="font-size: 0.84rem; color: #5B596E; margin-bottom: 8px;">
                    Postings are filtered and normalized from the configured source dataset. See the README for provenance and limitations.
                </p>
                <div style="background-color: #F8F5FF; border-radius: 8px; padding: 6px 10px; font-size: 0.8rem; margin-bottom: 6px;">
                    <b>Sample Pool:</b> {len(jobs_df)} clean postings
                </div>
                <div style="background-color: #DDF3E4; border-radius: 8px; padding: 6px 10px; font-size: 0.8rem;">
                    <b>Matching Engine:</b> {analysis['matcher_engine']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)


    # 3. Your Skill Gaps (Two Visually Distinct Groups)
    st.markdown("#### 3. Your Skill Gaps")
    gaps_col1, gaps_col2 = st.columns(2)

    with gaps_col1:
        st.markdown(
            """
            <div class="c2c-card" style="border-top: 4px solid #7ED6A5;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 1.05rem; color: #1E5C32;">✓ Already on your list</span>
                    <span class="c2c-badge badge-mint">In your toolkit</span>
                </div>
                <p style="font-size: 0.88rem; color: #5A586B; margin-bottom: 12px;">Skills you selected that match requirements for this role:</p>
            """,
            unsafe_allow_html=True
        )
        if analysis["matched_skills"]:
            pills_html = " ".join([f"<span class='c2c-badge badge-mint' style='margin-right: 6px; margin-bottom: 8px; font-size: 0.88rem;'>{s}</span>" for s in analysis["matched_skills"]])
            st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>{pills_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 0.88rem; color: #7B798D;'>No direct matches yet — you are starting with a clean slate!</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with gaps_col2:
        st.markdown(
            """
            <div class="c2c-card" style="border-top: 4px solid #B8A7EA;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 1.05rem; color: #4B3A80;">✦ Worth learning next</span>
                    <span class="c2c-badge badge-lavender">Recommended focus</span>
                </div>
                <p style="font-size: 0.88rem; color: #5A586B; margin-bottom: 12px;">Important role requirements to prioritize in your upcoming projects:</p>
            """,
            unsafe_allow_html=True
        )
        if analysis["gap_skills"]:
            pills_html = " ".join([f"<span class='c2c-badge badge-lavender' style='margin-right: 6px; margin-bottom: 8px; font-size: 0.88rem;'>{s}</span>" for s in analysis["gap_skills"]])
            st.markdown(f"<div style='display: flex; flex-wrap: wrap;'>{pills_html}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size: 0.88rem; color: #2E7D32;'>Great work! You already cover all core requirements evaluated for this role.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption("ℹ️ Note: Having a skill on your list indicates coursework or familiarity, not verified workplace proficiency.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)


    # 4. Your Personalized Roadmap (Four-Week Plan)
    st.markdown("#### 4. Your Personalized Four-Week Roadmap")
    
    badge_style = "badge-lavender" if roadmap.get("is_ai_generated") else "badge-peach"
    st.markdown(
        f"""
        <div style="margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">
            <span class="c2c-badge {badge_style}">{roadmap.get('engine_badge', 'Roadmap Engine')}</span>
            <span style="font-size: 0.84rem; color: #6E6C82;">{roadmap.get('notes', '')}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    weeks = roadmap.get("roadmap", [])
    for idx, wk in enumerate(weeks):
        week_num = wk.get("week", idx + 1)
        card_class = f"week-card week-{week_num}"
        objectives_html = "".join([f"<li style='margin-bottom: 2px;'>{obj}</li>" for obj in wk.get("objectives", [])])
        
        st.markdown(
            f"""
            <div class="{card_class}">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                    <div style="font-weight: 700; font-size: 1.08rem; color: #292838;">{wk.get('title', f'Week {week_num}')}</div>
                    <span class="c2c-badge badge-lavender" style="font-size: 0.76rem;">Week {week_num} &bull; {st.session_state.weekly_hours}h</span>
                </div>
                <div style="margin-bottom: 8px;">
                    <div style="font-weight: 600; font-size: 0.86rem; color: #4E4376; margin-bottom: 2px;">Learning Objectives:</div>
                    <ul style="padding-left: 1.2rem; margin-top: 0; margin-bottom: 6px;">{objectives_html}</ul>
                </div>
                <div style="background-color: #FAF8F5; border-radius: 10px; padding: 10px 14px; margin-bottom: 6px;">
                    <span style="font-weight: 600; font-size: 0.86rem; color: #2E7D32;">🛠️ Practical Task:</span>
                    <span style="font-size: 0.88rem; color: #434255;"> {wk.get('practical_task', '')}</span>
                </div>
                <div style="font-size: 0.84rem; color: #585669;">
                    <b>Completion Outcome:</b> {wk.get('completion_outcome', '')}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)


    # 5. Build Something Real (Two Portfolio Projects)
    st.markdown("#### 5. Build Something Real")
    st.markdown("<p style='font-size: 0.92rem; color: #69677C; margin-top: -6px;'>Two realistic, non-generic portfolio projects tailored to your target role and skill gaps.</p>", unsafe_allow_html=True)

    projects = roadmap.get("portfolio_projects", [])
    proj_col1, proj_col2 = st.columns(2)

    for idx, proj in enumerate(projects[:2]):
        target_col = proj_col1 if idx == 0 else proj_col2
        with target_col:
            skills_pills = " ".join([f"<span class='c2c-badge badge-peach' style='margin-right: 4px; margin-bottom: 4px; font-size: 0.78rem;'>{s}</span>" for s in proj.get("skills_demonstrated", [])])
            st.markdown(
                f"""
                <div class="project-card">
                    <div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                            <span class="c2c-badge badge-lavender" style="font-size: 0.76rem;">Project #{idx+1}</span>
                        </div>
                        <h4 style="font-size: 1.12rem; margin-bottom: 6px; color: #2D2746;">{proj.get('title', 'Portfolio Project')}</h4>
                        <p style="font-size: 0.88rem; color: #514F63; margin-bottom: 12px; line-height: 1.5;">{proj.get('what_it_does', '')}</p>
                        <div style="margin-bottom: 12px;">
                            <div style="font-size: 0.8rem; font-weight: 600; color: #6E6C82; margin-bottom: 4px;">Skills it demonstrates:</div>
                            <div style="display: flex; flex-wrap: wrap;">{skills_pills}</div>
                        </div>
                    </div>
                    <div style="background-color: #FAF7F2; border-left: 3px solid #7ED6A5; border-radius: 8px; padding: 10px 12px; margin-top: 10px;">
                        <div style="font-size: 0.8rem; font-weight: 600; color: #1E5C32; margin-bottom: 2px;">🏁 Small First Milestone:</div>
                        <div style="font-size: 0.85rem; color: #434154;">{proj.get('first_milestone', '')}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("<div style='height: 32px;'></div>", unsafe_allow_html=True)


    # 6. Start Over Action
    start_over_col1, start_over_col2, start_over_col3 = st.columns([1.5, 2, 1.5])
    with start_over_col2:
        if st.button("🔄 Start over with another career or skills", use_container_width=True):
            st.session_state.analyzed = False
            st.session_state.analysis_data = None
            st.session_state.roadmap_data = None
            st.rerun()


# Footer
st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align: center; border-top: 1px solid #ECE7DE; padding-top: 1.5rem; color: #8F8D9E; font-size: 0.85rem;">
        <b>Campus2Career AI</b> &bull; A student-friendly career roadmap companion.<br>
        Built with Streamlit, Plotly, Scikit-learn, and Hugging Face Dataset &bull; Warm &bull; Playful &bull; Honest
    </div>
    """,
    unsafe_allow_html=True
)
