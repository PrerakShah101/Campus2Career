"""
Analytics and Visualization Module for Campus2Career AI
Creates polished, accessible, pastel-themed Plotly charts for skill demand analysis.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


# Friendly pastel color sequence for bars
PASTEL_COLORS = [
    "#B8A7EA",  # Soft lavender
    "#7ED6A5",  # Pastel mint
    "#FFB896",  # Pale peach
    "#FDE68A",  # Butter yellow
    "#A5D8FF",  # Soft sky
    "#D0BFFF",  # Light lilac
    "#B2F2BB",  # Light sage
    "#FFD8A8",  # Soft apricot
    "#E2D9F3",  # Muted lavender
    "#C3E6CB",  # Muted mint
]


def create_skill_frequency_chart(
    frequency_data: List[Dict[str, Any]],
    target_role: str,
    total_postings: int,
    source_label: str = "Hugging Face (MichaelYitzchak/LinkedInJobPostings)"
) -> go.Figure:
    """
    Generate a clean horizontal bar chart showing the top employer-requested skills.
    Adheres strictly to the pastel, student-friendly visual identity.
    """
    if not frequency_data:
        # Empty placeholder figure
        fig = go.Figure()
        fig.update_layout(
            title="No skill data available for this role.",
            paper_bgcolor="#FFFCF7",
            plot_bgcolor="#FFFCF7"
        )
        return fig

    # Sort ascending so highest appears at the top in horizontal layout
    sorted_data = sorted(frequency_data, key=lambda x: x["percentage"], reverse=False)

    skills = [item["skill"] for item in sorted_data]
    percentages = [item["percentage"] for item in sorted_data]
    counts = [item["count"] for item in sorted_data]

    # Cycle colors
    colors = [PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(skills))]

    # Text label for each bar
    text_labels = [f"<b>{pct}%</b> ({cnt} postings)" for pct, cnt in zip(percentages, counts)]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=skills,
        x=percentages,
        orientation="h",
        text=text_labels,
        textposition="outside",
        textfont=dict(
            family="Inter, -apple-system, sans-serif",
            size=12,
            color="#343344"
        ),
        marker=dict(
            color=colors,
            line=dict(color="#E2D9F3", width=1.5),
            cornerradius=8
        ),
        hovertemplate="<b>%{y}</b><br>Mentioned in: %{x}% of postings (%{customdata} jobs)<extra></extra>",
        customdata=counts
    ))

    max_pct = max(percentages) if percentages else 100
    x_range_max = min(100, max_pct + 18)

    fig.update_layout(
        paper_bgcolor="#FFFCF7",
        plot_bgcolor="#FFFCF7",
        margin=dict(l=10, r=40, t=20, b=30),
        height=380,
        xaxis=dict(
            title=dict(
                text="Percentage of Analyzed Postings (%)",
                font=dict(family="Inter, sans-serif", size=12, color="#6B7280")
            ),
            range=[0, x_range_max],
            showgrid=True,
            gridcolor="#F3EFE6",
            zeroline=False,
            tickfont=dict(family="Inter, sans-serif", size=11, color="#6B7280")
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family="Inter, sans-serif", size=13, color="#343344"),
            automargin=True
        ),
        showlegend=False
    )

    return fig
