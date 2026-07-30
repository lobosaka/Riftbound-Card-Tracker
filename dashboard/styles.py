import streamlit as st


RIFT_DARK = "#0B3041"
RIFT_PANEL = "#0E3A4F"
RIFT_PANEL_LIGHT = "#12485F"
RIFT_GOLD = "#FFC000"
RIFT_GOLD_DARK = "#D6A000"
RIFT_TEXT = "#FFFFFF"
RIFT_MUTED = "#AEAEAE"

def apply_riftbound_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #0B3041;
            color: #FFFFFF;
        }

        h1, h2, h3 {
            color: #FFFFFF;
        }

        h1 {
            border-bottom: 2px solid #FFC000;
            padding-bottom: 0.45rem;
        }

        [data-testid="stSidebar"] {
            background: #082634;
            border-right: 1px solid rgba(255, 192, 0, 0.45);
        }

        [data-testid="stMetric"] {
            background: #0E3A4F;
            border: 1px solid rgba(255, 192, 0, 0.55);
            border-radius: 14px;
            padding: 0.65rem 0.85rem;
            min-height: 105px;
        }

        [data-testid="stMetricLabel"] {
            color: #AEAEAE;
            font-size: 0.95rem;
        }

        [data-testid="stMetricValue"] {
            color: #FFC000;
            font-size: 2rem;
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid #FFC000;
            background: #FFC000;
            color: #0B3041;
            font-weight: 800;
        }

        .stButton > button:hover {
            border-color: #FFD84D;
            background: #FFD84D;
            color: #0B3041;
            box-shadow: 0 0 0 3px rgba(255, 192, 0, 0.25);
        }

        .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid #D6A93A;
            background: linear-gradient(180deg, #F5C542 0%, #D6A93A 100%);
            color: #07111F;
            font-weight: 800;
        }

        [data-baseweb="select"],
        [data-baseweb="input"],
        [data-baseweb="textarea"] {
            border-radius: 12px;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #F5C542;
        }

        a {
            color: #F5C542;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=RIFT_TEXT,
        title_font_color=RIFT_TEXT,
        legend_title_font_color=RIFT_MUTED,
        legend_font_color=RIFT_TEXT,
        xaxis=dict(
            gridcolor="rgba(182, 194, 209, 0.15)",
            zerolinecolor="rgba(182, 194, 209, 0.20)",
            tickfont=dict(color=RIFT_TEXT),
            title_font=dict(color=RIFT_MUTED),
        ),
        yaxis=dict(
            gridcolor="rgba(182, 194, 209, 0.15)",
            zerolinecolor="rgba(182, 194, 209, 0.20)",
            tickfont=dict(color=RIFT_TEXT),
            title_font=dict(color=RIFT_MUTED),
        ),
    )

    fig.update_traces(
        marker_line_color=RIFT_DARK,
        marker_line_width=1,
    )

    return fig