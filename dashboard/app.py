import streamlit as st


st.set_page_config(
    page_title="Riftbound Collection",
    page_icon="🃏",
    layout="wide",
    initial_sidebar_state="expanded",
)


pages = {
    "Sammlung": [
        st.Page(
            "pages/overview.py",
            title="Übersicht",
            icon="📊",
            default=True,
        ),
        st.Page(
            "pages/collection.py",
            title="Meine Sammlung",
            icon="🗂️",
        ),
        st.Page(
            "pages/missing_cards.py",
            title="Fehlende Karten",
            icon="🔍",
        ),
    ],
    "Karten": [
        st.Page(
            "pages/all_cards.py",
            title="Alle Karten",
            icon="🃏",
        ),
        st.Page(
            "pages/camera_live.py",
            title="Kamera Live",
            icon="📷",
        ),
    ],
}


navigation = st.navigation(pages)
navigation.run()
