import streamlit as st

from card_components import (
    create_card_filters,
    render_card,
)
from database import load_missing_cards


st.title("Fehlende Karten")
st.caption(
    "Alle Karten, deren Inventarbestand aktuell null ist"
)


cards = load_missing_cards()


if cards.empty:
    st.success(
        "Du besitzt bereits jede Karte!"
    )
    st.stop()


filtered_cards = create_card_filters(
    cards=cards,
    key_prefix="missing",
)


if filtered_cards.empty:
    st.warning(
        "Für diese Filterkombination wurden keine fehlenden Karten gefunden."
    )
    st.stop()


for _, card in filtered_cards.iterrows():
    render_card(
        card=card,
        show_inventory=False,
    )