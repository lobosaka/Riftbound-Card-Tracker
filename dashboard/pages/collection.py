import streamlit as st

from card_components import (
    create_card_filters,
    render_card,
)
from database import load_collection


st.title("Meine Sammlung")
st.caption(
    "Alle Karten, die du mindestens einmal besitzt"
)


cards = load_collection()


if cards.empty:
    st.info(
        "Deine Sammlung enthält momentan keine Karten."
    )
    st.stop()


filtered_cards = create_card_filters(
    cards=cards,
    key_prefix="collection",
)


if filtered_cards.empty:
    st.warning(
        "Für diese Filterkombination wurden keine Karten gefunden."
    )
    st.stop()


for _, card in filtered_cards.iterrows():
    render_card(
        card=card,
        show_inventory=True,
    )