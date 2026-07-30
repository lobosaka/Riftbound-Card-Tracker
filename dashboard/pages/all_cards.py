import pandas as pd
import streamlit as st

from database import load_all_cards


st.title("Alle Riftbound-Karten")
st.caption("Durchsuche und filtere den vollständigen Kartenkatalog")


cards = load_all_cards()


with st.sidebar:
    st.header("Filter")

    search_term = st.text_input(
        "Kartensuche",
        placeholder="Name, Code oder Kartentext",
    )

    available_sets = sorted(
        cards["setName"].dropna().unique().tolist()
    )

    selected_sets = st.multiselect(
        "Sets",
        options=available_sets,
    )

    available_rarities = sorted(
        cards["rarity"].dropna().unique().tolist()
    )

    selected_rarities = st.multiselect(
        "Seltenheiten",
        options=available_rarities,
    )

    available_types = sorted(
        cards["cardType"].dropna().unique().tolist()
    )

    selected_types = st.multiselect(
        "Kartentypen",
        options=available_types,
    )


filtered_cards = cards.copy()


if search_term:
    normalized_search = search_term.strip().lower()

    searchable_columns = [
        "name",
        "publicCode",
        "ability",
        "effect",
        "tags",
        "illustrator",
    ]

    search_mask = pd.Series(
        False,
        index=filtered_cards.index,
    )

    for column in searchable_columns:
        search_mask |= (
            filtered_cards[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.contains(
                normalized_search,
                regex=False,
            )
        )

    filtered_cards = filtered_cards[search_mask]


if selected_sets:
    filtered_cards = filtered_cards[
        filtered_cards["setName"].isin(selected_sets)
    ]


if selected_rarities:
    filtered_cards = filtered_cards[
        filtered_cards["rarity"].isin(selected_rarities)
    ]


if selected_types:
    filtered_cards = filtered_cards[
        filtered_cards["cardType"].isin(selected_types)
    ]


st.write(
    f"**{len(filtered_cards)} Karten gefunden**"
)


display_columns = [
    "image",
    "name",
    "setName",
    "collectorNumber",
    "rarity",
    "cardType",
    "domain_1",
    "domain_2",
    "energy",
    "power",
    "inventory_count",
]


st.dataframe(
    filtered_cards[display_columns],
    hide_index=True,
    use_container_width=True,
    column_config={
        "image": st.column_config.ImageColumn(
            "Karte",
            width="small",
        ),
        "name": "Name",
        "setName": "Set",
        "collectorNumber": "Nummer",
        "rarity": "Seltenheit",
        "cardType": "Kartentyp",
        "domain_1": "Domäne 1",
        "domain_2": "Domäne 2",
        "energy": "Energie",
        "power": "Stärke",
        "inventory_count": st.column_config.NumberColumn(
            "Bestand",
            format="%d",
        ),
    },
)