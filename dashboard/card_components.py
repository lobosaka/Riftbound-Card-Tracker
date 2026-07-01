import math
from typing import Any

import pandas as pd
import streamlit as st


def display_value(value: Any) -> str:
    """
    Wandelt Datenbankwerte in gut lesbaren Text um.

    None, NaN und leere Strings werden als Strich dargestellt.
    """
    if value is None:
        return "–"

    try:
        if pd.isna(value):
            return "–"
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return "–"

    return text


def create_card_filters(
    cards: pd.DataFrame,
    key_prefix: str,
) -> pd.DataFrame:
    """
    Erstellt kombinierbare Filter für Karten.

    Die Filter werden als Schnittmenge angewendet:
    Set UND Seltenheit UND Kartentyp UND Suchbegriff.
    """
    filtered_cards = cards.copy()

    with st.container(border=True):
        st.subheader("Filter")

        search_term = st.text_input(
            "Kartensuche",
            placeholder="Name, Code, Fähigkeit, Effekt, Tags oder Illustrator",
            key=f"{key_prefix}_search",
        )

        filter_column_1, filter_column_2, filter_column_3 = st.columns(3)

        available_sets = sorted(
            cards["setName"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        available_rarities = sorted(
            cards["rarity"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        available_card_types = sorted(
            cards["cardType"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        with filter_column_1:
            selected_sets = st.multiselect(
                "Sets",
                options=available_sets,
                key=f"{key_prefix}_sets",
            )

        with filter_column_2:
            selected_rarities = st.multiselect(
                "Seltenheiten",
                options=available_rarities,
                key=f"{key_prefix}_rarities",
            )

        with filter_column_3:
            selected_card_types = st.multiselect(
                "Kartentypen",
                options=available_card_types,
                key=f"{key_prefix}_card_types",
            )

        sort_column_1, sort_column_2 = st.columns(2)

        with sort_column_1:
            sort_option = st.selectbox(
                "Sortieren nach",
                options=[
                    "Set und Kartennummer",
                    "Name",
                    "Seltenheit",
                    "Kartentyp",
                    "Bestand aufsteigend",
                    "Bestand absteigend",
                ],
                key=f"{key_prefix}_sort",
            )

        with sort_column_2:
            cards_per_page = st.selectbox(
                "Karten pro Seite",
                options=[10, 20, 30, 50],
                index=1,
                key=f"{key_prefix}_page_size",
            )

    if search_term:
        normalized_search = search_term.strip().lower()

        searchable_columns = [
            "name",
            "collectorNumber",
            "publicCode",
            "setName",
            "cardType",
            "superType",
            "rarity",
            "domain_1",
            "domain_2",
            "tags",
            "illustrator",
            "ability",
            "effect",
        ]

        search_mask = pd.Series(
            False,
            index=filtered_cards.index,
        )

        for column in searchable_columns:
            if column not in filtered_cards.columns:
                continue

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

    if selected_card_types:
        filtered_cards = filtered_cards[
            filtered_cards["cardType"].isin(selected_card_types)
        ]

    if sort_option == "Name":
        filtered_cards = filtered_cards.sort_values(
            by=["name", "setName", "collectorNumber"],
            na_position="last",
        )

    elif sort_option == "Seltenheit":
        filtered_cards = filtered_cards.sort_values(
            by=["rarity", "setName", "collectorNumber"],
            na_position="last",
        )

    elif sort_option == "Kartentyp":
        filtered_cards = filtered_cards.sort_values(
            by=["cardType", "setName", "collectorNumber"],
            na_position="last",
        )

    elif sort_option == "Bestand aufsteigend":
        filtered_cards = filtered_cards.sort_values(
            by=["inventory_count", "name"],
            ascending=[True, True],
            na_position="last",
        )

    elif sort_option == "Bestand absteigend":
        filtered_cards = filtered_cards.sort_values(
            by=["inventory_count", "name"],
            ascending=[False, True],
            na_position="last",
        )

    else:
        filtered_cards = filtered_cards.sort_values(
            by=["setName", "collectorNumber", "name"],
            na_position="last",
        )

    filtered_cards = filtered_cards.reset_index(drop=True)

    return paginate_cards(
        cards=filtered_cards,
        cards_per_page=cards_per_page,
        key_prefix=key_prefix,
    )


def paginate_cards(
    cards: pd.DataFrame,
    cards_per_page: int,
    key_prefix: str,
) -> pd.DataFrame:
    """Teilt eine Kartenliste in Seiten auf."""
    total_cards = len(cards)

    st.write(f"**{total_cards} Karten gefunden**")

    if total_cards == 0:
        return cards

    total_pages = max(
        1,
        math.ceil(total_cards / cards_per_page),
    )

    if total_pages == 1:
        return cards

    page_number = st.number_input(
        "Seite",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1,
        key=f"{key_prefix}_page",
    )

    st.caption(
        f"Seite {page_number} von {total_pages}"
    )

    start_index = (page_number - 1) * cards_per_page
    end_index = start_index + cards_per_page

    return cards.iloc[start_index:end_index]


def render_card(card: pd.Series, show_inventory: bool) -> None:
    """Zeigt eine Karte mit sämtlichen vorhandenen Datenbankfeldern."""
    with st.container(border=True):
        image_column, details_column = st.columns(
            [1, 3],
            vertical_alignment="top",
        )

        with image_column:
            image_url = display_value(card.get("image"))

            if image_url != "–":
                st.image(
                    image_url,
                    use_container_width=True,
                )
            else:
                st.info("Kein Kartenbild vorhanden")

        with details_column:
            title_column, inventory_column = st.columns(
                [4, 1],
                vertical_alignment="top",
            )

            with title_column:
                st.subheader(display_value(card.get("name")))

                st.caption(
                    f"{display_value(card.get('setName'))} · "
                    f"#{display_value(card.get('collectorNumber'))} · "
                    f"{display_value(card.get('rarity'))}"
                )

            if show_inventory:
                with inventory_column:
                    st.metric(
                        "Bestand",
                        display_value(
                            card.get("inventory_count")
                        ),
                        border=True,
                    )

            general_tab, values_tab = st.tabs(
                [
                    "Allgemein",
                    "Spielwerte",
                ]
            )

            with general_tab:
                general_column_1, general_column_2 = st.columns(2)

                with general_column_1:
                    write_card_field(
                        "Name",
                        card.get("name"),
                    )
                    write_card_field(
                        "Set",
                        card.get("setName"),
                    )
                    write_card_field(
                        "Sammlernummer",
                        card.get("collectorNumber"),
                    )
                    write_card_field(
                        "Öffentlicher Code",
                        card.get("publicCode"),
                    )
                    write_card_field(
                        "Seltenheit",
                        card.get("rarity"),
                    )

                with general_column_2:
                    write_card_field(
                        "Kartentyp",
                        card.get("cardType"),
                    )
                    write_card_field(
                        "Obertyp",
                        card.get("superType"),
                    )
                    write_card_field(
                        "Primäre Domäne",
                        card.get("domain_1"),
                    )
                    write_card_field(
                        "Sekundäre Domäne",
                        card.get("domain_2"),
                    )
                    write_card_field(
                        "Tags",
                        card.get("tags"),
                    )

            with values_tab:
                value_column_1, value_column_2 = st.columns(2)

                with value_column_1:
                    write_card_field(
                        "Energie",
                        card.get("energy"),
                    )
                    write_card_field(
                        "Macht",
                        card.get("might"),
                    )

                with value_column_2:
                    write_card_field(
                        "Machtbonus",
                        card.get("mightBonus"),
                    )
                    write_card_field(
                        "Power",
                        card.get("power"),
                    )



def write_card_field(
    label: str,
    value: Any,
    multiline: bool = False,
) -> None:
    """Zeigt ein beschriftetes Datenbankfeld an."""
    formatted_value = display_value(value)

    if multiline:
        st.markdown(f"**{label}**")
        st.write(formatted_value)
        return

    st.markdown(
        f"**{label}:** {formatted_value}"
    )