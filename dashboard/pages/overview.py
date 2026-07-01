import plotly.express as px
import streamlit as st

from database import (
    load_all_cards,
    load_collection_statistics,
)

def create_progress_dataframe(
    cards,
    group_column: str,
):
    """
    Berechnet gesammelte und fehlende Karten
    für eine bestimmte Kategorie.
    """
    progress = (
        cards.groupby(
            group_column,
            dropna=False,
        )
        .agg(
            gesamt=("id", "count"),
            gesammelt=(
                "inventory_count",
                lambda values: int((values > 0).sum()),
            ),
        )
        .reset_index()
    )

    progress[group_column] = (
        progress[group_column]
        .fillna("Nicht angegeben")
        .astype(str)
    )

    progress["fehlend"] = (
        progress["gesamt"]
        - progress["gesammelt"]
    )

    progress["Fortschritt"] = (
        progress["gesammelt"]
        / progress["gesamt"]
        * 100
    )

    return progress.sort_values(
        by="Fortschritt",
        ascending=False,
    )


def create_progress_chart(
    data,
    x_column: str,
    x_label: str,
    title: str,
):
    """Erstellt ein gestapeltes Fortschrittsdiagramm."""
    chart = px.bar(
        data,
        x=x_column,
        y=["gesammelt", "fehlend"],
        labels={
            x_column: x_label,
            "value": "Anzahl Karten",
            "variable": "Status",
        },
        title=title,
        barmode="stack",
    )

    chart.update_layout(
        xaxis_title=x_label,
        yaxis_title="Anzahl Karten",
        legend_title="Status",
    )

    return chart


st.title("Riftbound-Sammlungsübersicht")
st.caption(
    "Dein aktueller Sammlungsfortschritt auf einen Blick"
)


statistics = load_collection_statistics()
cards = load_all_cards()


total_cards = statistics["total_cards"] or 0
owned_cards = statistics["owned_unique_cards"] or 0
missing_cards = statistics["missing_unique_cards"] or 0
physical_cards = statistics["physical_cards"] or 0
duplicate_cards = statistics["duplicate_cards"] or 0

collection_percentage = (
    owned_cards / total_cards * 100
    if total_cards > 0
    else 0
)


metric_columns = st.columns(6)

metric_columns[0].metric(
    "Karten insgesamt",
    f"{total_cards:,}".replace(",", "."),
    border=True,
)

metric_columns[1].metric(
    "Verschiedene gesammelt",
    f"{owned_cards:,}".replace(",", "."),
    border=True,
)

metric_columns[2].metric(
    "Sammlungsfortschritt",
    f"{collection_percentage:.1f} %",
    border=True,
)

metric_columns[3].metric(
    "Physische Karten",
    f"{physical_cards:,}".replace(",", "."),
    border=True,
)

metric_columns[4].metric(
    "Fehlende Karten",
    f"{missing_cards:,}".replace(",", "."),
    border=True,
)

metric_columns[5].metric(
    "Zusätzliche Exemplare",
    f"{duplicate_cards:,}".replace(",", "."),
    border=True,
)


st.progress(
    min(collection_percentage / 100, 1.0),
    text=(
        f"{owned_cards} von {total_cards} "
        f"unterschiedlichen Karten gesammelt"
    ),
)


st.divider()


left_column, right_column = st.columns(2)


with left_column:
    st.subheader("Fortschritt nach Set")

    set_progress = create_progress_dataframe(
        cards=cards,
        group_column="setName",
    )

    set_chart = create_progress_chart(
        data=set_progress,
        x_column="setName",
        x_label="Set",
        title="Gesammelte und fehlende Karten je Set",
    )

    st.plotly_chart(
        set_chart,
        use_container_width=True,
    )


with right_column:
    st.subheader("Fortschritt nach Seltenheit")

    rarity_progress = create_progress_dataframe(
        cards=cards,
        group_column="rarity",
    )

    rarity_chart = create_progress_chart(
        data=rarity_progress,
        x_column="rarity",
        x_label="Seltenheit",
        title=(
            "Gesammelte und fehlende Karten "
            "nach Seltenheit"
        ),
    )

    st.plotly_chart(
        rarity_chart,
        use_container_width=True,
    )


st.divider()


st.subheader("Fortschritt nach Kartentyp")

card_type_progress = create_progress_dataframe(
    cards=cards,
    group_column="cardType",
)

card_type_chart = create_progress_chart(
    data=card_type_progress,
    x_column="cardType",
    x_label="Kartentyp",
    title="Gesammelte und fehlende Karten nach Kartentyp",
)

st.plotly_chart(
    card_type_chart,
    use_container_width=True,
)


st.divider()


st.subheader("Inventarverteilung")

inventory_distribution = (
    cards.groupby("inventory_count")
    .size()
    .reset_index(name="card_count")
)

inventory_distribution["inventory_label"] = (
    inventory_distribution["inventory_count"]
    .astype(str)
    .map(lambda value: f"{value} Exemplare")
)

inventory_chart = px.bar(
    inventory_distribution,
    x="inventory_label",
    y="card_count",
    labels={
        "inventory_label": "Bestand pro Karte",
        "card_count": "Anzahl unterschiedlicher Karten",
    },
    title="Anzahl Karten nach Inventarbestand",
)

inventory_chart.update_layout(
    xaxis_title="Bestand pro Karte",
    yaxis_title="Unterschiedliche Karten",
)

st.plotly_chart(
    inventory_chart,
    use_container_width=True,
)