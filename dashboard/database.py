import sqlite3
from contextlib import closing

import pandas as pd
import streamlit as st

from config import DATABASE_PATH


def check_database_exists() -> None:
    """Prüft, ob die konfigurierte SQLite-Datenbank existiert."""
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Die Datenbank wurde nicht gefunden:\n{DATABASE_PATH}"
        )


def get_connection() -> sqlite3.Connection:
    """
    Erstellt eine neue SQLite-Verbindung.

    Für jede Abfrage beziehungsweise Änderung wird bewusst
    eine neue Verbindung geöffnet.
    """
    check_database_exists()

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


@st.cache_data
def load_all_cards() -> pd.DataFrame:
    """Lädt alle Karten aus der Datenbank."""
    query = """
        SELECT
            id,
            name,
            collectorNumber,
            publicCode,
            setName,
            cardType,
            superType,
            typeIcon,
            rarity,
            rarityIcon,
            domain_1,
            domainIcon_1,
            domain_2,
            domainIcon_2,
            energy,
            might,
            mightBonus,
            power,
            tags,
            illustrator,
            ability,
            effect,
            image,
            inventory_count
        FROM cards
        ORDER BY setName, collectorNumber, name
    """

    with closing(get_connection()) as connection:
        return pd.read_sql_query(query, connection)


@st.cache_data
def load_collection() -> pd.DataFrame:
    """Lädt ausschließlich Karten, die mindestens einmal vorhanden sind."""
    query = """
        SELECT *
        FROM cards
        WHERE inventory_count > 0
        ORDER BY setName, collectorNumber, name
    """

    with closing(get_connection()) as connection:
        return pd.read_sql_query(query, connection)


@st.cache_data
def load_missing_cards() -> pd.DataFrame:
    """Lädt ausschließlich Karten mit einem Bestand von null."""
    query = """
        SELECT *
        FROM cards
        WHERE inventory_count = 0
        ORDER BY setName, collectorNumber, name
    """

    with closing(get_connection()) as connection:
        return pd.read_sql_query(query, connection)


@st.cache_data
def load_collection_statistics() -> dict:
    """Berechnet die wichtigsten Kennzahlen der Sammlung."""
    query = """
        SELECT
            COUNT(*) AS total_cards,

            SUM(
                CASE
                    WHEN inventory_count > 0 THEN 1
                    ELSE 0
                END
            ) AS owned_unique_cards,

            SUM(
                CASE
                    WHEN inventory_count = 0 THEN 1
                    ELSE 0
                END
            ) AS missing_unique_cards,

            SUM(inventory_count) AS physical_cards,

            SUM(
                CASE
                    WHEN inventory_count > 1
                    THEN inventory_count - 1
                    ELSE 0
                END
            ) AS duplicate_cards

        FROM cards
    """

    with closing(get_connection()) as connection:
        row = connection.execute(query).fetchone()

    return dict(row)


def update_inventory(card_id: str, new_quantity: int) -> None:
    """Setzt den Bestand einer Karte auf einen neuen Wert."""
    if new_quantity < 0:
        raise ValueError("Der Bestand darf nicht negativ sein.")

    query = """
        UPDATE cards
        SET inventory_count = ?
        WHERE id = ?
    """

    with closing(get_connection()) as connection:
        cursor = connection.execute(
            query,
            (new_quantity, card_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Es wurde keine Karte mit der ID {card_id!r} gefunden."
            )

        connection.commit()

    clear_database_cache()


def change_inventory(card_id: str, difference: int) -> None:
    """
    Erhöht oder verringert den Bestand einer Karte.

    Der Bestand wird niemals unter null gesetzt.
    """
    query = """
        UPDATE cards
        SET inventory_count = MAX(inventory_count + ?, 0)
        WHERE id = ?
    """

    with closing(get_connection()) as connection:
        cursor = connection.execute(
            query,
            (difference, card_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(
                f"Es wurde keine Karte mit der ID {card_id!r} gefunden."
            )

        connection.commit()

    clear_database_cache()


def clear_database_cache() -> None:
    """Leert die zwischengespeicherten Datenbankabfragen."""
    load_all_cards.clear()
    load_collection.clear()
    load_missing_cards.clear()
    load_collection_statistics.clear()