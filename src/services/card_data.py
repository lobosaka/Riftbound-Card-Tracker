"""
This module provides functions to interact with the SQLite database containing card data.
It includes functions to load all cards, load the user's collection, load missing cards,
load collection statistics, and update inventory counts."""
import sqlite3
from contextlib import closing
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATABASE_PATH = PROJECT_ROOT / "riftbound.db"


def check_database_exists() -> None:
    # Ensure that the database file exists before attempting to connect to it
    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            f"Die Datenbank wurde nicht gefunden:\n{DATABASE_PATH}"
        )


def get_connection() -> sqlite3.Connection:
    # Ensure the database exists before establishing a connection
    check_database_exists()

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def fetch_rows(query: str, parameters: tuple = ()) -> list[dict]:
    # Fetch multiple rows from the database and return them as a list of dictionaries
    with closing(get_connection()) as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [dict(row) for row in rows]


def fetch_row(query: str, parameters: tuple = ()) -> dict | None:
    # Fetch a single row from the database and return it as a dictionary
    with closing(get_connection()) as connection:
        row = connection.execute(query, parameters).fetchone()

    if row is None:
        return None

    return dict(row)


def execute_write(query: str, parameters: tuple = ()) -> int:
    # Execute a write operation (INSERT, UPDATE, DELETE) and return the number of affected rows
    with closing(get_connection()) as connection:
        cursor = connection.execute(query, parameters)
        connection.commit()
        return cursor.rowcount


def load_all_cards() -> list[dict]:
    # Fetch all cards from the database and return them as a list of dictionaries
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

    return fetch_rows(query)


def load_collection() -> list[dict]:
    # Fetch all cards that are in the user's collection (inventory_count > 0) from the database
    query = """
        SELECT *
        FROM cards
        WHERE inventory_count > 0
        ORDER BY setName, collectorNumber, name
    """

    return fetch_rows(query)


def load_missing_cards() -> list[dict]:
    # Fetch all cards that are missing from the user's collection (inventory_count = 0) from the database
    query = """
        SELECT *
        FROM cards
        WHERE inventory_count = 0
        ORDER BY setName, collectorNumber, name
    """

    return fetch_rows(query)


def load_collection_statistics() -> dict:
    # Fetch statistics about the user's card collection from the database
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

    statistics = fetch_row(query)
    return statistics or {}


def load_cards_for_code_index() -> list[dict]:
    # Command to fetch only the necessary fields for building the card code index
    query = """
        SELECT
            id,
            name,
            publicCode,
            collectorNumber
        FROM cards
    """

    return fetch_rows(query)


def update_inventory(card_id: str, new_quantity: int) -> int:
    # Update the inventory count for a specific card in the database
    query = """
        UPDATE cards
        SET inventory_count = ?
        WHERE id = ?
    """

    return execute_write(query, (new_quantity, card_id))


def change_inventory(card_id: str, difference: int) -> int | None:
    # Adjust the inventory count for a specific card by a given difference
    query = """
        UPDATE cards
        SET inventory_count = MAX(inventory_count + ?, 0)
        WHERE id = ?
    """

    rowcount = execute_write(query, (difference, card_id))
    if rowcount == 0:
        return None

    row = fetch_row(
        "SELECT inventory_count FROM cards WHERE id = ?",
        (card_id,),
    )
    if row is None:
        return None

    return row["inventory_count"]
