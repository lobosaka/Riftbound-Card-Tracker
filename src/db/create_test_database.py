import random
import shutil
import sqlite3
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()

# Von src/db/create_test_inventory.py zurück zum Projektordner
PROJECT_FOLDER = SCRIPT_PATH.parent.parent.parent

# Originaldatenbank
SOURCE_DATABASE = PROJECT_FOLDER / "riftbound.db"

# Neue Testdatenbank mit Inventarwerten
OUTPUT_DATABASE = PROJECT_FOLDER / "riftbound_test_inventory.db"


TABLE_NAME = "cards"
CARD_ID_COLUMN = "id"
INVENTORY_COLUMN = "inventory_count"

MIN_INVENTORY = 0
MAX_INVENTORY = 6

# Gleicher Seed = gleiche Testwerte bei jedem Durchlauf
RANDOM_SEED = 42


def table_exists(
    connection: sqlite3.Connection,
    table_name: str,
) -> bool:
    """Prüft, ob eine Tabelle existiert."""

    result = connection.execute(
        """
        SELECT 1
        FROM sqlite_master
        WHERE type = 'table'
          AND name = ?
        """,
        (table_name,),
    ).fetchone()

    return result is not None


def column_exists(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
) -> bool:
    """Prüft, ob eine Spalte in einer Tabelle existiert."""

    columns = connection.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return any(column[1] == column_name for column in columns)


def create_test_inventory_database() -> None:
    """
    Erstellt eine Kopie der Riftbound-Datenbank.

    Jede einzelne Karte in der Tabelle cards erhält in der
    Spalte inventory_count einen individuellen Zufallswert
    zwischen 0 und 6.
    """

    if not SOURCE_DATABASE.exists():
        raise FileNotFoundError(
            "Die Quelldatenbank wurde nicht gefunden:\n"
            f"{SOURCE_DATABASE}"
        )

    # Vorhandene Testdatenbank löschen
    if OUTPUT_DATABASE.exists():
        OUTPUT_DATABASE.unlink()

    # Originaldatenbank kopieren
    shutil.copy2(
        SOURCE_DATABASE,
        OUTPUT_DATABASE,
    )

    connection = sqlite3.connect(OUTPUT_DATABASE)

    try:
        if not table_exists(connection, TABLE_NAME):
            raise RuntimeError(
                f"Die Tabelle '{TABLE_NAME}' wurde nicht gefunden."
            )

        # Neue Inventarspalte ergänzen
        if not column_exists(
            connection,
            TABLE_NAME,
            INVENTORY_COLUMN,
        ):
            connection.execute(
                f"""
                ALTER TABLE "{TABLE_NAME}"
                ADD COLUMN "{INVENTORY_COLUMN}"
                INTEGER NOT NULL DEFAULT 0
                """
            )

        # Alle 960 eindeutigen Karten-IDs laden
        cards = connection.execute(
            f"""
            SELECT "{CARD_ID_COLUMN}"
            FROM "{TABLE_NAME}"
            ORDER BY "{CARD_ID_COLUMN}"
            """
        ).fetchall()

        if not cards:
            raise RuntimeError(
                "In der Tabelle wurden keine Karten gefunden."
            )

        random_generator = random.Random(RANDOM_SEED)

        inventory_updates = []

        for card in cards:
            card_id = card[0]

            # Eigener Inventarwert für genau diese Karte
            inventory_count = random_generator.randint(
                MIN_INVENTORY,
                MAX_INVENTORY,
            )

            inventory_updates.append(
                (
                    inventory_count,
                    card_id,
                )
            )

        # Jede Karte anhand ihrer eindeutigen ID aktualisieren
        connection.executemany(
            f"""
            UPDATE "{TABLE_NAME}"
            SET "{INVENTORY_COLUMN}" = ?
            WHERE "{CARD_ID_COLUMN}" = ?
            """,
            inventory_updates,
        )

        connection.commit()

        # Kontrolle
        total_cards = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM "{TABLE_NAME}"
            """
        ).fetchone()[0]

        cards_with_inventory = connection.execute(
            f"""
            SELECT COUNT(*)
            FROM "{TABLE_NAME}"
            WHERE "{INVENTORY_COLUMN}" BETWEEN ? AND ?
            """,
            (
                MIN_INVENTORY,
                MAX_INVENTORY,
            ),
        ).fetchone()[0]

        print("Testdatenbank erfolgreich erstellt.")
        print(f"Datenbank: {OUTPUT_DATABASE}")
        print(f"Karten insgesamt: {total_cards}")
        print(
            "Karten mit individuellem Inventarwert: "
            f"{cards_with_inventory}"
        )

        print("\nBeispielwerte:")

        examples = connection.execute(
            f"""
            SELECT
                "{CARD_ID_COLUMN}",
                name,
                "{INVENTORY_COLUMN}"
            FROM "{TABLE_NAME}"
            ORDER BY "{CARD_ID_COLUMN}"
            LIMIT 10
            """
        ).fetchall()

        for card_id, card_name, inventory_count in examples:
            print(
                f"{card_id} | {card_name} | "
                f"Inventar: {inventory_count}"
            )

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


if __name__ == "__main__":
    try:
        create_test_inventory_database()

    except Exception as error:
        print("Fehler:")
        print(error)
        raise

