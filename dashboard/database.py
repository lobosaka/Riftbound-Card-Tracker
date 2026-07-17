import pandas as pd
import requests
import streamlit as st

from config import API_BASE_URL


REQUEST_TIMEOUT_SECONDS = 10


def build_url(path: str) -> str:
    return f"{API_BASE_URL.rstrip('/')}/{path.lstrip('/')}"


def raise_api_error(response: requests.Response) -> None:
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        detail = None

        try:
            payload = response.json()
        except ValueError:
            payload = None

        if isinstance(payload, dict):
            detail = payload.get("detail")

        if detail:
            raise RuntimeError(detail) from error

        raise RuntimeError(
            f"API request failed with status {response.status_code}."
        ) from error


def get_json(path: str):
    response = requests.get(
        build_url(path),
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    raise_api_error(response)
    return response.json()


@st.cache_data
def load_all_cards() -> pd.DataFrame:
    return pd.DataFrame(get_json("/cards"))


@st.cache_data
def load_collection() -> pd.DataFrame:
    return pd.DataFrame(get_json("/cards/collection"))


@st.cache_data
def load_missing_cards() -> pd.DataFrame:
    return pd.DataFrame(get_json("/cards/missing"))


@st.cache_data
def load_collection_statistics() -> dict:
    return get_json("/cards/statistics")


def update_inventory(card_id: str, new_quantity: int) -> None:
    response = requests.put(
        build_url(f"/cards/{card_id}/inventory"),
        json={"new_quantity": new_quantity},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    raise_api_error(response)
    clear_database_cache()


def change_inventory(card_id: str, difference: int) -> None:
    response = requests.post(
        build_url(f"/cards/{card_id}/inventory/change"),
        json={"difference": difference},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    raise_api_error(response)
    clear_database_cache()


def clear_database_cache() -> None:
    load_all_cards.clear()
    load_collection.clear()
    load_missing_cards.clear()
    load_collection_statistics.clear()
