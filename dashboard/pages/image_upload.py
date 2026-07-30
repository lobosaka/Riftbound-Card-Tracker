from urllib.parse import quote

import requests
import streamlit as st

from card_components import render_card
from config import API_BASE_URL
from database import change_inventory, raise_api_error


MODEL_OPTIONS = {
    "OCR": "ocr",
    "Classification": "classification",
    "Representation-Learning": "representation-learning",
}

REQUEST_TIMEOUT_SECONDS = 30
PREDICTION_TIMEOUT_SECONDS = 180


def reset_detection_state() -> None:
    """
    Setzt alle Werte zurück, die zu einer vorherigen Erkennung gehören.
    """
    st.session_state.detected_card = None
    st.session_state.prediction_result = None
    st.session_state.manual_search_mode = False
    st.session_state.active_upload_signature = None


def initialize_session_state() -> None:
    """
    Legt die benötigten Session-State-Werte beim ersten Seitenaufruf an.
    """
    defaults = {
        "detected_card": None,
        "prediction_result": None,
        "manual_search_mode": False,
        "active_upload_signature": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def fetch_card_by_identifier(identifier: str) -> dict:
    """
    Lädt eine Karte anhand ihrer internen ID oder ihres Public Codes.
    """
    encoded_identifier = quote(identifier, safe="")

    response = requests.get(
        f"{API_BASE_URL}/cards/{encoded_identifier}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    raise_api_error(response)
    return response.json()


def predict_card(
    image_name: str,
    image_bytes: bytes,
    mime_type: str,
    model_label: str,
) -> dict:
    """
    Sendet das hochgeladene Bild an das ausgewählte Vorhersagemodell.
    """
    endpoint = MODEL_OPTIONS[model_label]

    response = requests.post(
        f"{API_BASE_URL}/predict/{endpoint}",
        files={
            "file": (
                image_name,
                image_bytes,
                mime_type,
            ),
        },
        timeout=PREDICTION_TIMEOUT_SECONDS,
    )

    raise_api_error(response)
    return response.json()


def extract_identifier_from_prediction(
    prediction_result: dict,
) -> str | None:
    """
    Versucht, eine Karten-ID oder einen Kartencode aus verschiedenen
    möglichen API-Antwortformaten zu ermitteln.
    """
    direct_keys = [
        "card_id",
        "id",
        "code",
        "publicCode",
        "card_name",
        "cardName",
        "prediction",
        "predicted_class",
        "class",
    ]

    for key in direct_keys:
        value = prediction_result.get(key)

        if value:
            return str(value)

    nested_candidates = [
        prediction_result.get("best_match"),
        prediction_result.get("result"),
        prediction_result.get("card"),
    ]

    for candidate in nested_candidates:
        if not isinstance(candidate, dict):
            continue

        for key in direct_keys:
            value = candidate.get(key)

            if value:
                return str(value)

    return None


def build_upload_signature(
    uploaded_image,
    selected_model: str,
) -> str | None:
    """
    Erstellt eine Signatur, damit bei einem neuen Bild oder Modell
    alte Erkennungsergebnisse zurückgesetzt werden.
    """
    if uploaded_image is None:
        return None

    return (
        f"{uploaded_image.name}:"
        f"{uploaded_image.size}:"
        f"{uploaded_image.type}:"
        f"{selected_model}"
    )


def sync_detection_state(
    current_upload_signature: str | None,
) -> None:
    """
    Erkennt, ob ein anderes Bild oder Modell ausgewählt wurde.
    """
    previous_signature = st.session_state.active_upload_signature

    if previous_signature is None:
        return

    if previous_signature == current_upload_signature:
        return

    reset_detection_state()


@st.dialog(
    "Erkannte Karte bestätigen",
    width="large",
)
def show_detected_card_dialog() -> None:
    """
    Zeigt die erkannte oder manuell ausgewählte Karte an.
    """
    card = st.session_state.detected_card

    if card is None:
        return

    st.write("Die Erkennung hat folgende Karte gefunden:")

    render_card(
        card=card,
        show_inventory=True,
    )

    st.divider()

    confirm_column, cancel_column = st.columns(2)

    with confirm_column:
        if st.button(
            "Ja, das ist die richtige Karte",
            type="primary",
            use_container_width=True,
            key="confirm_uploaded_card",
        ):
            try:
                change_inventory(card["id"], 1)

            except RuntimeError as error:
                st.error(str(error))

            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Prüfe, ob das Backend auf "
                    f"{API_BASE_URL} läuft."
                )
                st.caption(str(error))

            else:
                reset_detection_state()

                st.success(
                    "Die Karte wurde deinem Inventar hinzugefügt."
                )

                st.rerun()

    with cancel_column:
        if st.button(
            "Abbrechen",
            use_container_width=True,
            key="cancel_uploaded_card",
        ):
            reset_detection_state()
            st.rerun()

    st.divider()

    with st.expander("Das ist nicht deine Karte?"):
        manual_identifier = st.text_input(
            "Karten-ID oder Code manuell eingeben",
            placeholder="Zum Beispiel ogn-001-298",
            key="manual_identifier_in_upload_dialog",
        )

        if st.button(
            "Manuelle Karte prüfen",
            use_container_width=True,
            key="check_manual_card_in_upload_dialog",
        ):
            cleaned_identifier = manual_identifier.strip()

            if not cleaned_identifier:
                st.warning(
                    "Bitte gib zuerst eine Karten-ID oder einen Code ein."
                )
                return

            try:
                corrected_card = fetch_card_by_identifier(
                    cleaned_identifier
                )

            except RuntimeError as error:
                st.error(str(error))

            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Prüfe, ob das Backend auf "
                    f"{API_BASE_URL} läuft."
                )
                st.caption(str(error))

            else:
                st.session_state.detected_card = corrected_card
                st.session_state.manual_search_mode = False
                st.rerun()


st.title("Bild-Upload")

st.caption(
    "Lade ein Kartenbild hoch, wähle ein Modell aus und starte "
    "die Analyse direkt im Dashboard."
)


initialize_session_state()


uploaded_image = st.file_uploader(
    "Kartenbild hochladen",
    type=[
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
    help=(
        "Unterstützt gängige Bildformate für Kartenscans oder Fotos."
    ),
)


selected_model = st.selectbox(
    "Modellauswahl",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    help=(
        "Wähle das Modell aus, das für die Vorhersage verwendet werden soll."
    ),
)


current_upload_signature = build_upload_signature(
    uploaded_image,
    selected_model,
)

sync_detection_state(current_upload_signature)


if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Hochgeladenes Bild",
        use_container_width=True,
    )

    start_analysis = st.button(
        "Analyse starten",
        type="primary",
        use_container_width=True,
    )

    if start_analysis:
        image_bytes = uploaded_image.getvalue()

        mime_type = (
            uploaded_image.type
            or "application/octet-stream"
        )

        st.session_state.active_upload_signature = (
            current_upload_signature
        )

        with st.spinner("Bild wird analysiert..."):
            try:
                prediction_result = predict_card(
                    image_name=uploaded_image.name,
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    model_label=selected_model,
                )

                st.session_state.prediction_result = (
                    prediction_result
                )

                detected_identifier = (
                    extract_identifier_from_prediction(
                        prediction_result
                    )
                )

                if detected_identifier is None:
                    st.session_state.manual_search_mode = True
                    st.rerun()

                card = fetch_card_by_identifier(
                    detected_identifier
                )

            except RuntimeError as error:
                st.error(str(error))

            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Prüfe, ob das Backend auf "
                    f"{API_BASE_URL} läuft."
                )
                st.caption(str(error))

            else:
                st.session_state.detected_card = card
                st.session_state.manual_search_mode = False
                st.rerun()

else:
    st.info(
        "Lade ein Bild hoch, um die Analyse zu starten."
    )


if st.session_state.manual_search_mode:
    st.warning(
        "Es konnte keine Karte automatisch erkannt werden."
    )

    st.info(
        "Du kannst die Karten-ID oder den Public Code manuell eingeben."
    )

    manual_identifier = st.text_input(
        "Karten-ID oder Code manuell eingeben",
        placeholder="Zum Beispiel VEN-012/166",
        key="manual_identifier_after_failed_upload_detection",
    )

    if st.button(
        "Manuelle Karte suchen",
        type="primary",
        use_container_width=True,
        key="manual_search_after_failed_upload_detection",
    ):
        cleaned_identifier = manual_identifier.strip()

        if not cleaned_identifier:
            st.warning(
                "Bitte gib zuerst eine Karten-ID oder einen Code ein."
            )
            st.stop()

        try:
            card = fetch_card_by_identifier(
                cleaned_identifier
            )

        except RuntimeError as error:
            st.error(str(error))

        except requests.RequestException as error:
            st.error(
                "Die API ist nicht erreichbar. Prüfe, ob das Backend auf "
                f"{API_BASE_URL} läuft."
            )
            st.caption(str(error))

        else:
            st.session_state.detected_card = card
            st.session_state.manual_search_mode = False
            st.rerun()

    if st.session_state.prediction_result is not None:
        with st.expander("Modellantwort anzeigen"):
            st.json(
                st.session_state.prediction_result
            )


if st.session_state.detected_card is not None:
    show_detected_card_dialog()