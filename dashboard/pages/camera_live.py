import base64
import binascii
import hashlib
from urllib.parse import quote

import requests
import streamlit as st
import cv2
import numpy as np

from card_components import render_card
from components.browser_camera import browser_camera
from config import API_BASE_URL
from database import change_inventory, raise_api_error


MODEL_OPTIONS = {
    "OCR": "ocr",
    "Classification": "classification",
    "Representation-Learning": "representation-learning",
}

REQUEST_TIMEOUT_SECONDS = 30
PREDICTION_TIMEOUT_SECONDS = 180

CAMERA_MATRIX = np.array(
                [[1239.3367501467612, 0.0, 367.8840428570087], [0.0, 1242.9872838383162, 673.4001708112677], [0.0, 0.0, 1.0]]
                , dtype=np.float32)

DIST_COEFFS = np.array(
            [[-0.23029730654775918, -0.06403435155721206, 0.00026812586219995214, 0.0002515795799608243, -0.10366338898133773]]
            , dtype=np.float32)


def reset_detection_state() -> None:
    st.session_state.detected_card = None
    st.session_state.prediction_result = None
    st.session_state.manual_search_mode = False


def initialize_session_state() -> None:
    defaults = {
        "detected_card": None,
        "prediction_result": None,
        "manual_search_mode": False,
        "active_detection_signature": None,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def fetch_card_by_identifier(identifier: str) -> dict:
    encoded_identifier = quote(identifier, safe="")
    response = requests.get(
        f"{API_BASE_URL}/cards/{encoded_identifier}",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    raise_api_error(response)
    return response.json()


def decode_camera_capture(camera_capture: dict) -> tuple[bytes, str]:
    image_data_url = camera_capture.get("imageDataUrl")
    if not isinstance(image_data_url, str):
        raise ValueError("Die Kamera hat kein Bild geliefert.")

    header, separator, encoded_image = image_data_url.partition(",")
    if not separator or not header.startswith("data:image/") or ";base64" not in header:
        raise ValueError("Das Kamerabild hat ein ungültiges Format.")

    mime_type = camera_capture.get("mimeType")
    if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
        mime_type = header.removeprefix("data:").split(";", maxsplit=1)[0]

    try:
        image_bytes = base64.b64decode(encoded_image, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Das Kamerabild konnte nicht gelesen werden.") from error

    if not image_bytes:
        raise ValueError("Das Kamerabild ist leer.")

    return image_bytes, mime_type


def predict_card(image_bytes: bytes, mime_type: str, model_label: str) -> dict:
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.undistort(frame, CAMERA_MATRIX, DIST_COEFFS)

    ext = '.' + mime_type.split('/')[-1] if '/' in mime_type else '.jpg'
    success, encoded_image = cv2.imencode(ext, frame)
    if not success:
        raise ValueError("Failed to encode processed frame back to bytes.")
    processed_bytes = encoded_image.tobytes()

    endpoint = MODEL_OPTIONS[model_label]
    response = requests.post(
        f"{API_BASE_URL}/predict/{endpoint}",
        files={
            "file": (
                f"camera_capture{ext}",
                processed_bytes,
                mime_type,
            ),
        },
        timeout=PREDICTION_TIMEOUT_SECONDS,
    )
    raise_api_error(response)
    return response.json()

def extract_identifier_from_prediction(prediction_result: dict) -> str | None:
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


def build_capture_signature(image_bytes: bytes, mime_type: str) -> str:
    image_hash = hashlib.blake2s(image_bytes, digest_size=12).hexdigest()
    return f"{mime_type}:{image_hash}"


def build_detection_signature(
    capture_signature: str | None,
    model_label: str,
) -> str | None:
    if capture_signature is None:
        return None

    return f"{model_label}:{capture_signature}"


def sync_detection_state(current_detection_signature: str | None) -> None:
    previous_signature = st.session_state.active_detection_signature

    if previous_signature == current_detection_signature:
        return

    if previous_signature is not None:
        reset_detection_state()
        st.session_state.active_detection_signature = None


@st.dialog("Erkannte Karte bestätigen", width="large")
def show_detected_card_dialog() -> None:
    card = st.session_state.detected_card

    st.write("Die Erkennung hat folgende Karte gefunden:")
    render_card(card=card, show_inventory=True)

    st.divider()
    confirm_column, cancel_column = st.columns(2)

    with confirm_column:
        if st.button(
            "Ja, das ist die richtige Karte",
            type="primary",
            use_container_width=True,
        ):
            try:
                change_inventory(card["id"], 1)
            except RuntimeError as error:
                st.error(str(error))
            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Pruefe, ob das Backend auf "
                    f"{API_BASE_URL} laeuft."
                )
                st.caption(str(error))
            else:
                reset_detection_state()
                st.success("Die Karte wurde deinem Inventar hinzugefügt.")
                st.rerun()

    with cancel_column:
        if st.button(
            "Abbrechen",
            use_container_width=True,
        ):
            reset_detection_state()
            st.rerun()

    st.divider()

    with st.expander("Das ist nicht deine Karte?"):
        manual_identifier = st.text_input(
            "Karten-ID oder Code manuell eingeben",
            placeholder="Zum Beispiel ID oder Public Code",
        )

        if st.button(
            "Manuelle Karte prüfen",
            use_container_width=True,
        ):
            cleaned_identifier = manual_identifier.strip()

            if not cleaned_identifier:
                st.warning("Bitte gib zuerst eine Karten-ID ein.")
                return

            try:
                corrected_card = fetch_card_by_identifier(cleaned_identifier)
            except RuntimeError as error:
                st.error(str(error))
            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Pruefe, ob das Backend auf "
                    f"{API_BASE_URL} laeuft."
                )
                st.caption(str(error))
            else:
                st.session_state.detected_card = corrected_card
                st.session_state.manual_search_mode = False
                st.rerun()


st.title("Kamera Live")
st.caption(
    "Live-Vorschau direkt ueber den Browser. Damit kannst du die externe USB-Kamera waehlen,"
    " statt ueber OpenCV-Geraeteindizes zu raten."
)

st.info(
    "Beim ersten Zugriff fragt der Browser nach Kamerarechten. Waehle dort die externe USB-Kamera."
)

camera_capture = browser_camera(
    key="card_camera",
    height=760,
)

st.caption(
    "Auswahl, Live-Vorschau und Fotoaufnahme verwenden denselben Kamera-Stream."
)

initialize_session_state()

selected_model = st.selectbox(
    "Modellauswahl",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    key="model_selection",
    help="Waehle das Modell aus, das für die Vorhersage verwendet werden soll.",
)

captured_photo = None
captured_mime_type = "image/jpeg"
capture_signature = None

if camera_capture is not None:
    try:
        captured_photo, captured_mime_type = decode_camera_capture(camera_capture)
        capture_signature = build_capture_signature(
            captured_photo,
            captured_mime_type,
        )
    except ValueError as error:
        st.error(str(error))
        captured_photo = None

current_detection_signature = build_detection_signature(
    capture_signature,
    selected_model,
)
sync_detection_state(current_detection_signature)

if camera_capture is not None and captured_photo is not None:
    st.image(
        captured_photo,
        caption=(
            "Aufgenommenes Bild"
            f" – {camera_capture.get('deviceLabel', 'ausgewählte Kamera')}"
        ),
        use_container_width=True,
    )

    if st.button(
        "Erkennung starten",
        type="primary",
        use_container_width=True,
    ):
        st.session_state.active_detection_signature = current_detection_signature
        with st.spinner("Bild wird analysiert..."):
            try:
                prediction_result = predict_card(
                    captured_photo,
                    captured_mime_type,
                    selected_model,
                )
                st.session_state.prediction_result = prediction_result

                detected_identifier = extract_identifier_from_prediction(
                    prediction_result
                )

                if detected_identifier is None:
                    st.session_state.manual_search_mode = True
                    st.rerun()

                card = fetch_card_by_identifier(detected_identifier)

            except RuntimeError as error:
                st.error(str(error))
            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Pruefe, ob das Backend auf "
                    f"{API_BASE_URL} laeuft."
                )
                st.caption(str(error))
            else:
                st.session_state.detected_card = card
                st.session_state.manual_search_mode = False
                st.rerun()
else:
    st.info("Nimm ein Foto auf, um die Erkennung zu starten.")

if st.session_state.manual_search_mode:
    st.warning("Es konnte keine Karte automatisch erkannt werden.")
    st.info("Du kannst die Karten-ID oder den Public Code manuell eingeben.")

    manual_identifier = st.text_input(
        "Karten-ID oder Code manuell eingeben",
        placeholder="Zum Beispiel VEN-012/166",
        key="manual_identifier_after_failed_detection",
    )

    if st.button(
        "Manuelle Karte suchen",
        type="primary",
        key="manual_search_after_failed_detection",
    ):
        cleaned_identifier = manual_identifier.strip()

        if not cleaned_identifier:
            st.warning("Bitte gib zuerst eine Karten-ID ein.")
            st.stop()

        try:
            card = fetch_card_by_identifier(cleaned_identifier)
        except RuntimeError as error:
            st.error(str(error))
        except requests.RequestException as error:
            st.error(
                "Die API ist nicht erreichbar. Pruefe, ob das Backend auf "
                f"{API_BASE_URL} laeuft."
            )
            st.caption(str(error))
        else:
            st.session_state.detected_card = card
            st.session_state.manual_search_mode = False
            st.rerun()

    if st.session_state.prediction_result is not None:
        with st.expander("Modellantwort anzeigen"):
            st.json(st.session_state.prediction_result)

if st.session_state.detected_card is not None:
    show_detected_card_dialog()
