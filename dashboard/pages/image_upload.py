import requests
import streamlit as st

from config import API_BASE_URL


MODEL_OPTIONS = {
    "OCR": "ocr",
    "Classification": "classification",
    "Representation-Learning": "representation-learning",
}


st.title("Bild-Upload")
st.caption(
    "Lade ein Kartenbild hoch, waehle ein Modell aus und starte die Analyse direkt im Dashboard."
)


uploaded_image = st.file_uploader(
    "Kartenbild hochladen",
    type=["png", "jpg", "jpeg", "webp"],
    help="Unterstuetzt gaengige Bildformate fuer Kartenscans oder Fotos.",
)


selected_model = st.selectbox(
    "Modellauswahl",
    options=list(MODEL_OPTIONS.keys()),
    index=0,
    help="Waehle das Modell aus, das fuer die Vorhersage verwendet werden soll.",
)


if uploaded_image is not None:
    st.image(
        uploaded_image,
        caption="Hochgeladenes Bild",
        use_container_width=True,
    )

    if st.button(
        "Analyse starten",
        type="primary",
        use_container_width=True,
    ):
        endpoint = MODEL_OPTIONS[selected_model]

        with st.spinner("Bild wird analysiert..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/predict/{endpoint}",
                    files={
                        "file": (
                            uploaded_image.name,
                            uploaded_image.getvalue(),
                            uploaded_image.type or "application/octet-stream",
                        ),
                    },
                    timeout=180,
                )
                response.raise_for_status()
            except requests.HTTPError as error:
                detail = None
                try:
                    detail = error.response.json()
                except ValueError:
                    detail = error.response.text

                st.error("Die Analyse ist mit einem API-Fehler fehlgeschlagen.")
                if detail:
                    st.json(detail)
            except requests.RequestException as error:
                st.error(
                    "Die API ist nicht erreichbar. Pruefe, ob das Backend auf "
                    f"{API_BASE_URL} laeuft."
                )
                st.caption(str(error))
            else:
                st.success("Analyse abgeschlossen.")
                st.json(response.json())
else:
    st.info("Lade ein Bild hoch, um die Analyse zu starten.")
