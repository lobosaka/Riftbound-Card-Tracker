import requests
import streamlit as st
import streamlit.components.v1 as components

from config import API_BASE_URL


st.title("Kamera Live")
st.caption(
    "Live-Vorschau direkt ueber den Browser. Damit kannst du die externe USB-Kamera waehlen,"
    " statt ueber OpenCV-Geraeteindizes zu raten."
)

st.info(
    "Beim ersten Zugriff fragt der Browser nach Kamerarechten. Waehle dort die externe USB-Kamera."
)

components.html(
    """
    <style>
    .camera-shell {
      border: 1px solid #d1d5db;
      border-radius: 14px;
      padding: 1rem;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    .camera-shell video {
      width: 100%;
      max-height: 70vh;
      background: #111827;
      border-radius: 12px;
      object-fit: contain;
    }
    .camera-shell select,
    .camera-shell button {
      font: inherit;
      padding: 0.55rem 0.8rem;
      border-radius: 10px;
      border: 1px solid #cbd5e1;
      background: white;
      margin-right: 0.5rem;
      margin-bottom: 0.75rem;
    }
    .camera-shell .status {
      font-size: 0.95rem;
      color: #334155;
      margin-bottom: 0.75rem;
    }
    </style>
    <div class="camera-shell">
      <div class="status" id="camera-status">Initialisiere Kameraauswahl...</div>
      <div>
        <select id="camera-select"></select>
        <button id="refresh-cameras" type="button">Neu laden</button>
      </div>
      <video id="camera-preview" autoplay playsinline muted></video>
    </div>
    <script>
    const statusEl = document.getElementById("camera-status");
    const selectEl = document.getElementById("camera-select");
    const videoEl = document.getElementById("camera-preview");
    const refreshButton = document.getElementById("refresh-cameras");
    let activeStream = null;

    function setStatus(message) {
      statusEl.textContent = message;
    }

    function stopStream() {
      if (!activeStream) {
        return;
      }
      for (const track of activeStream.getTracks()) {
        track.stop();
      }
      activeStream = null;
    }

    async function startStream(deviceId) {
      stopStream();
      try {
        const constraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false
        };
        activeStream = await navigator.mediaDevices.getUserMedia(constraints);
        videoEl.srcObject = activeStream;
        const track = activeStream.getVideoTracks()[0];
        const label = track ? track.label : "Unbekannte Kamera";
        setStatus(`Aktive Kamera: ${label}`);
      } catch (error) {
        setStatus(`Kamera konnte nicht gestartet werden: ${error.message}`);
      }
    }

    async function loadDevices() {
      setStatus("Frage Kamerazugriff an...");
      try {
        const tempStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        for (const track of tempStream.getTracks()) {
          track.stop();
        }

        const devices = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = devices.filter((device) => device.kind === "videoinput");

        selectEl.innerHTML = "";
        if (videoDevices.length === 0) {
          setStatus("Keine Videogeraete gefunden.");
          return;
        }

        for (const [index, device] of videoDevices.entries()) {
          const option = document.createElement("option");
          option.value = device.deviceId;
          option.textContent = device.label || `Kamera ${index + 1}`;
          selectEl.appendChild(option);
        }

        const preferredOption = Array.from(selectEl.options).find((option) =>
          option.textContent.toLowerCase().includes("usb") ||
          option.textContent.toLowerCase().includes("hbvcam")
        );

        if (preferredOption) {
          selectEl.value = preferredOption.value;
        }

        await startStream(selectEl.value);
      } catch (error) {
        setStatus(`Kamerazugriff fehlgeschlagen: ${error.message}`);
      }
    }

    selectEl.addEventListener("change", async () => {
      await startStream(selectEl.value);
    });

    refreshButton.addEventListener("click", async () => {
      await loadDevices();
    });

    loadDevices();
    </script>
    """,
    height=760,
    scrolling=False,
)

st.markdown(
    "Wenn mehrere Kameras angezeigt werden, waehle `USB Camera` oder `HBVCAM Camera`."
)

st.divider()
st.subheader("Foto aufnehmen und OCR starten")
st.caption(
    "Dieser Schritt nutzt ebenfalls die Browser-Kameraauswahl und umgeht damit die OpenCV-Geraeteindizes auf dem Backend."
)

captured_photo = st.camera_input(
    "Kartenfoto aufnehmen",
    help="Waehle im Browser die externe USB-Kamera aus und nimm dann ein Bild auf.",
)

if captured_photo is not None:
    st.image(
        captured_photo,
        caption="Aufgenommenes Bild",
        use_container_width=True,
    )
    st.selectbox(
        "Modellauswahl",
        options=["OCR", "Classification", "Representation-Learning"],
        key="model_selection",
        help="Waehle das Modell aus, das für die Vorhersage verwendet werden soll.",
    )
    if st.session_state.model_selection:
      model_response = requests.post(
          f"{API_BASE_URL}/predict/{st.session_state.model_selection.lower().replace(' ', '-')}",
          files={"file": captured_photo.getvalue()},
          timeout=180,
      )
      model_response.raise_for_status()
      model_result = model_response.json()
      st.json(model_result)


