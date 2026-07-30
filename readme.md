# Riftbound Card Tracker

Lokale Anwendung zum Erkennen und Verwalten einer Riftbound-Kartensammlung.
Das Projekt kombiniert ein **Streamlit-Dashboard**, eine **FastAPI**,
eine **SQLite-Datenbank** und mehrere Computer-Vision-Verfahren.

## Schnellstart mit Docker

Das fertige Image ist über die GitHub Container Registry verfügbar:

```bash
docker pull ghcr.io/lobosaka/cv-app:latest

docker compose up --build
```

Das Dashboard ist anschließend unter
[http://localhost:8501](http://localhost:8501) erreichbar.


## Lokale Installation

Vorausgesetzt werden Python 3.11 oder neuer sowie Git LFS.

```bash
git clone https://github.com/lobosaka/Riftbound-Card-Tracker.git
cd Riftbound-Card-Tracker

git lfs install
git lfs pull

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
for unix-based systems

Backend starten:

```bash
python src/api.py
```

Dashboard in einem zweiten Terminal starten:

```bash
source venv/bin/activate
streamlit run dashboard/app.py
```

Danach stehen folgende Oberflächen bereit:

- Dashboard: [http://localhost:8501](http://localhost:8501)
- API-Dokumentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Verwendung

Im Dashboard kann der gesamte Kartenkatalog durchsucht und nach Set,
Seltenheit, Kartentyp oder Sammlungsstatus gefiltert werden. Der Bestand einer
Karte lässt sich direkt erhöhen, verringern oder auf eine bestimmte Menge
setzen.

Über **Bild-Upload** oder **Kamera Live** wird ein Kartenbild mit einem der drei
Modelle analysiert. Der erkannte Treffer muss bestätigt werden, bevor die Karte
zur Sammlung hinzugefügt wird. Ist der Treffer falsch oder nicht eindeutig,
kann die Karten-ID beziehungsweise der Public Code manuell eingegeben werden.

## Wichtigste API-Endpunkte

| Methode | Pfad | Beschreibung |
| --- | --- | --- |
| `POST` | `/predict/ocr` | Karte per Kartencode erkennen |
| `POST` | `/predict/classification` | Karte per Klassifikation erkennen |
| `POST` | `/predict/representation-learning` | Karte per Ähnlichkeitssuche erkennen |
| `GET` | `/cards` | Alle Karten laden |
| `GET` | `/cards?inventory=collection` | Vorhandene Karten laden |
| `GET` | `/cards?inventory=missing` | Fehlende Karten laden |
| `GET` | `/cards?stats=true` | Sammlungsstatistiken laden |
| `GET` | `/cards/{identifier}` | Einzelne Karte laden |
| `PUT` | `/cards/{card_id}/inventory` | Kartenbestand ändern |

Beispiel für eine OCR-Anfrage:

```bash
curl -X POST "http://127.0.0.1:8000/predict/ocr" \
  -F "file=@/absoluter/pfad/zur/karte.jpg"
```

Beispiel für eine Bestandsänderung:

```bash
curl -X PUT "http://127.0.0.1:8000/cards/unl-130-219/inventory" \
  -H "Content-Type: application/json" \
  -d '{"difference": 1}'
```
## Optionales Data Scraping

```bash
python src/db/scraper.py
python src/db/picture_loader.py
```

## Training Commands
Klassifizierungsmodell:
Stelle im vorhinein sicher, dass die Variable "num_classes" (src/classification/train.py, l. 28) der Anzahl an distinkten Karten in data/card_images entspricht
```bash
python src/classification/train.py
```
Representation Learning Modell:
```bash
python src/services/generate_splits.py
python src/representation_learning/train.py
```