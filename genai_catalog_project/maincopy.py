import os
import base64
import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# 🔹 Funktion: Bild in Base64 umwandeln
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ----------------------------------------------------------
# 🔹 .env laden (API-Key)
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("❌ Kein OPENAI_API_KEY gefunden! Bitte .env prüfen.")

# ----------------------------------------------------------
# 🔹 GPT-4 Vision Endpoint
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ----------------------------------------------------------
# 🔹 Prompt für Museumskatalog
BASE_PROMPT = (
    "You are a professional museum curator. "
    "Generate a concise catalog text for this museum object. "
    "Describe its physical characteristics, material, function, "
    "and historical context if visible. Avoid phrases like 'This image shows'. "
    "Write in a neutral, academic tone suitable for a museum catalog."
)

# ----------------------------------------------------------
# 🔸 Funktion: GPT Vision API-Aufruf

def generate_catalog_text(image_paths):
    """
    image_paths: Liste mit Pfaden zu Bildern
    """
    image_contents = []

    for path in image_paths:
        base64_image = encode_image(path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    payload = {
        "model": "gpt-4o-mini",  # Schnell & günstig
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": BASE_PROMPT},
                    *image_contents
                ]
            }
        ],
        "max_tokens": 400
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"❌ Fehler bei API-Anfrage: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("📩 Response:", e.response.text)
        return f"❌ API-Fehler: {str(e)}"


# ----------------------------------------------------------
# 🔸 Hauptfunktion: Excel verarbeiten
def process_excel(excel_path, base_folder, start_line=1, end_line=None):
    df = pd.read_excel(excel_path)

    # 🔹 Spalten für ID und Bildpfade
    possible_id_cols = ["T1", "t1"]
    possible_img_cols = ["T13", "t13"]

    id_col = next((c for c in df.columns if c in possible_id_cols), None)
    img_col = next((c for c in df.columns if c in possible_img_cols), None)

    if not id_col or not img_col:
        print("\n❌ Spaltennamen nicht gefunden!")
        print("👉 Verfügbare Spalten:", list(df.columns))
        raise ValueError("❌ Erwartete Spalten T1/t1 oder T13/t13 fehlen!")

    # 🔹 Zeilenbereich auswählen
    df = df.iloc[start_line - 1:end_line] if end_line else df

    results = []

    print(f"\n📄 Verarbeite {len(df)} Einträge aus {excel_path} ...\n")

    for idx, row in df.iterrows():
        try:
            object_id = str(row[id_col]).strip()
            if not object_id or object_id.lower() == "nan":
                continue

            print(f"🔍 Bearbeite Objekt-ID: {object_id}")

            image_field = str(row[img_col]).strip() if not pd.isna(row[img_col]) else ""
            if not image_field:
                print(f"⚠️ Keine Bildpfade für {object_id} gefunden\n")
                continue

            # 🔹 Pfade bereinigen & splitten
            clean_field = image_field.replace("\\\\", "\\").replace("\r", "").strip()
            image_lines = [line.strip() for line in clean_field.splitlines() if line.strip()]

            image_paths = []
            for path in image_lines:
                filename = os.path.basename(path.replace("\\", "/"))

                # 🔹 Nur gültige Bilddateien berücksichtigen
                if "." not in filename or not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"⚠️ Übersprungen (kein gültiger Bildpfad): {filename}")
                    continue

                # 🔹 Jahr aus ID extrahieren (z. B. 2024 aus 1/2024/0582)
                year = next((p for p in object_id.split("/") if p.isdigit() and len(p) == 4), None)
                if year:
                    full_path = os.path.join(base_folder, year, filename)
                else:
                    full_path = os.path.join(base_folder, filename)

                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    print(f"⚠️ Bild nicht gefunden: {full_path}")

            if not image_paths:
                print(f"⚠️ Keine gültigen Bildpfade für {object_id}\n")
                results.append({
                    "Objekt-ID": object_id,
                    "Bilder": "",
                    "Katalogtext": "❌ Kein gültiges Bild gefunden"
                })
                continue

            # 🔹 GPT Vision Anfrage
            catalog_text = generate_catalog_text(image_paths)

            results.append({
                "Objekt-ID": object_id,
                "Bilder": ", ".join([os.path.basename(p) for p in image_paths]),
                "Katalogtext": catalog_text
            })

            print(f"✅ {object_id} → Text generiert\n")

        except Exception as e:
            print(f"❌ Fehler bei {object_id}: {e}\n")
            results.append({
                "Objekt-ID": object_id,
                "Bilder": "",
                "Katalogtext": f"❌ Fehler: {str(e)}"
            })

    # 🔹 Ergebnisse speichern
    out_path = "catalog_results.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False)
    print(f"\n📘 Ergebnisse gespeichert in: {out_path}")

# ----------------------------------------------------------
# 🔹 MAIN
if __name__ == "__main__":
    excel_path = input("Pfad zur Excel-Datei: ").strip()
    base_folder = input("Pfad zum Basisordner mit Bildern: ").strip()

    process_excel(excel_path, base_folder)
