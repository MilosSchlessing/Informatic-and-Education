import os
import base64
import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# 🔹 .env laden
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Kein OpenAI API-Key gefunden. Bitte in .env eintragen.")

# 🔹 GPT-4 Vision Endpoint
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# 🔹 Prompt für Katalogtext
BASE_PROMPT = (
    "You are a professional museum curator. "
    "Generate a concise catalog text for this museum object based on the provided images. "
    "Describe its physical characteristics, material, function, and historical context if visible. "
    "Avoid phrases like 'This image shows'. Write in a neutral, academic tone suitable for a museum catalog."
)

# ----------------------------------------------------------
# 🔸 Funktion: Bild einlesen und base64 encoden
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# ----------------------------------------------------------
# 🔸 GPT-4 Vision Anfrage
def generate_catalog_text(image_paths):
    images_data = []
    for img_path in image_paths:
        try:
            img_base64 = encode_image(img_path)
            images_data.append({"type": "input_image", "image_data": img_base64})
        except Exception as e:
            print(f"⚠️ Fehler beim Laden von {img_path}: {e}")

    if not images_data:
        return "⚠️ Keine gültigen Bilder gefunden."

    payload = {
        "model": "gpt-4o-mini",  # oder "gpt-4o"
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": BASE_PROMPT}] + images_data
            }
        ],
        "max_output_tokens": 400
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(OPENAI_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()

# ----------------------------------------------------------
# 🔸 Hilfsfunktion: Pfad zur Jahresmappe finden
def get_year_from_id(object_id: str):
    """
    Beispiel-ID: 1/1996/6864 0 → Jahr = 1996
    """
    try:
        parts = object_id.split("/")
        for p in parts:
            if p.isdigit() and len(p) == 4:
                return p
    except Exception:
        return None

# ----------------------------------------------------------
# 🔸 Hilfsfunktion: Bilder im Jahr-Ordner finden
def find_images_for_id(base_folder, object_id):
    """
    ID z. B. 1/1996/6864 0 → Bilder heißen 1-1996-6864-000-000.JPG usw.
    """
    year = get_year_from_id(object_id)
    if not year:
        return []

    folder_path = os.path.join(base_folder, year)
    if not os.path.exists(folder_path):
        print(f"⚠️ Jahr-Ordner nicht gefunden: {folder_path}")
        return []

    # ID normalisieren: "1/1996/6864 0" → "1-1996-6864"
    base_name = "-".join(object_id.split("/")[:3]).split()[0]
    matched_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if base_name in f and f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return matched_files

# ----------------------------------------------------------
# 🔸 Excel einlesen und Zeilenbereich wählen
def load_filtered_data(file_path, start_line, end_line):
    df = pd.read_excel(file_path)
    return df.iloc[start_line - 1:end_line]

# ----------------------------------------------------------
# 🔸 Hauptfunktion
def process_excel_and_images(base_folder, excel_paths, line_ranges):
    results = []

    for excel_path, (start, end) in zip(excel_paths, line_ranges):
        df_filtered = load_filtered_data(excel_path, start, end)
        print(f"📄 Verarbeite {excel_path} (Zeilen {start}-{end}), {len(df_filtered)} Einträge")

        # Suche Spalte mit Objekt-ID automatisch
        # id_col = next((c for c in df_filtered.columns if "ID" in c or "Id" in c), None)
        # if not id_col:
        #    raise ValueError(f"❌ Keine ID-Spalte in {excel_path} gefunden!")
        
        # 🔹 Zielspalten (Groß-/Kleinschreibung tolerant)
        possible_id_cols = ["T1", "t1"]
        possible_img_cols = ["T3", "t3"]

        # Finde tatsächliche Spaltennamen
        id_col = next((c for c in df_filtered.columns if c in possible_id_cols), None)
        img_col = next((c for c in df_filtered.columns if c in possible_img_cols), None)

        if not id_col or not img_col:
            print("\n❌ Spaltennamen nicht gefunden!")
            print("👉 Verfügbare Spalten:", list(df_filtered.columns))
            raise ValueError(f"❌ Erwartete Spalten T1/t1 oder T3/t3 fehlen in {excel_path}!")

        for idx, row in df_filtered.iterrows():
            object_id = str(row[id_col]).strip()
            if not object_id or object_id == "nan":
                continue

            print(f"\n🔍 Bearbeite Objekt-ID: {object_id}")
            image_paths = find_images_for_id(base_folder, object_id)

            if not image_paths:
                print(f"⚠️ Keine Bilder gefunden für {object_id}")
                continue

            try:
                catalog_text = generate_catalog_text(image_paths)
                results.append({
                    "Excel": os.path.basename(excel_path),
                    "Zeile": idx + 1,
                    "Objekt-ID": object_id,
                    "Bilder": ", ".join([os.path.basename(p) for p in image_paths]),
                    "Katalogtext": catalog_text
                })
                print(f"✅ {object_id} ({len(image_paths)} Bilder) → Text generiert")
            except Exception as e:
                print(f"❌ Fehler bei {object_id}: {e}")

    # 📘 Ergebnisse speichern
    out_path = "catalog_results.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False)
    print(f"\n📘 Ergebnisse gespeichert in: {out_path}")

# ----------------------------------------------------------
# 🔹 MAIN
if __name__ == "__main__":
    base_folder = input("Pfad zum Basisordner (z. B. Downloads/Objektbilder): ").strip()
    excel_paths = [
        input("Pfad zur 1. Excel-Datei: ").strip(),
        input("Pfad zur 2. Excel-Datei: ").strip()
    ]
    line_ranges = [(601, 900), (501, 1000)]  # Anpassen nach Bedarf
    process_excel_and_images(base_folder, excel_paths, line_ranges)
