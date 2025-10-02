import os
import base64
import pandas as pd
import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# ----------------------------------------------------------
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Kein OPENAI_API_KEY gefunden! Bitte .env prüfen.")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

BASE_PROMPT = (
    "You are a professional museum documentation specialist. "
    "Generate a factual, objective, and academically appropriate catalog entry for an object in a museum collection. "
    "Focus strictly on verifiable characteristics: physical description, materials, construction details, visible features, and any inscriptions or markings. "
    "Do not speculate about the object's function or historical significance unless this information is explicitly provided. "
    "Do not interpret the design or context. "
    "Avoid vague or embellished language. Do not use phrases such as 'possibly', 'suggests', 'may have been used for', etc. "
    "Use a neutral tone and precise, technical vocabulary suitable for museum inventory records."
)

# ----------------------------------------------------------
def generate_catalog_text(image_paths, prompt_text):
    image_contents = []
    for path in image_paths:
        base64_image = encode_image(path)
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}, *image_contents]
        }],
    }

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    try:
        response = requests.post(OPENAI_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        print(f"❌ Fehler bei API-Anfrage: {e}")
        if hasattr(e, "response") and e.response is not None:
            print("📩 Response:", e.response.text)
        return f"❌ API-Fehler: {str(e)}"

# ----------------------------------------------------------
def process_excel(excel_path, base_folder):
    df = pd.read_excel(excel_path)

    # Spalten erkennen
    id_col = next((c for c in ["T1", "t1"] if c in df.columns), None)
    img_col = next((c for c in ["T13", "t13"] if c in df.columns), None)
    possible_context_cols = {
        "hersteller": ["T2", "t2"],
        "beschreibung": ["T5", "t5"],
        "datierung": ["T14", "t14"],
        "standort": ["T8", "t8"]
    }

    if not id_col or not img_col:
        raise ValueError("❌ Erwartete Spalten T1/t1 oder T13/t13 fehlen!")

    # 🔁 Gruppieren nach Objekt-ID
    grouped = df.groupby(id_col)
    results = []

    print(f"\n📄 Verarbeite {len(grouped)} Objekte aus {excel_path} ...\n")

    for object_id, group in grouped:
        try:
            if not isinstance(object_id, str):
                object_id = str(object_id)

            print(f"🔍 Bearbeite Objekt-ID: {object_id}")

            # 🔹 Alle Bildpfade sammeln
            image_fields = group[img_col].dropna().astype(str).tolist()
            image_paths = []
            for field in image_fields:
                clean_field = field.replace("\\\\", "\\").replace("\r", "").strip()
                for line in clean_field.splitlines():
                    filename = os.path.basename(line.replace("\\", "/")).strip()
                    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                        year = next((p for p in object_id.split("/") if p.isdigit() and len(p) == 4), None)
                        full_path = os.path.join(base_folder, year, filename) if year else os.path.join(base_folder, filename)
                        if os.path.exists(full_path):
                            image_paths.append(full_path)
                        else:
                            print(f"⚠️ Bild nicht gefunden: {full_path}")

            if not image_paths:
                print(f"⚠️ Keine gültigen Bilder für {object_id}\n")
                results.append({
                    "Objekt-ID": object_id,
                    "Bilder": "",
                    "Katalogtext": "❌ Kein gültiges Bild gefunden"
                })
                continue

            # 🔹 Kontextdaten aus erster Zeile
            first_row = group.iloc[0]

            def get_value(cols):
                return next((first_row[c] for c in cols if c in df.columns and not pd.isna(first_row[c])), "")

            hersteller = get_value(possible_context_cols["hersteller"])
            beschreibung = get_value(possible_context_cols["beschreibung"])
            datierung = get_value(possible_context_cols["datierung"])
            standort = get_value(possible_context_cols["standort"])

            metadata_context = f"""
**Object ID:** {object_id}
**Manufacturer / Collection:** {hersteller}
**Description / Dimensions:** {beschreibung}
**Date / Era:** {datierung}
**Location / Storage:** {standort}
"""

            custom_prompt = f"{BASE_PROMPT}\n\n{metadata_context}\n\nUse this information to refine your description."

            catalog_text = generate_catalog_text(image_paths, custom_prompt)

            results.append({
                "Objekt-ID": object_id,
                "Bilder": ", ".join([os.path.basename(p) for p in image_paths]),
                "Katalogtext": catalog_text
            })

            print(f"✅ {object_id} → Text generiert\n")

        except Exception as e:
            print(f"❌ Fehler bei {object_id}: {e}")
            results.append({
                "Objekt-ID": object_id,
                "Bilder": "",
                "Katalogtext": f"❌ Fehler: {str(e)}"
            })

    # Speichern
    out_path = "catalog_results_grouped.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False)
    print(f"\n📘 Ergebnisse gespeichert in: {out_path}")

# ----------------------------------------------------------
if __name__ == "__main__":
    excel_path = input("Pfad zur Excel-Datei: ").strip()
    base_folder = input("Pfad zum Basisordner mit Bildern: ").strip()
    process_excel(excel_path, base_folder)
