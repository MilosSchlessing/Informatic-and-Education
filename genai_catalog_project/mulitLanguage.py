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
# 🔹 ENV laden
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ Kein OPENAI_API_KEY gefunden! Bitte .env prüfen.")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# ----------------------------------------------------------
# 🔹 Sprach-Mapping für Titel & Beschreibung
LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG"),
    "English": ("HEADLINE", "DESCRIPTION"),
    "Polski": ("NAGŁÓWEK", "OPIS"),
    "Lietuvių": ("ANTRAŠTĖ", "APRAŠYMAS")
}

# ----------------------------------------------------------
# 🔹 Prompt-Vorlage für mehrsprachige Museumstexte
CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to analyze the **physical object** in the uploaded image(s) "
    "and generate a **formal catalog text**.\n\n"
    "Ignore any photographic elements (lighting, background, perspective). Focus exclusively on the object's material, shape, color, function, and historical context.\n"
    "Write in a **clear, academic, and museum-suitable tone**.\n\n"
    "**Output must be in {language_name}.**\n\n"
    "Provide your response in the following format:\n"
    "{headline_tag}: [A short, clear title (5–10 words) summarizing material, type, or period.]\n"
    "{description_tag}: [A well-structured catalog description in {language_name}, about 80–120 words. "
    "Include: (1) Physical characteristics & materials, (2) Function or usage, (3) Historical/cultural context.]"
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
        "model": "gpt-4o",
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
def process_excel(excel_path, base_folder, language_choice):
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

    # Gruppieren nach Objekt-ID
    grouped = df.groupby(id_col)
    results = []

    language_name = language_choice.strip().capitalize()
    if language_name not in LANGUAGE_MAPPING:
        raise ValueError(f"❌ Sprache '{language_name}' nicht unterstützt. Verfügbar: {list(LANGUAGE_MAPPING.keys())}")

    headline_tag, description_tag = LANGUAGE_MAPPING[language_name]

    print(f"\n🌍 Generiere Katalogtexte in: {language_name}")
    print(f"📄 Verarbeite {len(grouped)} Objekte aus {excel_path} ...\n")

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
Object ID: {object_id}
Manufacturer / Collection: {hersteller}
Description / Dimensions: {beschreibung}
Date / Era: {datierung}
Location / Storage: {standort}
"""

            # 🔹 Finalen Prompt bauen
            custom_prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language_name,
                headline_tag=headline_tag,
                description_tag=description_tag
            )
            full_prompt = f"{custom_prompt}\n\nAdditional context:\n{metadata_context}"

            catalog_text = generate_catalog_text(image_paths, full_prompt)

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

    out_path = f"catalog_results_{language_name}.xlsx"
    pd.DataFrame(results).to_excel(out_path, index=False)
    print(f"\n📘 Ergebnisse gespeichert in: {out_path}")

# ----------------------------------------------------------
if __name__ == "__main__":
    excel_path = input("Pfad zur Excel-Datei: ").strip()
    base_folder = input("Pfad zum Basisordner mit Bildern: ").strip()
    language_choice = input("🌍 Sprache wählen (Deutsch / English / Polski / Lietuvių): ").strip()
    process_excel(excel_path, base_folder, language_choice)
