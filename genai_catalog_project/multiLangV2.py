import os
import re
import base64
import time
import pandas as pd
import requests
from dotenv import load_dotenv

# ==============================
# Config & helpers
# ==============================
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("❌ No OPENAI_API_KEY found! Please check your .env")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
MODEL = "gpt-4o"  # Vision-fähig. Bei Bedarf auf gpt-4o-mini (ohne Vision) NICHT wechseln.

# Sprachen -> Spaltensuffix + "not specified" Phrase
LANG_SETTINGS = {
    "Deutsch":  {"suffix": "DE", "ns": "nicht angegeben"},
    "English":  {"suffix": "EN", "ns": "not specified"},
    "Polski":   {"suffix": "PL", "ns": "nie podano"},
    "Lietuvių": {"suffix": "LT", "ns": "nenurodyta"},
}

# Basis-Instruktion (auf Englisch, damit strikt & klar), aber Output-Sprache wird erzwungen:
BASE_PROMPT = (
    "You are a registrar / documentation specialist in a technical museum. "
    "Create a factual, verifiable catalog entry. "
    "Describe only characteristics explicitly stated in the metadata or clearly visible/identified: "
    "form, materials, construction/components, visible surface features, inscriptions/markings, dimensions/weight, condition traces. "
    "Do not provide interpretations, assumptions, historical context, or inferred functions. "
    "Do not mention functions or uses unless explicitly stated. "
    "Avoid subjective adjectives and speculative language. "
    "Write in precise, neutral museum terminology. "
    "Use only the provided information."
)

# Einheitliches (englisches) Label-Schema – einfach zu parsen, Text ist in Zielsprache
OUTPUT_SCHEMA = (
    "Provide the entry in the following exact label format. "
    "Write the content in {language_name} (the labels remain in English):\n"
    "Title: <text or '{ns}'>\n"
    "Object ID: <value>\n"
    "Manufacturer/Collection: <value or '{ns}'>\n"
    "Date: <value or '{ns}'>\n"
    "Dimensions: <value or '{ns}'>\n"
    "Weight: <value or '{ns}'>\n"
    "Location: <value or '{ns}'>\n"
    "Description: 2–5 sentences. Only observable/stated features. No function, context, or evaluation."
)

# ==============================
# OpenAI call
# ==============================
def generate_catalog_text(image_paths, prompt_text):
    # mehrere Bilder in einem Request
    image_contents = []
    for p in image_paths:
        b64 = encode_image(p)
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}, *image_contents]
        }],
        # kein max_tokens nötig; server-seitig vernünftiger Default
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    for attempt in range(3):
        try:
            resp = requests.post(OPENAI_URL, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                txt = e.response.text
                if "rate_limit_exceeded" in txt or e.response.status_code == 429:
                    wait = 10 * (attempt + 1)
                    print(f"⏳ Rate limit. Waiting {wait}s and retrying...")
                    time.sleep(wait)
                    continue
                print("📩 Response:", txt)
            return f"❌ API Error: {str(e)}"
    return "❌ Failed after retries."

# ==============================
# Parse model output (labels in EN, content in target language)
# ==============================
KNOWN_LABELS = [
    "Title", "Object ID", "Manufacturer/Collection", "Date",
    "Dimensions", "Weight", "Location", "Description"
]
LABEL_RE = re.compile(r"^\s*(" + "|".join(re.escape(l) for l in KNOWN_LABELS) + r")\s*:\s*(.*)$", re.IGNORECASE)

def parse_structured(text: str):
    """
    Erwartet das definierte Label-Format.
    Gibt ein Dict mit allen Feldern zurück.
    Description kann mehrzeilig sein (bis zum Ende).
    """
    result = {k: "" for k in KNOWN_LABELS}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = LABEL_RE.match(lines[i])
        if m:
            label = m.group(1)  # wie im Schema
            value = m.group(2).strip()
            if label.lower() == "description":
                # sammle alle Folgezeilen bis zum nächsten Label bzw. Ende
                desc_lines = [value] if value else []
                i += 1
                while i < len(lines) and not LABEL_RE.match(lines[i]):
                    if lines[i].strip():
                        desc_lines.append(lines[i].strip())
                    i += 1
                result["Description"] = " ".join(desc_lines).strip()
                continue  # schon i weitergeschoben
            else:
                result[label] = value
        i += 1
    return result

# ==============================
# Main Excel processing
# ==============================
def process_excel(excel_path, base_folder, languages):
    df = pd.read_excel(excel_path)

    # Pflichtspalten
    id_col  = next((c for c in ["T1", "t1"]  if c in df.columns), None)
    img_col = next((c for c in ["T13", "t13"] if c in df.columns), None)
    if not id_col or not img_col:
        raise ValueError("❌ Expected columns T1/t1 and T13/t13 not found!")

    # Kontextspalten (optional, wenn vorhanden)
    ctx_map = {
        "title":       ["T3", "t3"],
        "manufacturer":["T2", "t2"],
        "desc":        ["T5", "t5"],   # Maße/Beschreibung
        "date":        ["T14", "t14"],
        "weight":      ["T6", "t6"],
        "location":    ["T8", "t8"],
        "notes":       ["T7", "t7"],
    }

    grouped = df.groupby(id_col)
    results_rows = []

    # Ergebnis-Columns vorbereiten
    base_cols = ["Object ID", "Images"]
    lang_cols = []
    for lang in languages:
        suf = LANG_SETTINGS[lang]["suffix"]
        lang_cols += [
            f"Title_{suf}",
            f"Manufacturer_{suf}",
            f"Date_{suf}",
            f"Dimensions_{suf}",
            f"Weight_{suf}",
            f"Location_{suf}",
            f"Description_{suf}",
        ]
    all_cols = base_cols + lang_cols

    print(f"\n🌍 Languages: {', '.join(languages)}")
    print(f"📄 Processing {len(grouped)} objects from {excel_path} ...\n")

    for object_id, group in grouped:
        object_id = str(object_id).strip() if not isinstance(object_id, str) else object_id.strip()
        if not object_id:
            continue

        print(f"🔍 Object ID: {object_id}")

        # Alle Bildpfade einsammeln
        image_paths = []
        for field in group[img_col].dropna().astype(str).tolist():
            clean = field.replace("\\\\", "\\").replace("\r", "").strip()
            for line in clean.splitlines():
                filename = os.path.basename(line.replace("\\", "/")).strip()
                if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                year = next((p for p in object_id.split("/") if p.isdigit() and len(p) == 4), None)
                full_path = os.path.join(base_folder, year, filename) if year else os.path.join(base_folder, filename)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    print(f"⚠️ Image not found: {full_path}")

        if not image_paths:
            print(f"⚠️ No valid images → skipping\n")
            row = {"Object ID": object_id, "Images": ""}
            for lang in languages:
                suf = LANG_SETTINGS[lang]["suffix"]
                row.update({
                    f"Title_{suf}": "",
                    f"Manufacturer_{suf}": "",
                    f"Date_{suf}": "",
                    f"Dimensions_{suf}": "",
                    f"Weight_{suf}": "",
                    f"Location_{suf}": "",
                    f"Description_{suf}": "❌ No valid image found",
                })
            results_rows.append(row)
            continue

        # Kontext aus erster Zeile der Gruppe
        first = group.iloc[0]

        def pick(cols):
            return next((first[c] for c in cols if c in df.columns and pd.notna(first[c])), "")

        ctx_title        = pick(ctx_map["title"])
        ctx_manufacturer = pick(ctx_map["manufacturer"])
        ctx_desc         = pick(ctx_map["desc"])
        ctx_date         = pick(ctx_map["date"])
        ctx_weight       = pick(ctx_map["weight"])
        ctx_location     = pick(ctx_map["location"])
        ctx_notes        = pick(ctx_map["notes"])

        # Ergebnis-Zeile initialisieren
        row = {"Object ID": object_id, "Images": ", ".join(os.path.basename(p) for p in image_paths)}

        # Für jede Sprache separat generieren und parsen
        for lang in languages:
            ns = LANG_SETTINGS[lang]["ns"]
            suf = LANG_SETTINGS[lang]["suffix"]

            # fehlende Felder sprachspezifisch ersetzen (nur im Prompt-Kontext)
            title_p   = ctx_title if str(ctx_title).strip() else ns
            manuf_p   = ctx_manufacturer if str(ctx_manufacturer).strip() else ns
            date_p    = ctx_date if str(ctx_date).strip() else ns
            dim_p     = ctx_desc if str(ctx_desc).strip() else ns
            weight_p  = ctx_weight if str(ctx_weight).strip() else ns
            loc_p     = ctx_location if str(ctx_location).strip() else ns
            notes_p   = ctx_notes if str(ctx_notes).strip() else ns

            metadata_context = (
                f"Object ID: {object_id}\n"
                f"Title: {title_p}\n"
                f"Manufacturer/Collection: {manuf_p}\n"
                f"Date: {date_p}\n"
                f"Dimensions/Description: {dim_p}\n"
                f"Weight: {weight_p}\n"
                f"Location: {loc_p}\n"
                f"Additional Notes: {notes_p}\n"
            )
            prompt = (
                f"{BASE_PROMPT}\n\n"
                f"Write the content in {lang}.\n\n"
                f"{metadata_context}\n\n"
                f"{OUTPUT_SCHEMA.format(language_name=lang, ns=ns)}"
            )

            text = generate_catalog_text(image_paths, prompt)
            parsed = parse_structured(text)

            # in Ergebnis-Zeile setzen
            row.update({
                f"Title_{suf}":        parsed.get("Title", ""),
                f"Manufacturer_{suf}": parsed.get("Manufacturer/Collection", ""),
                f"Date_{suf}":         parsed.get("Date", ""),
                f"Dimensions_{suf}":   parsed.get("Dimensions", ""),
                f"Weight_{suf}":       parsed.get("Weight", ""),
                f"Location_{suf}":     parsed.get("Location", ""),
                f"Description_{suf}":  parsed.get("Description", ""),
            })

            print(f"   ✅ {lang} done")

            # kleine Pause reduziert Rate-Limit-Risiko
            time.sleep(1.5)

        results_rows.append(row)
        print("")

    # DataFrame & speichern
    df_out = pd.DataFrame(results_rows, columns=all_cols)
    out_path = "catalog_results_multilang.xlsx"
    df_out.to_excel(out_path, index=False)
    print(f"\n📘 Results saved to: {out_path}")

# ==============================
# Entry point
# ==============================
if __name__ == "__main__":
    excel_path = input("Path to Excel file: ").strip()
    base_folder = input("Path to base folder with images: ").strip()
    langs_in = input("Languages (comma-separated) [Deutsch, English, Polski, Lietuvių]: ").strip()

    if not langs_in:
        languages = ["English"]
    else:
        chosen = [s.strip() for s in langs_in.split(",")]
        # validieren & normalisieren
        valid = []
        for x in chosen:
            # versuche exakte Keys oder case-insensitive Match
            if x in LANG_SETTINGS:
                valid.append(x)
            else:
                for k in LANG_SETTINGS.keys():
                    if x.lower() == k.lower():
                        valid.append(k)
                        break
        if not valid:
            print("⚠️ No valid languages recognized. Falling back to English.")
            valid = ["English"]
        languages = valid

    process_excel(excel_path, base_folder, languages)