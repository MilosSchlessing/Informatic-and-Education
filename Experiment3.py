"""
Object Description Generator using Google Gemini AI (CSV-Enriched Version)
Reads image files, enriches them with data from a CSV based on a shared object ID,
and generates fact-based, museum-quality descriptions.
"""

import os
import csv
import time
import random
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# ==============================================================================
# KONFIGURATIONEN UND PROMPTS
# ==============================================================================

@dataclass
class ProcessingConfig:
    input_path: str
    csv_path: str
    output_csv: str
    language: str = "Deutsch"
    max_image_size: Tuple[int, int] = (2000, 2000)
    rate_limit_batch: int = 5
    rate_limit_delay: float = 25.0

CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to generate a formal public exhibition label (wall text). "
    "Analyze the provided images and use the factual data from the museum database as the primary source of truth. "
    "\n\n**DATABASE INFORMATION (Source of Truth):**"
    "\n- **Object ID:** {object_id}"
    "\n- **Date:** {date}"
    "\n- **Material:** {material}"
    "\n- **Dimensions:** {dimensions}"
    "\n\n**YOUR TASK:**"
    "\n1. Synthesize the database information with the visual evidence from the images."
    "\n2. Focus exclusively on the object's physical characteristics, context, and significance."
    "\n3. **Do NOT contradict the database information.** If an image seems to show something different, assume the database is correct."
    "\n4. If a database field is marked as 'N/A', do not invent information for it."
    "\n\n**Output must be in {language_name}.**"
    "\n\nProvide your response in the following format:"
    "\n{headline_tag}: [A short, compelling title for the museum object, consistent with the provided data.]"
    "\n{description_tag}: [The formal museum description in {language_name}, approx. **70-90 words**. Weave the database facts naturally into a descriptive text about the object's appearance, context, and function.]"
)

LANGUAGE_MAPPING = {"Deutsch": ("TITEL", "BESCHREIBUNG"), "English": ("HEADLINE", "DESCRIPTION")}

# ==============================================================================
# DATENLADE-FUNKTION (MIT DER KORREKTUR)
# ==============================================================================

def load_and_prepare_data(csv_path: str) -> Dict[str, Dict[str, str]]:
    """Loads the CSV, cleans the lookup key, and prepares it for quick lookup."""
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("N/A")
        data_map = {}

        # =================================================================
        # HIER IST DIE KORREKTUR:
        # .str.strip() entfernt führende/folgende Leerzeichen aus jeder ID
        # =================================================================
        df['cleaned_t1'] = df['t1'].str.strip()
        df['lookup_key'] = df['cleaned_t1'].str.replace('/', '-', regex=False).str.split(' ').str[0]
        
        df_unique = df.drop_duplicates(subset='lookup_key', keep='first')
        
        for _, row in df_unique.iterrows():
            key = row['lookup_key']
            data_map[key] = {
                "material": row.get('T3', 'N/A'),
                "dimensions": row.get('T5', 'N/A'),
                "date": row.get('T14', 'N/A')
            }
        print(f"✅ Successfully loaded and prepared data for {len(data_map)} unique objects from CSV.")
        return data_map
    except FileNotFoundError:
        print(f"❌ ERROR: CSV file not found at {csv_path}. Please check the path.")
        return {}
    except Exception as e:
        print(f"❌ ERROR: Could not process CSV file. Reason: {e}")
        return {}


# ==============================================================================
# GENERATOR-KLASSE
# ==============================================================================

class GeminiCaptionGenerator:
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite-preview-06-17"):
        load_dotenv(); genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_object_description(self, image_paths: List[str], object_data: Dict[str, str], language: str) -> Optional[Tuple[str, str]]:
        try:
            images = []
            for p in image_paths:
                img = Image.open(p)
                img.thumbnail((2000, 2000))
                images.append(img)
            if not images: return None
            
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language, headline_tag=headline_tag, description_tag=description_tag,
                object_id=object_data.get('object_id', 'N/A'), date=object_data.get('date', 'N/A'),
                material=object_data.get('material', 'N/A'), dimensions=object_data.get('dimensions', 'N/A')
            )
            
            response = self.model.generate_content([prompt, *images])
            return self._parse_response(response.text, headline_tag, description_tag) if hasattr(response, 'text') else None
        except Exception as e:
            print(f"  Error generating description: {e}")
            return None

    @staticmethod
    def _parse_response(text, h_tag, d_tag):
        lines, h, d = text.strip().split('\n'), "", ""
        for line in lines:
            if line.upper().startswith(f'{h_tag.upper()}:'): h = line[len(h_tag)+1:].strip()
            elif line.upper().startswith(f'{d_tag.upper()}:'): d = line[len(d_tag)+1:].strip()
        return h or "Untitled", d or "Description not available."

# ==============================================================================
# PROZESSOR-KLASSE
# ==============================================================================

class ObjectProcessor:
    def __init__(self, config: ProcessingConfig, data_map: Dict[str, Dict[str, str]]):
        self.config = config
        self.generator = GeminiCaptionGenerator()
        self.data_map = data_map

    def process(self):
        object_groups = self._group_images_by_id(self.config.input_path)
        if not object_groups: return
        print(f"\nFound {len(object_groups)} unique image groups to process.")
        self._process_objects(object_groups)

    @staticmethod
    def _group_images_by_id(directory: str) -> dict:
        image_groups = defaultdict(list)
        for filename in os.listdir(directory):
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    object_id = '-'.join(filename.split('-')[:4])
                    image_groups[object_id].append(os.path.join(directory, filename))
                except IndexError: continue
        return dict(image_groups)

    def _process_objects(self, object_groups: dict):
        with open(self.config.output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["object_id", "headline", "description", "material", "date", "dimensions"])

            for idx, (object_id, image_files) in enumerate(object_groups.items(), 1):
                lookup_key = '-'.join(object_id.split('-')[:3])
                object_data = self.data_map.get(lookup_key, {})
                object_data['object_id'] = object_id

                files_to_process = sorted(image_files)[:4]
                print(f"[{idx}/{len(object_groups)}] Processing ID: {object_id} (using {len(files_to_process)} images)...")
                
                try:
                    result = self.generator.generate_object_description(files_to_process, object_data, self.config.language)
                    headline, description = result if result else ("Untitled", "Fallback.")

                    print(f"  -> Headline: {headline}")
                    writer.writerow([
                        object_id, headline, description,
                        object_data.get('material', 'N/A'),
                        object_data.get('date', 'N/A'),
                        object_data.get('dimensions', 'N/A')
                    ])
                    
                    if idx % self.config.rate_limit_batch == 0 and idx < len(object_groups):
                        print(f"  Pausing for {self.config.rate_limit_delay}s to respect API rate limits...")
                        time.sleep(self.config.rate_limit_delay)
                except Exception as e:
                    print(f"  -> FATAL ERROR for {object_id}: {e}")

# ==============================================================================
# HAUPTPROGRAMM
# ==============================================================================

def get_user_input() -> ProcessingConfig:
    print("\nMUSEUM OBJECT DESCRIPTION GENERATOR (CSV-ENRICHED)")
    print("=" * 70)
    input_path = input("Enter path to the folder with your JPG images: ").strip()
    csv_path = input("Enter path to your CSV data file (e.g., cleaned_data.csv): ").strip()
    output_csv = input("Output CSV filename (default: descriptions_enriched.csv): ").strip() or "descriptions_enriched.csv"
    lang = input("Select language (Deutsch, English) [Deutsch]: ").strip() or "Deutsch"
    
    return ProcessingConfig(input_path=input_path, csv_path=csv_path, output_csv=output_csv, language=lang)

def main():
    try:
        config = get_user_input()
        data_map = load_and_prepare_data(config.csv_path)
        if not data_map: return
        
        print("\n" + "=" * 70 + "\nSTARTING PROCESSING...\n" + "=" * 70)
        processor = ObjectProcessor(config, data_map)
        processor.process()
        print("\n" + "=" * 70 + "\nPROCESSING COMPLETE\n" + "=" * 70)
        print(f"Enriched descriptions saved to: {config.output_csv}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    try:
        import pandas
    except ImportError:
        print("Pandas library not found. Please install it by running: pip install pandas")
    else:
        main()