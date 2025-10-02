"""
Object Description Generator using Google Gemini AI (Single Folder Version)
Scans a single folder, groups images by a 4-part ID, and uses a maximum of the first 4 images per object.
Includes conservative rate limiting for the free API tier.
"""

import os
import csv
import time
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple
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
    output_csv: str
    language: str = "Deutsch"
    max_image_size: Tuple[int, int] = (2000, 2000)
    # ===================================================================
    # HIER IST DIE ÄNDERUNG: WENIGER ANFRAGEN, LÄNGERE PAUSE
    # Wir machen jetzt nur 5 Anfragen und warten dann 25 Sekunden.
    # Das hält uns sicher unter dem Limit von 15 Anfragen pro Minute.
    rate_limit_batch: int = 5
    rate_limit_delay: float = 25.0
    # ===================================================================


# Der Prompt bleibt unverändert
CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to analyze the **physical object** shown in the uploaded images "
    "to generate a **formal public exhibition label** (wall text). You will be provided with one or more images of the **same object**, potentially from different angles. "
    "Synthesize the information from all images to create a single, comprehensive description. "
    "\n\n**CRITICAL INSTRUCTION: Do NOT invent, guess, or assume information that is not visually evident.** This applies especially to dates, epochs, and specific origins. If the time period cannot be determined from the object's style, material, or construction, then **omit this information**. It is better to provide no date than an incorrect one. Focus strictly on observable facts."
    "\n\n**Ignore** any photographic elements (lighting, perspective, background). **Focus exclusively** on the object's material, style, and potential function. "
    "The target audience is a general, non-academic museum visitor."
    "\n\n**Output must be in {language_name}.**"
    "\n\nProvide your response in the following format:"
    "\n{headline_tag}: [A short, compelling title for the museum object. State the primary material and the object's type or form. Mention an epoch **only if it is clearly identifiable** from the style.]"
    "\n{description_tag}: [The formal museum description in {language_name}, approx. **70-90 words**. Cover: 1. **Visual Identification** (material, technique, main motif). 2. **Stylistic Context** (describe the visual style; mention an epoch or culture only if visually obvious). 3. **Inferred Function** (what the object's form suggests about its use). Avoid filler text and speculation.]"
)

LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG"),
    "English": ("HEADLINE", "DESCRIPTION"),
}


# ==============================================================================
# GENERATOR-KLASSE (Unverändert)
# ==============================================================================

class GeminiCaptionGenerator:
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite-preview-06-17"):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please create a .env file with your key.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_object_description(self, image_paths: List[str], language: str, max_image_size: Tuple[int, int]) -> Optional[Tuple[str, str]]:
        try:
            images = []
            for path in image_paths:
                img = Image.open(path)
                img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
                images.append(img)
            if not images: return None

            headline_tag, description_tag = LANGUAGE_MAPPING.get(language, LANGUAGE_MAPPING["English"])
            prompt = CAPTION_PROMPT_TEMPLATE.format(language_name=language, headline_tag=headline_tag, description_tag=description_tag)
            
            response = self.model.generate_content([prompt, *images], generation_config={"max_output_tokens": 300, "temperature": 0.4})
            
            return self._parse_response(response.text, headline_tag, description_tag) if hasattr(response, 'text') else None
        except Exception as e:
            print(f"  Error generating description: {e}")
            return None

    @staticmethod
    def _parse_response(text: str, headline_tag: str, description_tag: str) -> Tuple[str, str]:
        lines = text.strip().split('\n')
        headline, description = "", ""
        headline_prefix, desc_prefix = f'{headline_tag.upper()}:', f'{description_tag.upper()}:'
        for line in lines:
            if line.upper().startswith(headline_prefix):
                headline = line[len(headline_prefix):].strip()
            elif line.upper().startswith(desc_prefix):
                description = line[len(desc_prefix):].strip()
        return headline or "Untitled Object", description or "Detailed object description."


# ==============================================================================
# PROZESSOR-KLASSE (Mit kleinem Bugfix)
# ==============================================================================

class ObjectProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.generator = GeminiCaptionGenerator()

    def process(self):
        object_groups = self._group_images_by_id(self.config.input_path)
        if not object_groups:
            print("No images found or could not form object groups!")
            return
        print(f"\nFound {len(object_groups)} unique objects to process.")
        print(f"Output Language: {self.config.language}\n")
        self._process_objects(object_groups)

    @staticmethod
    def _group_images_by_id(directory: str) -> dict:
        print(f"Scanning for images in '{directory}'...")
        image_groups = defaultdict(list)
        image_extensions = {".png", ".jpg", ".jpeg"}

        for filename in os.listdir(directory):
            if Path(filename).suffix.lower() in image_extensions:
                try:
                    parts = filename.split('-')
                    object_id = '-'.join(parts[:4])
                    # KORREKTUR: os.path.join statt os.join
                    full_path = os.path.join(directory, filename)
                    image_groups[object_id].append(full_path)
                except IndexError:
                    print(f"  Skipping file with unexpected name format: {filename}")
                    continue
        return dict(image_groups)

    def _process_objects(self, object_groups: dict):
        processed_count = 0
        total_objects = len(object_groups)
        headline_tag, description_tag = LANGUAGE_MAPPING["English"]

        with open(self.config.output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["object_id", headline_tag.lower(), description_tag.lower()])

            for idx, (object_id, image_files) in enumerate(object_groups.items(), 1):
                sorted_files = sorted(image_files)
                files_to_process = sorted_files[:4]

                print(f"[{idx}/{total_objects}] Processing Object ID: {object_id} (using {len(files_to_process)} of {len(image_files)} images)...")
                
                try:
                    result = self.generator.generate_object_description(
                        files_to_process,
                        self.config.language, 
                        self.config.max_image_size
                    )
                    headline, description = result if result else ("Untitled Object", "Fallback description.")
                    if not result: print("  -> Using fallback description.")

                    print(f"  -> Headline: {headline}")
                    writer.writerow([object_id, headline, description])
                    processed_count += 1
                    
                    if processed_count % self.config.rate_limit_batch == 0 and idx < total_objects:
                        print(f"  Pausing for {self.config.rate_limit_delay}s to respect API rate limits...")
                        time.sleep(self.config.rate_limit_delay)
                except Exception as e:
                    print(f"  -> FATAL ERROR for object {object_id}: {e}")
                    time.sleep(2)


# ==============================================================================
# HAUPTPROGRAMM (Unverändert)
# ==============================================================================

def get_user_input() -> ProcessingConfig:
    print("\nMUSEUM OBJECT DESCRIPTION GENERATOR (SINGLE FOLDER)")
    print("=" * 70)
    input_path = input("Enter path to the folder containing all your JPG images: ").strip()
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Directory does not exist: {input_path}")
    output_csv = input("Output CSV filename (default: object_descriptions.csv): ").strip() or "object_descriptions.csv"
    available_languages = ", ".join(LANGUAGE_MAPPING.keys())
    while True:
        lang_input = input(f"Select output language ({available_languages}) [Default: Deutsch]: ").strip() or "Deutsch"
        selected_language = next((lang for lang in LANGUAGE_MAPPING if lang.lower() == lang_input.lower()), None)
        if selected_language: break
        print(f"Invalid language. Please choose from: {available_languages}")
    return ProcessingConfig(input_path=input_path, output_csv=output_csv, language=selected_language)


def main():
    try:
        config = get_user_input()
        print("\n" + "=" * 70 + "\nSTARTING PROCESSING...\n" + "=" * 70)
        ObjectProcessor(config).process()
        print("\n" + "=" * 70 + "\nPROCESSING COMPLETE\n" + "=" * 70)
        print(f"Object descriptions saved to: {config.output_csv}")
    except Exception as e:
        print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()