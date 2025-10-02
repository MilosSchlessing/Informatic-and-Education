"""
Museum Caption Generator combining Excel/CSV Metadaten and Image Analysis
ANPASSUNG: Sucht nach Bilddateinamen in Excel und verwendet entsprechende Metadaten
"""

import os
import csv
import pandas as pd
import tempfile
import time
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# ==============================================================================
# 1. KONFIGURATIONEN UND PROMPTS
# ==============================================================================

@dataclass
class ProcessingConfig:
    """Configuration for processing"""
    excel_path: str
    image_dir: str
    output_csv: str
    language: str = "Deutsch"
    max_image_size: Tuple[int, int] = (2000, 2000)
    rate_limit_batch: int = 3
    rate_limit_delay: float = 1.5

# Prompt-Vorlage: Fokussiert auf Kurator-Rolle und kombiniert Excel-Daten und Bild
CAPTION_PROMPT_TEMPLATE = (
    "Rolle:\n"
    "Du bist ein hochspezialisierter **Kurator und Texter**, dessen Aufgabe es ist, eine **finale öffentliche Ausstellungsbeschriftung** zu verfassen. Deine Analyse muss **sowohl** das hochgeladene Bild **als auch** die bereitgestellten strukturierten Metadaten berücksichtigen.\n\n"
    "Zielsetzung:\n"
    "Erstelle eine **prägnante, informative und fesselnde Beschreibung** des Objekts, die direkt als Text für eine Museumstafel verwendet werden kann.\n\n"
    "Fokus und Einschränkungen:\n"
    "1. **Priorität:** Die bereitgestellten Metadaten sind **faktisch korrekt und bindend**. Verwende diese Daten als Basis.\n"
    "2. **Länge:** Exakt **70-90 Wörter**.\n"
    "3. **Ton:** Akademisch fundiert, aber **klar und zugänglich** für ein breites Publikum.\n"
    "4. **Ausschluss:** Ignoriere fotografische Merkmale (Belichtung, Bildwinkel).\n\n"
    "Ausgabeformat und Sprache:\n"
    "* Die Ausgabe muss ausschließlich in **{Zielsprache}** erfolgen.\n"
    "* Halte dich strikt an das Format mit den Tags **HEADLINE:** und **DESCRIPTION:**, um die maschinelle Zuordnung zu gewährleisten.\n\n"
    "Bereitgestellte Metadaten (aus Excel):\n"
    "* **Inventarnummer:** {Inventar_ID}\n"
    "* **Künstler / Hersteller:** {Kuenstler_Hersteller}\n"
    "* **Abmessungen / Materialhinweis:** {Abmessungen_Material}\n"
    "* **Datierung/Epoche:** {Datierung_Epoche}\n\n"
    "Anweisung zur Beschreibung (DESCRIPTION):\n"
    "Verfasse einen einzigen Absatz. Nutze die Metadaten, um das Objekt korrekt zu identifizieren. Der Hauptteil muss die **visuelle Analyse des Bildes** (Form, Zustand, Ikonografie, Stilmerkmale) und die **kulturelle/historische Bedeutung** (Funktion, Relevanz) darstellen."
)

# Mapping der Schlüsselwörter für das Parsen und die Sprachauswahl
LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG"),
    "English": ("HEADLINE", "DESCRIPTION"),
    "Polski": ("NAGŁÓWEK", "OPIS"),
    "Lietuvių": ("ANTRAŠTĖ", "APRAŠYMAS")
}

# Fallback captions and unwanted phrases
FALLBACK_CAPTIONS = [
    "a detailed scene with multiple visual elements and characters",
    "a composition featuring people in an environment with various details",
    "a visual narrative showing characters and their surroundings"
]

UNWANTED_PHRASES = [
    "this image shows", "in this scene", "in the image", "the image depicts", "here we see", "this is a",
    "dieses bild zeigt", "in dieser szene", "im bild ist", "die abbildung zeigt", "auf diesem bild", "man sieht hier",
    "ten obraz przedstawia", "na tej scenie", "na zdjęciu", "zdjęcie przedstawia", "tu widzimy", "jest to",
    "šis paveikslas rodo", "šioje scenoje", "nuotraukoje", "nuotrauka vaizduoja", "čia matome", "tai yra"
]

# Spaltenzuordnung (MUSS der Struktur Ihrer CSV entsprechen)
COLUMN_MAP = {
    'Inventar_ID': 't1',
    'Kuenstler_Hersteller': 'T2',
    'Abmessungen_Material': 'T5',
    'Datierung_Epoche': 'T14',
    'Bildpfade': 'T13'  # Wird für die Zuordnung zum Bild verwendet
}

# ==============================================================================
# 2. GENERATOR-KLASSE (API-Kommunikation)
# ==============================================================================

class GeminiCaptionGenerator:
    """Handles caption generation using Google Gemini API"""
    
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite-preview-06-17"):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate_caption(
        self,
        image_path: str,
        metadata: Dict[str, str],
        language: str,
        max_image_size: Tuple[int, int]
    ) -> Optional[Tuple[str, str]]:
        """Generates a caption by combining the image and structured metadata."""
        
        try:
            img = Image.open(image_path)
            
            # Skalierung
            if img.size[0] > max_image_size[0] or img.size[1] > max_image_size[1]:
                img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
            
            # Prompt-Formatierung
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            
            dynamic_prompt = CAPTION_PROMPT_TEMPLATE.format(
                Zielsprache=language,
                Inventar_ID=metadata['Inventar_ID'],
                Kuenstler_Hersteller=metadata['Kuenstler_Hersteller'],
                Abmessungen_Material=metadata['Abmessungen_Material'],
                Datierung_Epoche=metadata['Datierung_Epoche'],
                headline_tag=headline_tag,
                description_tag=description_tag
            )
            
            response = self.model.generate_content(
                [dynamic_prompt, img],
                generation_config={
                    "max_output_tokens": 250,
                    "temperature": 0.4,
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            caption_text = response.text if hasattr(response, 'text') else None
            
            if caption_text and caption_text.strip():
                return self._parse_response(caption_text, headline_tag, description_tag)
            
            return None
            
        except Exception as e:
            print(f"  [Error] API call failed for {Path(image_path).name}: {e}")
            return None
    
    @staticmethod
    def _parse_response(text: str, expected_headline_tag: str, expected_description_tag: str) -> Tuple[str, str]:
        """Parses the response to extract headline and description based on dynamic tags."""
        lines = text.strip().split('\n')
        headline = ""
        description = ""
        
        headline_prefix = f'{expected_headline_tag.upper()}:'
        description_prefix = f'{expected_description_tag.upper()}:'
        
        for line in lines:
            line = line.strip()
            if line.upper().startswith(headline_prefix):
                headline = line[len(headline_prefix):].strip()
            elif line.upper().startswith(description_prefix):
                description = line[len(description_prefix):].strip()
            elif headline and not description:
                description = line
            elif description:
                description += " " + line
        
        headline = GeminiCaptionGenerator._clean_caption(headline) if headline else "Untitled Object"
        description = GeminiCaptionGenerator._clean_caption(description) if description else random.choice(FALLBACK_CAPTIONS)
        
        return headline, description
    
    @staticmethod
    def _clean_caption(text: str) -> str:
        """Cleans and normalizes caption text."""
        caption = text.strip()
        caption_lower = caption.lower()
        
        for phrase in UNWANTED_PHRASES:
            if caption_lower.startswith(phrase):
                original_phrase = caption[caption_lower.find(phrase):caption_lower.find(phrase) + len(phrase)]
                caption = caption[len(original_phrase):].strip()
                caption_lower = caption.lower()
                caption = caption.lstrip(',:.')
        
        caption = caption.rstrip('.,;!?')
        
        return caption if caption and len(caption) >= 10 else "a detailed object description is not possible"
    
    @staticmethod
    def _get_fallback_caption() -> Tuple[str, str]:
        """Returns a fallback caption when API fails."""
        return "Untitled Object", random.choice(FALLBACK_CAPTIONS)


# ==============================================================================
# 3. PROZESSOR-KLASSE (Workflow-Steuerung - ANGEPASST)
# ==============================================================================

class ImageProcessor:
    """Handles data loading, image matching, and processing workflow."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.generator = GeminiCaptionGenerator()
        
    def _load_and_match_data(self) -> List[Tuple[Dict[str, str], str]]:
        """Lädt Daten aus Excel und findet passende Metadaten für jedes Bild."""
        try:
            file_ext = Path(self.config.excel_path).suffix.lower()
            
            if file_ext in ['.xlsx', '.xls']:
                print(f"[INFO] Reading Excel file: {self.config.excel_path}")
                df = pd.read_excel(self.config.excel_path)
            elif file_ext == '.csv':
                print(f"[INFO] Reading CSV file: {self.config.excel_path}")
                try:
                    df = pd.read_csv(self.config.excel_path)
                except UnicodeDecodeError:
                    print("[INFO] UTF-8 decoding failed. Trying 'latin-1' encoding...")
                    df = pd.read_csv(self.config.excel_path, encoding='latin-1')
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Use .xlsx, .xls, or .csv")
            
            df = df.fillna('')
        except Exception as e:
            raise IOError(f"Error reading data from {self.config.excel_path}: {e}")

        matched_data = []
        
        # Erstelle ein Dictionary für schnellen Zugriff auf Excel-Daten basierend auf Dateinamen
        excel_data_by_filename = {}
        
        print(f"-> Found {len(df)} entries in data file.")
        
        # Durchlaufe jede Zeile in der Excel und indexiere nach Dateinamen
        for idx, row in df.iterrows():
            metadata = {k: str(row[v]).strip() for k, v in COLUMN_MAP.items()}
            image_paths_raw = metadata.get('Bildpfade', '')
            
            # Extrahiere alle Dateinamen aus der Bildpfad-Spalte
            for p in image_paths_raw.split('\n'):
                p = p.strip()
                if p:
                    filename = Path(p).name.lower()
                    excel_data_by_filename[filename] = metadata
                    
                    # Füge auch Variante ohne Erweiterung hinzu für flexiblere Suche
                    filename_no_ext = Path(p).stem.lower()
                    if filename_no_ext != filename:
                        excel_data_by_filename[filename_no_ext] = metadata
        
        print(f"-> Indexed {len(excel_data_by_filename)} unique filenames from Excel data.")
        
        # Jetzt durchlaufe alle Bilder im Verzeichnis
        image_files = list(Path(self.config.image_dir).glob('*'))
        image_files = [f for f in image_files if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        print(f"-> Found {len(image_files)} image files in directory.")
        
        matched_count = 0
        unmatched_count = 0
        
        for image_path in image_files:
            image_filename = image_path.name.lower()
            image_stem = image_path.stem.lower()
            
            # Suche nach exaktem Dateinamen oder Stammname in Excel-Daten
            matched_metadata = None
            
            if image_filename in excel_data_by_filename:
                matched_metadata = excel_data_by_filename[image_filename]
                print(f"  [Match {matched_count + 1}] Image: {image_filename} -> ID: {matched_metadata['Inventar_ID']}")
            elif image_stem in excel_data_by_filename:
                matched_metadata = excel_data_by_filename[image_stem]
                print(f"  [Match {matched_count + 1}] Image: {image_filename} (by stem) -> ID: {matched_metadata['Inventar_ID']}")
            
            if matched_metadata:
                matched_data.append((matched_metadata, str(image_path)))
                matched_count += 1
            else:
                unmatched_count += 1
                if unmatched_count <= 5:  # Zeige nur die ersten 5 nicht gefundenen Bilder
                    print(f"  [No Match] Image: {image_filename} - No corresponding data found in Excel")

        print(f"\n[Summary] Matched: {matched_count} | Unmatched: {unmatched_count}")
        return matched_data

    def process(self) -> Tuple[int, int]:
        """Main processing loop."""
        
        # 1. Daten laden und zuordnen
        matched_items = self._load_and_match_data()
        
        if not matched_items:
            print("\n[ERROR] No images could be matched to Excel data. Check filenames and T13 column data.")
            return 0, 0
        
        print(f"\n[INFO] Starting processing for {len(matched_items)} matched items.")
        
        processed = 0
        successful = 0
        
        # 2. CSV-Header vorbereiten und öffnen
        headline_tag_en, description_tag_en = LANGUAGE_MAPPING["English"]
        output_headers = list(COLUMN_MAP.values()) + [headline_tag_en.lower(), description_tag_en.lower()]
        
        with open(self.config.output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(output_headers)
            
            # 3. Hauptschleife - Verarbeite jedes Bild mit seinen Metadaten
            for idx, (metadata, image_path) in enumerate(matched_items, 1):
                inventar_id = metadata['Inventar_ID']
                file_name = Path(image_path).name
                
                print(f"\n[Step {idx}/{len(matched_items)}] ID: {inventar_id} | Image: {file_name}")
                
                try:
                    # Generierung der Beschreibung
                    result = self.generator.generate_caption(
                        image_path=image_path,
                        metadata=metadata,
                        language=self.config.language,
                        max_image_size=self.config.max_image_size
                    )
                    
                    if result is None:
                        headline, caption = self.generator._get_fallback_caption()
                        print("  [Warning] Using fallback caption (API error or empty response).")
                    else:
                        headline, caption = result
                        successful += 1
                    
                    print(f"  [Output] Headline: {headline}")
                    
                    # Schreibe die Originaldaten + die neuen Captions in die CSV
                    output_row = [metadata[k] for k in COLUMN_MAP.keys()] + [headline, caption]
                    writer.writerow(output_row)
                    processed += 1
                    
                    # Rate Limiting
                    if processed % self.config.rate_limit_batch == 0 and processed < len(matched_items):
                        print(f"  [Pause] Pausing for {self.config.rate_limit_delay}s to respect rate limits.")
                        time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    print(f"  [Fatal Error] Could not process {inventar_id}: {e}")
                    # Schreibe die Originaldaten + leere/Fallback-Felder bei Fehler
                    headline, caption = self.generator._get_fallback_caption()
                    output_row = [metadata[k] for k in COLUMN_MAP.keys()] + [headline, caption]
                    writer.writerow(output_row)
                    processed += 1
                    time.sleep(2)  # Längere Pause nach schwerem Fehler

        return processed, successful

# ==============================================================================
# 4. BENUTZEREINGABE UND MAIN-FUNKTION
# ==============================================================================

def get_user_input() -> ProcessingConfig:
    """Gets configuration from user input."""
    print("\nMUSEUM CAPTION GENERATOR (Excel + Images)")
    print("=" * 70)
    print("ANPASSUNG: Sucht nach Bilddateinamen in Excel und verwendet entsprechende Metadaten")
    print("=" * 70)
    
    excel_path_input = input("Enter path to the metadata file (e.g., Liste1 or Liste1.xlsx): ").strip()
    
    # Try to find the file with various extensions
    excel_path = None
    if os.path.exists(excel_path_input):
        excel_path = excel_path_input
    else:
        # Try adding common extensions
        for ext in ['.xlsx', '.xls', '.csv']:
            test_path = excel_path_input + ext
            if os.path.exists(test_path):
                excel_path = test_path
                print(f"[INFO] Found file: {excel_path}")
                break
    
    if not excel_path:
        current_dir = Path.cwd()
        available_files = list(current_dir.glob('*.xlsx')) + list(current_dir.glob('*.xls')) + list(current_dir.glob('*.csv'))
        
        error_msg = f"Metadata file does not exist: {excel_path_input}"
        if available_files:
            error_msg += f"\n\nAvailable files in current directory:"
            for f in available_files[:10]:  # Show first 10 files
                error_msg += f"\n  - {f.name}"
        
        raise FileNotFoundError(error_msg)
    
    # Pfad zum Ordner mit den Bildern
    image_dir = input("Enter path to the folder containing image files: ").strip()
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    
    # Output file
    output_csv = input("Output CSV filename (will include original data + new captions) [default: output_captions.csv]: ").strip() or "output_captions.csv"
    
    # Sprachauswahl
    available_languages = ", ".join(LANGUAGE_MAPPING.keys())
    while True:
        language_input = input(f"Select output language ({available_languages}) [Default: Deutsch]: ").strip() or "Deutsch"
        selected_language = next((lang for lang in LANGUAGE_MAPPING if lang.lower() == language_input.lower()), None)
        
        if selected_language:
            break
        print(f"Ungültige Sprache. Bitte wählen Sie aus: {available_languages}")
        
    return ProcessingConfig(
        excel_path=excel_path,
        image_dir=image_dir,
        output_csv=output_csv,
        language=selected_language
    )


def main():
    """Main entry point."""
    try:
        config = get_user_input()
        
        print("\n" + "=" * 70)
        print("STARTING PROCESSING")
        print("=" * 70)
        print(f"Metadata File: {config.excel_path}")
        print(f"Image Directory: {config.image_dir}")
        print(f"Output Language: {config.language}")
        print(f"Output CSV: {config.output_csv}")
        print(f"Expected Columns: {COLUMN_MAP}")
        print("=" * 70 + "\n")
        
        processor = ImageProcessor(config)
        total, successful = processor.process()
        
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Successfully generated: {successful} descriptions.")
        print(f"Total entries processed (with/without image): {total}")
        print(f"Results saved to: {config.output_csv}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
    except Exception as e:
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    main()