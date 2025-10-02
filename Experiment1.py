"""
Image Caption Generator using Google Gemini AI
Processes images from folders or ZIP files and generates detailed descriptive captions with headlines.
Optimized for multi-lingual, museum-quality object descriptions.
"""

import os
import zipfile
import csv
import tempfile
import time
import random
from pathlib import Path
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
    """Configuration for image processing"""
    input_path: str
    output_csv: str
    language: str = "Deutsch"  # Standardwert für die Sprache
    max_image_size: Tuple[int, int] = (2000, 2000)
    rate_limit_batch: int = 3
    rate_limit_delay: float = 1.5


# Vorlage für den Prompt mit Fokus auf das Objekt (Museumstauglich)
CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to analyze the **physical object** in the uploaded image "
    "to generate a **formal public exhibition label** (wall text). "
    "**Ignore** any photographic elements (lighting, perspective). **Focus exclusively** on the object's material, date, style, and historical function. "
    "The target audience is a general, non-academic museum visitor. "
    "\n\n**Output must be in {language_name}.**"
    "\n\nProvide your response in the following format:"
    "\n{headline_tag}: [A short, compelling title for the museum object, stating material and epoch, max. 5-10 words]"
    "\n{description_tag}: [The formal museum description in {language_name}. The description must be approx. **70-90 words** long and cover: 1. **Visual Identification** (material, technique, main motif). 2. **Context/Origin** (epoch, culture). 3. **Historical or Cultural Significance** (function, relevance). Avoid unnecessary filler text.]"
)

# Mapping der Schlüsselwörter für das Parsen und die Sprachauswahl
LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG"),
    "English": ("HEADLINE", "DESCRIPTION"),
    "Polski": ("NAGŁÓWEK", "OPIS"),
    "Lietuvių": ("ANTRAŠTĖ", "APRAŠYMAS")
}

# Fallback captions for error cases (in English, as a default safety net)
FALLBACK_CAPTIONS = [
    "a detailed scene with multiple visual elements and characters",
    "a composition featuring people in an environment with various details",
    "a visual narrative showing characters and their surroundings",
    "an image depicting figures in a setting with atmospheric elements",
    "a scene with characters, environment, and compositional details"
]

# AKTUELLE ANPASSUNG: Unerwünschte Phrasen in allen vier Sprachen
UNWANTED_PHRASES = [
    # English
    "this image shows", "in this scene", "in the image",
    "the image depicts", "here we see", "this is a",
    # Deutsch
    "dieses bild zeigt", "in dieser szene", "im bild ist",
    "die abbildung zeigt", "auf diesem bild", "man sieht hier",
    # Polski (Polnisch)
    "ten obraz przedstawia", "na tej scenie", "na zdjęciu",
    "zdjęcie przedstawia", "tu widzimy", "jest to",
    # Lietuvių (Litauisch)
    "šis paveikslas rodo", "šioje scenoje", "nuotraukoje",
    "nuotrauka vaizduoja", "čia matome", "tai yra"
]


# ==============================================================================
# GENERATOR-KLASSE
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
        language: str,  # Sprache als Parameter
        max_image_size: Tuple[int, int] = (2000, 2000)
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a headline and detailed caption for an image using Gemini API
        """
        try:
            img = Image.open(image_path)
            
            # Resize if needed
            if img.size[0] > max_image_size[0] or img.size[1] > max_image_size[1]:
                img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
            
            # Prompt dynamisch formatieren
            if language not in LANGUAGE_MAPPING:
                raise ValueError(f"Unsupported language: {language}")
                
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            
            # Den Prompt mit den spezifischen Tags und der Zielsprache füllen
            dynamic_prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language,
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
            
            # Try multiple ways to extract text from response
            caption_text = None
            
            # Method 1: Direct text attribute
            if hasattr(response, 'text'):
                try:
                    caption_text = response.text
                except Exception:
                    pass
            
            # Method 2: Extract from candidates
            if not caption_text and hasattr(response, 'candidates') and response.candidates:
                try:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts_text = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                parts_text.append(part.text)
                        if parts_text:
                            caption_text = ''.join(parts_text)
                except Exception:
                    pass
            
            if caption_text and caption_text.strip():
                # Tags an die Parsing-Funktion übergeben
                return self._parse_response(caption_text, headline_tag, description_tag)
            
            return None
            
        except Exception as e:
            print(f"  Error generating caption: {e}")
            return None
    
    @staticmethod
    def _parse_response(text: str, expected_headline_tag: str, expected_description_tag: str) -> Tuple[str, str]:
        """
        Parse the response to extract headline and description based on expected tags.
        """
        lines = text.strip().split('\n')
        headline = ""
        description = ""
        
        # Tags für das Parsen vorbereiten (Großschreibung und Doppelpunkt)
        headline_prefix = f'{expected_headline_tag.upper()}:'
        description_prefix = f'{expected_description_tag.upper()}:'
        
        for line in lines:
            line = line.strip()
            # Prüfen auf Headline-Tag
            if line.upper().startswith(headline_prefix):
                headline = line[len(headline_prefix):].strip()
            # Prüfen auf Description-Tag
            elif line.upper().startswith(description_prefix):
                description = line[len(description_prefix):].strip()
            elif headline and not description:
                # Fallback für reinen Text direkt nach dem Titel-Tag
                description = line
            elif description:
                # Kontinuierliches Hinzufügen zum Beschreibungstext
                description += " " + line
        
        # Clean both parts
        headline = GeminiCaptionGenerator._clean_caption(headline) if headline else "Untitled Scene"
        description = GeminiCaptionGenerator._clean_caption(description) if description else random.choice(FALLBACK_CAPTIONS)
        
        return headline, description
    
    @staticmethod
    def _clean_caption(text: str) -> str:
        """Clean and normalize caption text"""
        caption = text.strip()
        
        # Remove unwanted phrases
        caption_lower = caption.lower()
        for phrase in UNWANTED_PHRASES:
            if caption_lower.startswith(phrase):
                # Finde den Index der Phrase im Originaltext (Fall-sensitiv)
                original_phrase = caption[caption_lower.find(phrase):caption_lower.find(phrase) + len(phrase)]
                caption = caption[len(original_phrase):].strip()
                caption_lower = caption.lower() # Aktualisiere für weitere Prüfungen
                # Remove leading punctuation
                caption = caption.lstrip(',:.')
        
        # Remove trailing punctuation for descriptions, keep for headlines
        caption = caption.rstrip('.,;!?')
        
        if not caption or len(caption) < 10:
            return "a detailed scene with visual elements"
        
        return caption


# ==============================================================================
# PROZESSOR-KLASSE UND HAUPTPROGRAMM
# ==============================================================================

class ImageProcessor:
    """Handles image processing and caption generation workflow"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.generator = GeminiCaptionGenerator()
    
    def process(self) -> Tuple[int, int]:
        """
        Process all images and generate captions
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract images from ZIP if needed
            image_dir = self._prepare_images(temp_dir)
            
            # Find all image files
            image_files = self._find_images(image_dir)
            
            if not image_files:
                print("No images found to process!")
                return 0, 0
            
            print(f"\nFound {len(image_files)} images to process")
            print(f"Using museum object analysis prompt")
            print(f"Output Language: {self.config.language}\n")
            
            # Process images and write to CSV
            return self._process_images(image_files)
    
    def _prepare_images(self, temp_dir: str) -> str:
        """Extract images from ZIP or return directory path"""
        if zipfile.is_zipfile(self.config.input_path):
            print(f"Extracting ZIP file...")
            with zipfile.ZipFile(self.config.input_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir
        return self.config.input_path
    
    @staticmethod
    def _find_images(directory: str) -> List[str]:
        """Find all image files in directory"""
        image_extensions = {".png", ".jpg", ".jpeg"}
        image_files = []
        
        for root, _, files in os.walk(directory):
            for file_name in files:
                if Path(file_name).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file_name))
        
        return sorted(image_files)
    
    def _process_images(self, image_files: List[str]) -> Tuple[int, int]:
        """Process images and write captions to CSV"""
        processed = 0
        successful = 0
        
        # Die Tags für den CSV-Header holen (Englisch als neutraler Standard)
        headline_tag_en, description_tag_en = LANGUAGE_MAPPING["English"]
        
        with open(self.config.output_csv, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Schreibe den CSV-Header mit neutralen englischen Namen
            writer.writerow([headline_tag_en.lower(), description_tag_en.lower(), "image_file"])
            
            for idx, image_path in enumerate(image_files, 1):
                file_name = Path(image_path).name
                
                try:
                    print(f"[{idx}/{len(image_files)}] Processing {file_name}...")
                    
                    result = self.generator.generate_caption(
                        image_path,
                        language=self.config.language,  # Sprache übergeben
                        max_image_size=self.config.max_image_size
                    )
                    
                    # Use fallback if generation failed
                    if result is None:
                        headline, caption = self._get_fallback_caption()
                        print(f"  Using fallback caption")
                    else:
                        headline, caption = result
                        successful += 1
                    
                    print(f"  Headline: {headline}")
                    print(f"  Caption: {caption}")
                    
                    writer.writerow([headline, caption, file_name])
                    processed += 1
                    
                    # Rate limiting
                    if processed % self.config.rate_limit_batch == 0 and processed < len(image_files):
                        print(f"  Pausing to respect rate limits...")
                        time.sleep(self.config.rate_limit_delay)
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    headline, fallback = self._get_fallback_caption()
                    writer.writerow([headline, fallback, file_name])
                    processed += 1
                    time.sleep(2)
        
        return processed, successful
    
    @staticmethod
    def _get_fallback_caption() -> Tuple[str, str]:
        """Get a random fallback caption with headline"""
        return "Untitled Scene", random.choice(FALLBACK_CAPTIONS)


def get_user_input() -> ProcessingConfig:
    """Get configuration and language selection from user input"""
    print("\nMUSEUM CAPTION GENERATOR")
    print("=" * 70)
    
    # Input path
    input_path = input("Enter path to ZIP file or image folder: ").strip()
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Path does not exist: {input_path}")
    
    
    # Output file
    output_csv = input("Output CSV filename (default: caption.csv): ").strip() or "caption.csv"
    
    # Sprachauswahl
    available_languages = ", ".join(LANGUAGE_MAPPING.keys())
    while True:
        language_input = input(f"Select output language ({available_languages}) [Default: Deutsch]: ").strip() or "Deutsch"
        # Groß-/Kleinschreibung ignorieren
        selected_language = next((lang for lang in LANGUAGE_MAPPING if lang.lower() == language_input.lower()), None)
        
        if selected_language:
            break
        print(f"Ungültige Sprache. Bitte wählen Sie aus: {available_languages}")
        
    return ProcessingConfig(
        input_path=input_path,
        output_csv=output_csv,
        language=selected_language # Sprache in die Konfiguration übergeben
    )


def main():
    """Main entry point"""
    try:
        # Get configuration
        config = get_user_input()
        
        print("\n" + "=" * 70)
        print("STARTING PROCESSING")
        print("=" * 70)
        print(f"Input: {config.input_path}")
        print(f"Output Language: {config.language}")
        print(f"Output: {config.output_csv}")
        print("=" * 70 + "\n")
        
        # Process images
        processor = ImageProcessor(config)
        total, successful = processor.process()
        
        # Display results
        print("\n" + "=" * 70)
        print("PROCESSING COMPLETE")
        print("=" * 70)
        print(f"Successfully processed: {successful}/{total}")
        print(f"Captions saved to: {config.output_csv}")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise


if __name__ == "__main__":
    main()