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
# CONFIGURATIONS AND PROMPTS
# ==============================================================================

@dataclass
class ProcessingConfig:
    """Configuration class for the entire processing workflow."""
    input_path: str                 # Path to the folder containing image files
    csv_path: str                   # Path to the CSV file containing metadata
    output_csv: str                 # Name of the final output CSV
    language: str = "Deutsch"       # Target output language for the descriptions
    max_image_size: Tuple[int, int] = (2000, 2000) # Max dimensions for images sent to the API
    rate_limit_batch: int = 5       # Number of objects to process before pausing
    rate_limit_delay: float = 25.0  # Delay (in seconds) to respect API rate limits

# Prompt template focusing on integrating factual database information
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

# Mapping of keywords for parsing the Gemini response and selecting the target language
LANGUAGE_MAPPING = {"Deutsch": ("TITEL", "BESCHREIBUNG"), "English": ("HEADLINE", "DESCRIPTION")}

# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================

def load_and_prepare_data(csv_path: str) -> Dict[str, Dict[str, str]]:
    """
    Loads metadata from CSV, cleans it, and maps it to a dictionary
    using a standardized lookup key based on the 't1' column.
    """
    try:
        # Read CSV, treating all data as strings and filling missing values with "N/A"
        df = pd.read_csv(csv_path, dtype=str).fillna("N/A")
        data_map = {}
        
        # Clean 't1' column (Object ID)
        df['cleaned_t1'] = df['t1'].astype(str).str.strip()
        
        # Create a standardized lookup key from 't1' (e.g., 'A/B-C D' -> 'A-B-C')
        df['lookup_key'] = (
            df['cleaned_t1']
            .str.replace('/', '-', regex=False)  # Replace slashes with hyphens
            .str.split(' ').str[0]              # Take only the first part before a space
        )
        
        # Drop duplicates based on the lookup key, keeping the first valid entry
        df_unique = df.drop_duplicates(subset='lookup_key', keep='first')
        
        # Populate the final dictionary structure
        for _, row in df_unique.iterrows():
            key = row['lookup_key']
            data_map[key] = {
                # Use .get() for robust access to potentially missing columns
                "material": row.get('T3', 'N/A'),
                "dimensions": row.get('T5', 'N/A'),
                "date": row.get('T14', 'N/A')
            }
        print(f"✅ Successfully loaded data for {len(data_map)} unique objects from CSV.")
        return data_map
    except Exception as e:
        print(f"❌ ERROR processing CSV: {e}")
        return {}

# ==============================================================================
# GENERATOR CLASS (With Robust Parser)
# ==============================================================================

class GeminiCaptionGenerator:
    """Handles communication with the Gemini API for description generation."""
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite-preview-06-17"):
        # Load API key and configure client
        load_dotenv(); genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_object_description(self, image_paths: List[str], object_data: Dict[str, str], language: str) -> Optional[Tuple[str, str]]:
        """
        Formats the prompt with object data, sends images and prompt to Gemini,
        and returns the parsed headline and description.
        """
        try:
            # Load and resize images
            images = []
            for p in image_paths:
                img = Image.open(p)
                img.thumbnail((2000, 2000)) # Resize for efficient API transfer
                images.append(img)
            if not images: return None
            
            # Get language-specific tags
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            
            # Format the prompt with factual data
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language, headline_tag=headline_tag, description_tag=description_tag,
                object_id=object_data.get('object_id', 'N/A'), date=object_data.get('date', 'N/A'),
                material=object_data.get('material', 'N/A'), dimensions=object_data.get('dimensions', 'N/A')
            )
            
            # API Call: send prompt and all images
            response = self.model.generate_content([prompt, *images])
            
            # Parse the response text
            return self._parse_response(response.text, headline_tag, description_tag) if hasattr(response, 'text') else None
        except Exception as e:
            print(f"  Error generating description: {e}")
            return None

    @staticmethod
    def _parse_response(text: str, h_tag: str, d_tag: str) -> Tuple[str, str]:
        """
        Robustly parses the model's response using state tracking to handle
        multi-line descriptions correctly.
        """
        headline, description = "", ""
        in_description = False # State flag

        # Normalize tags for case-insensitive matching
        h_prefix = f'{h_tag.upper()}:'
        d_prefix = f'{d_tag.upper()}:'
        
        for line in text.strip().split('\n'):
            stripped_line = line.strip()
            if not stripped_line: continue

            # 1. Check for headline tag
            if stripped_line.upper().startswith(h_prefix):
                headline = stripped_line[len(h_prefix):].strip()
                in_description = False # Headline prefix resets description state

            # 2. Check for description tag
            elif stripped_line.upper().startswith(d_prefix):
                description = stripped_line[len(d_prefix):].strip()
                in_description = True # Start of a description block
            
            # 3. If currently in a description block, append the line
            elif in_description:
                description += " " + stripped_line

        # Simple cleanup and fallback for empty results
        return headline or "Untitled", description.strip() or "Description not available."

# ==============================================================================
# PROCESSOR CLASS
# ==============================================================================

class ObjectProcessor:
    """Manages the overall workflow: grouping, data lookup, and processing."""
    def __init__(self, config: ProcessingConfig, data_map: Dict[str, Dict[str, str]]):
        self.config = config
        self.generator = GeminiCaptionGenerator()
        self.data_map = data_map

    def process(self):
        """Orchestrates the image grouping and the sequential processing of objects."""
        # Group images based on their shared object ID prefix
        object_groups = self._group_images_by_id(self.config.input_path)
        if not object_groups: 
            print("No image groups found. Check input path and file naming convention.")
            return
        print(f"\nFound {len(object_groups)} unique image groups to process.")
        self._process_objects(object_groups)

    @staticmethod
    def _group_images_by_id(directory: str) -> dict:
        """
        Groups image files based on the first four hyphen-separated parts of the filename.
        Example: 'A-B-C-D-view1.jpg' -> key 'A-B-C-D'
        """
        image_groups = defaultdict(list)
        for filename in os.listdir(directory):
            # Only process common image file types
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    # Assumes object ID is the first four hyphen-separated parts
                    object_id = '-'.join(filename.split('-')[:4])
                    image_groups[object_id].append(os.path.join(directory, filename))
                except IndexError: 
                    # Ignores files that don't match the expected naming format
                    continue 
        return dict(image_groups)

    def _process_objects(self, object_groups: dict):
        """Iterates through grouped objects, fetches data, calls the generator, and saves results."""
        with open(self.config.output_csv, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Write the standardized CSV header
            writer.writerow(["object_id", "headline", "description", "material", "date", "dimensions"])

            for idx, (object_id, image_files) in enumerate(object_groups.items(), 1):
                # Create the shorter lookup key used in the CSV data map (e.g., 'A-B-C')
                lookup_key = '-'.join(object_id.split('-')[:3])
                
                # Fetch relevant metadata, defaulting to an empty dict if not found
                object_data = self.data_map.get(lookup_key, {}).copy()
                # Add the full object ID for the final output
                object_data['object_id'] = object_id

                # Select a maximum of 4 images for processing
                files_to_process = sorted(image_files)[:4]
                print(f"[{idx}/{len(object_groups)}] Processing ID: {object_id} (using {len(files_to_process)} images)...")
                
                try:
                    # Generate the description using up to 4 images and the metadata
                    result = self.generator.generate_object_description(files_to_process, object_data, self.config.language)
                    headline, description = result if result else ("Untitled", "Fallback: Generation failed.")

                    print(f"  -> Headline: {headline}")
                    
                    # Write the result to the output CSV
                    writer.writerow([
                        object_id, headline, description,
                        object_data.get('material', 'N/A'),
                        object_data.get('date', 'N/A'),
                        object_data.get('dimensions', 'N/A')
                    ])
                    
                    # Rate limiting mechanism
                    if idx % self.config.rate_limit_batch == 0 and idx < len(object_groups):
                        print(f"  Pausing for {self.config.rate_limit_delay}s to respect API rate limits...")
                        time.sleep(self.config.rate_limit_delay)
                except Exception as e:
                    # Log a fatal error for this specific object ID
                    print(f"  -> FATAL ERROR for {object_id}: {e}")
                    # You might consider writing a fallback line here too
                    writer.writerow([object_id, "Error", f"Fatal error: {e}", "N/A", "N/A", "N/A"])


# ==============================================================================
# MAIN PROGRAM
# ==============================================================================

def get_user_input() -> ProcessingConfig:
    """Collects necessary configuration and file paths from the user."""
    print("\nMUSEUM OBJECT DESCRIPTION GENERATOR (CSV-ENRICHED)")
    print("=" * 70)
    
    # Input prompts with defaults
    input_path = input("Enter path to the folder with your JPG images: ").strip()
    csv_path = input("Enter path to your CSV data file (e.g., cleaned_data.csv): ").strip()
    output_csv = input("Output CSV filename (default: descriptions_enriched.csv): ").strip() or "descriptions_enriched.csv"
    lang_input = input("Select language (Deutsch, English) [Deutsch]: ").strip() or "Deutsch"
    
    # Normalize language input
    lang = next((l for l in LANGUAGE_MAPPING if l.lower() == lang_input.lower()), "Deutsch")
    
    return ProcessingConfig(input_path=input_path, csv_path=csv_path, output_csv=output_csv, language=lang)

def main():
    """Main execution function."""
    try:
        config = get_user_input()
        
        # Load and prepare metadata before starting the heavy API calls
        data_map = load_and_prepare_data(config.csv_path)
        if not data_map: return
        
        print("\n" + "=" * 70 + "\nSTARTING PROCESSING...\n" + "=" * 70)
        
        processor = ObjectProcessor(config, data_map)
        processor.process()
        
        print("\n" + "=" * 70 + "\nPROCESSING COMPLETE\n" + "=" * 70)
        print(f"Enriched descriptions saved to: {config.output_csv}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check for pandas dependency before starting
    try:
        import pandas
    except ImportError:
        print("Pandas library not found. Please install it by running: pip install pandas")
    else:
        main()