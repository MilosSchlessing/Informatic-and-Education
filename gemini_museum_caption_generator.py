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
# CONFIGURATIONS AND PROMPTS
# ==============================================================================

@dataclass
class ProcessingConfig:
    """Configuration class for image processing and output settings."""
    input_path: str                 # Path to the source (ZIP file or directory)
    output_csv: str                 # Name of the output CSV file
    language: str = "Deutsch"       # Target language for the output captions
    max_image_size: Tuple[int, int] = (2000, 2000) # Max size for resizing images before sending to API
    rate_limit_batch: int = 3       # Number of images to process before pausing
    rate_limit_delay: float = 1.5   # Delay (in seconds) between batches


# Prompt template focusing on museum-quality object description
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

# Mapping of keywords for parsing the Gemini response and selecting the target language
LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG"),
    "English": ("HEADLINE", "DESCRIPTION"),
    "Polski": ("NAGŁÓWEK", "OPIS"),
    "Lietuvių": ("ANTRAŠTĖ", "APRAŠYMAS")
}

# Fallback captions for error cases (used as a default safety net)
FALLBACK_CAPTIONS = [
    "a detailed scene with multiple visual elements and characters",
    "a composition featuring people in an environment with various details",
    "a visual narrative showing characters and their surroundings",
    "an image depicting figures in a setting with atmospheric elements",
    "a scene with characters, environment, and compositional details"
]

# Unwanted phrases to be removed from the start of the generated captions in all four supported languages
UNWANTED_PHRASES = [
    # English
    "this image shows", "in this scene", "in the image",
    "the image depicts", "here we see", "this is a",
    # Deutsch
    "dieses bild zeigt", "in dieser szene", "im bild ist",
    "die abbildung zeigt", "auf diesem bild", "man sieht hier",
    # Polski (Polish)
    "ten obraz przedstawia", "na tej scenie", "na zdjęciu",
    "zdjęcie przedstawia", "tu widzimy", "jest to",
    # Lietuvių (Lithuanian)
    "šis paveikslas rodo", "šioje scenoje", "nuotraukoje",
    "nuotrauka vaizduoja", "čia matome", "tai yra"
]


# ==============================================================================
# GENERATOR CLASS
# ==============================================================================

class GeminiCaptionGenerator:
    """Handles caption generation using the Google Gemini API."""
    
    def __init__(self, model_name: str = "models/gemini-2.5-flash-lite-preview-06-17"):
        # Load API key from a .env file
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        # Configure the Gemini client
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def generate_caption(
        self,
        image_path: str,
        language: str,  # Target language passed as a parameter
        max_image_size: Tuple[int, int] = (2000, 2000)
    ) -> Optional[Tuple[str, str]]:
        """
        Generates a headline and detailed caption for an image using the Gemini API.
        """
        try:
            # 1. Image Preprocessing
            img = Image.open(image_path)
            
            # Resize image if its dimensions exceed the maximum size
            if img.size[0] > max_image_size[0] or img.size[1] > max_image_size[1]:
                # Use LANCZOS resampling for high-quality downsampling
                img.thumbnail(max_image_size, Image.Resampling.LANCZOS)
            
            # 2. Dynamic Prompt Formatting
            if language not in LANGUAGE_MAPPING:
                raise ValueError(f"Unsupported language: {language}")
                
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            
            # Fill the template with specific language tags and target language name
            dynamic_prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language,
                headline_tag=headline_tag,
                description_tag=description_tag
            )
            
            # 3. API Call
            response = self.model.generate_content(
                [dynamic_prompt, img],
                generation_config={
                    "max_output_tokens": 250, # Set max output length
                    "temperature": 0.4,       # Lower temperature for more formal/factual output
                    "top_p": 0.9,
                    "top_k": 40
                }
            )
            
            # 4. Extract Raw Text from Response (robust handling)
            caption_text = None
            
            # Method 1: Direct text attribute (most common)
            if hasattr(response, 'text'):
                try:
                    caption_text = response.text
                except Exception:
                    pass
            
            # Method 2: Extract from candidates structure (fallback for robust error handling)
            if not caption_text and hasattr(response, 'candidates') and response.candidates:
                try:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts_text = [part.text for part in candidate.content.parts if hasattr(part, 'text')]
                        if parts_text:
                            caption_text = ''.join(parts_text)
                except Exception:
                    pass
            
            if caption_text and caption_text.strip():
                # Pass the language-specific tags to the parsing function
                return self._parse_response(caption_text, headline_tag, description_tag)
            
            return None
            
        except Exception as e:
            # Log any exceptions during image opening, resizing, or API communication
            print(f"  Error generating caption: {e}")
            return None
    
    @staticmethod
    def _parse_response(text: str, expected_headline_tag: str, expected_description_tag: str) -> Tuple[str, str]:
        """
        Parses the raw API response text to extract the headline and description
        based on the expected language-specific tags.
        """
        lines = text.strip().split('\n')
        headline = ""
        description = ""
        
        # Prepare case-insensitive, colon-appended tags for parsing
        headline_prefix = f'{expected_headline_tag.upper()}:'
        description_prefix = f'{expected_description_tag.upper()}:'
        
        for line in lines:
            line = line.strip()
            # Check for Headline tag
            if line.upper().startswith(headline_prefix):
                # Extract text after the tag
                headline = line[len(headline_prefix):].strip()
            # Check for Description tag
            elif line.upper().startswith(description_prefix):
                # Extract text after the tag
                description = line[len(description_prefix):].strip()
            # Fallback: If a headline was found but no description tag yet, treat the next line as description.
            elif headline and not description:
                description = line
            # Continue appending if description has already started (multi-line response)
            elif description:
                description += " " + line
        
        # Clean both extracted parts
        headline = GeminiCaptionGenerator._clean_caption(headline) if headline else "Untitled Scene"
        description = GeminiCaptionGenerator._clean_caption(description) if description else random.choice(FALLBACK_CAPTIONS)
        
        return headline, description
    
    @staticmethod
    def _clean_caption(text: str) -> str:
        """Cleans and normalizes caption text by removing unwanted phrases and punctuation."""
        caption = text.strip()
        
        # Remove unwanted leading phrases (case-insensitive)
        caption_lower = caption.lower()
        for phrase in UNWANTED_PHRASES:
            if caption_lower.startswith(phrase):
                # Find the index of the phrase in the original text to maintain case
                original_phrase = caption[caption_lower.find(phrase):caption_lower.find(phrase) + len(phrase)]
                # Slice the original text to remove the phrase
                caption = caption[len(original_phrase):].strip()
                caption_lower = caption.lower() # Update for further checks
                # Remove leading punctuation after phrase removal
                caption = caption.lstrip(',:.')
        
        # Remove trailing punctuation (commas, periods, etc.)
        caption = caption.rstrip('.,;!?')
        
        # Fallback if cleaning resulted in too short a string
        if not caption or len(caption) < 10:
            return "a detailed scene with visual elements"
        
        return caption


# ==============================================================================
# PROCESSOR CLASS AND MAIN PROGRAM
# ==============================================================================

class ImageProcessor:
    """Handles the overall image processing and caption generation workflow."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.generator = GeminiCaptionGenerator()
    
    def process(self) -> Tuple[int, int]:
        """
        Main method to process all images and generate captions.
        """
        # Use a temporary directory for