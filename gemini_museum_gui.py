# -*- coding: utf-8 -*-
"""
Museum Object Description GUI
A user-friendly application for generating object descriptions 
with drag-and-drop functionality, powered by Gemini AI and a CSV data source.
"""
import os
import csv
import time
import random
import pandas as pd
import threading
import queue
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

# GUI Libraries
import customtkinter as ctk
# TkinterDnD is necessary for drag-and-drop functionality
from tkinterdnd2 import DND_FILES, TkinterDnD 
from tkinter import filedialog

# AI and Image Processing Libraries
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# ==============================================================================
# 1. BACKEND LOGIC (Core Processing Components)
# ==============================================================================

@dataclass
class ProcessingConfig:
    """Holds configuration parameters passed from the GUI to the processing thread."""
    input_path: str
    csv_path: str
    output_path: str
    language: str

# The core prompt template remains focused on museum-quality, fact-checked output.
CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to generate a formal public exhibition label (wall text). "
    "Analyze the provided images and use the factual data from the museum database as the primary source of truth. "
    "\n\n**DATABASE INFORMATION (Source of Truth):**"
    "\n- **Object ID:** {object_id}"
    "\n- **Date:** {date}"
    "\n- **Material:** {material}"
    "\n- **Dimensions:** {dimensions}"
    "\n\n**YOUR TASK:**"
    "\n1. Synthesize the database information with the visual evidence to describe the object."
    "\n2. **IMPORTANT:** Do NOT mention the specific measurements or weight (e.g., HxBxT, kg) from the 'Dimensions' field in the final description text, as this data will be displayed separately. You may, however, use qualitative descriptors like 'large', 'compact', or 'heavy'."
    "\n3. Focus exclusively on the object's physical characteristics, context, and significance."
    "\n4. Do NOT contradict the other database information (Date, Material)."
    "\n5. If a database field is marked as 'N/A', do not invent information for it."
    "\n\n**Output must be in {language_name}.**"
    "\n\nProvide your response in the following format:"
    "\n{headline_tag}: [A short, compelling title]"
    "\n{description_tag}: [The formal museum description, approx. 70-90 words, **without stating the exact measurements**.]"
)
LANGUAGE_MAPPING = {"Deutsch": ("TITEL", "BESCHREIBUNG"), "English": ("HEADLINE", "DESCRIPTION")}


def load_and_prepare_data(csv_path: str, log_queue: queue.Queue) -> Dict:
    """Loads CSV metadata, cleans 't1' column, and creates a lookup map."""
    try:
        # Load CSV, fill NaN/empty values with "N/A"
        df = pd.read_csv(csv_path, dtype=str).fillna("N/A")
        data_map = {}
        
        # Data cleaning and key generation logic (A/B-C D -> A-B-C)
        df['cleaned_t1'] = df['t1'].str.strip()
        df['lookup_key'] = df['cleaned_t1'].str.replace('/', '-', regex=False).str.split(' ').str[0]
        df_unique = df.drop_duplicates(subset='lookup_key', keep='first')
        
        # Map relevant metadata to the lookup key
        for _, row in df_unique.iterrows():
            key = row['lookup_key']
            data_map[key] = {
                "material": row.get('T3', 'N/A'), "dimensions": row.get('T5', 'N/A'),
                "date": row.get('T14', 'N/A')
            }
        
        log_queue.put(f"✅ CSV data successfully loaded for {len(data_map)} objects.")
        return data_map
    except Exception as e:
        log_queue.put(f"❌ ERROR processing CSV: {e}")
        return {}

class GeminiProcessor:
    """Handles communication with the Gemini API for content generation."""
    def __init__(self):
        # Load API key from environment (.env file)
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key: raise ValueError("API Key not found in environment variables!")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite-preview-06-17")

    def generate_description(self, image_paths: List[str], object_data: Dict[str, str], language: str) -> Tuple[Optional[str], str]:
        """Sends images and the formatted prompt to the Gemini API."""
        try:
            # 1. Load and prepare images
            images = []
            for p in image_paths:
                img = Image.open(p)
                img.thumbnail((2000, 2000)) # Resize for API efficiency
                images.append(img)
            
            # 2. Format prompt
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language, headline_tag=headline_tag, description_tag=description_tag,
                **object_data # Unpacks object_id, date, material, dimensions
            )
            
            # 3. API Call
            response = self.model.generate_content([prompt, *images])
            
            # 4. Parse response
            return self._parse_response(response.text, headline_tag, description_tag)
        except Exception as e:
            # Return None for headline and the error message for description in case of failure
            return (None, str(e))

    @staticmethod
    def _parse_response(text: str, h_tag: str, d_tag: str) -> Tuple[str, str]:
        """Robustly parses the model's response to extract headline and description."""
        headline, description, in_description = "", "", False
        
        h_prefix = f'{h_tag.upper()}:'
        d_prefix = f'{d_tag.upper()}:'
        
        for line in text.strip().split('\n'):
            stripped = line.strip()
            if not stripped: continue
            
            # Check for Headline
            if stripped.upper().startswith(h_prefix):
                headline = stripped[len(h_prefix):].strip()
                in_description = False
            # Check for Description
            elif stripped.upper().startswith(d_prefix):
                description = stripped[len(d_prefix):].strip()
                in_description = True
            # Continuation of Description
            elif in_description:
                description += " " + stripped
                
        return headline or "Untitled", description.strip() or "Description not available."

def run_processing_logic(config: ProcessingConfig, log_queue: queue.Queue):
    """The main thread function executed separately from the GUI."""
    try:
        # 1. Load Data
        log_queue.put("="*50 + "\nStep 1: Loading and processing CSV data...")
        data_map = load_and_prepare_data(config.csv_path, log_queue)
        if not data_map: return

        # 2. Group Images
        log_queue.put("\n" + "="*50 + "\nStep 2: Scanning image folder and creating groups...")
        image_groups = defaultdict(list)
        for filename in os.listdir(config.input_path):
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    # Group by the first four hyphen-separated parts (e.g., A-B-C-D)
                    object_id = '-'.join(filename.split('-')[:4])
                    image_groups[object_id].append(os.path.join(config.input_path, filename))
                except IndexError: continue
        log_queue.put(f"✅ Found {len(image_groups)} object groups.")

        # 3. AI Processing
        log_queue.put("\n" + "="*50 + "\nStep 3: Generating descriptions with AI...")
        generator = GeminiProcessor()
        total_objects = len(image_groups)
        
        # Open CSV output file
        with open(config.output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["object_id", "headline", "description", "material", "date", "dimensions"])
            
            # Iterate through each unique object group
            for idx, (object_id, image_files) in enumerate(image_groups.items(), 1):
                # Get the shorter lookup key (e.g., A-B-C) for CSV data
                lookup_key = '-'.join(object_id.split('-')[:3])
                object_data = data_map.get(lookup_key, {}) # Fetch metadata
                object_data['object_id'] = object_id # Add full ID for the prompt

                files_to_process = sorted(image_files)[:4] # Use max 4 images
                log_queue.put(f"[{idx}/{total_objects}] Processing ID: {object_id} ({len(files_to_process)} images)...")
                
                # Generate description
                result = generator.generate_description(files_to_process, object_data, config.language)
                
                if result[0] is None: # Error case (result[1] holds the error message)
                    headline, description = "Error", result[1]
                    log_queue.put(f"  -> ❌ Error: {description}")
                else: # Success case
                    headline, description = result
                    log_queue.put(f"  -> ✅ Headline: {headline}")

                # Write result to the output CSV
                writer.writerow([
                    object_id, headline, description,
                    object_data.get('material', 'N/A'), object_data.get('date', 'N/A'),
                    object_data.get('dimensions', 'N/A')
                ])

                # Rate limiting implementation
                if idx % 5 == 0 and idx < total_objects:
                    log_queue.put(f"  ... short pause (25s) to comply with API rate limits ...")
                    time.sleep(25)
        
        log_queue.put("\n" + "="*50 + f"\n✅ Processing finished! Data saved to '{config.output_path}'.")
    except Exception as e:
        log_queue.put(f"\n❌ A CRITICAL ERROR OCCURRED: {e}")
    finally:
        # Signal the GUI that the thread has completed execution
        log_queue.put("FINISHED")


# ==============================================================================
# 2. GRAPHICAL USER INTERFACE (GUI)
# ==============================================================================

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    """The main application window using CustomTkinter and Drag-and-Drop."""
    def __init__(self, *args, **kwargs):
        # Initialize both CTk (GUI) and TkinterDnD (Drag-and-Drop)
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("Museum Object Description Generator")
        self.geometry("800x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.log_queue = queue.Queue() # Queue for passing messages from thread to GUI
        self.processing_thread = None

        # --- Create Widgets ---
        self.create_widgets()
        
        # Start the background log updater
        self.update_log_widget()

    def create_widgets(self):
        """Lays out all GUI elements."""
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # 1. Image Folder Input
        ctk.CTkLabel(main_frame, text="1. Image Folder", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.folder_path_entry = ctk.CTkEntry(main_frame, placeholder_text="Drag folder here or click 'Browse'")
        self.folder_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.folder_path_entry)
        ctk.CTkButton(main_frame, text="Browse...", command=self.browse_folder).pack(anchor="e", padx=10, pady=(0, 20))

        # 2. CSV File Input
        ctk.CTkLabel(main_frame, text="2. CSV Data File", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.csv_path_entry = ctk.CTkEntry(main_frame, placeholder_text="Drag CSV file here or click 'Browse'")
        self.csv_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.csv_path_entry)
        ctk.CTkButton(main_frame, text="Browse...", command=self.browse_csv).pack(anchor="e", padx=10, pady=(0, 20))
        
        # 3. Language Selection
        ctk.CTkLabel(main_frame, text="3. Language", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.language_menu = ctk.CTkOptionMenu(main_frame, values=["Deutsch", "English"])
        self.language_menu.pack(fill="x", padx=10, pady=(0, 20))

        # 4. Start Button
        self.start_button = ctk.CTkButton(main_frame, text="Start Processing", font=("Arial", 16, "bold"), command=self.start_processing)
        self.start_button.pack(fill="x", padx=10, pady=10, ipady=10)

        # 5. Live Log Output
        ctk.CTkLabel(main_frame, text="Live Log", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.log_textbox = ctk.CTkTextbox(main_frame, state="disabled", wrap="word")
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_dnd(self, widget):
        """Registers a widget as a drag-and-drop target."""
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, widget))

    def on_drop(self, event, widget):
        """Handles the file/folder drop event."""
        # Clean path from potential braces added by some OS/DnD implementations
        path = event.data.replace("{", "").replace("}", "")
        widget.delete(0, "end")
        widget.insert(0, path)

    def browse_folder(self):
        """Opens a file dialog to select the image folder."""
        path = filedialog.askdirectory()
        if path:
            self.folder_path_entry.delete(0, "end")
            self.folder_path_entry.insert(0, path)
            
    def browse_csv(self):
        """Opens a file dialog to select the CSV file."""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path_entry.delete(0, "end")
            self.csv_path_entry.insert(0, path)

    def start_processing(self):
        """Validates inputs, prompts for output location, and starts the processing thread."""
        folder_path = self.folder_path_entry.get()
        csv_path = self.csv_path_entry.get()
        
        # Input validation
        if not folder_path or not os.path.isdir(folder_path):
            self.log("❌ Please provide a valid image folder.")
            return
        if not csv_path or not os.path.isfile(csv_path):
            self.log("❌ Please provide a valid CSV file.")
            return
            
        # Prompt user to select the output location
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="descriptions_enriched.csv",
            title="Select location to save the result CSV"
        )
        if not output_path:
            self.log("⚠️ Processing cancelled as no save location was selected.")
            return

        # Create configuration object
        config = ProcessingConfig(
            input_path=folder_path,
            csv_path=csv_path,
            output_path=output_path,
            language=self.language_menu.get()
        )
        
        # Clear log and disable button
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled", text="Processing in progress...")
        
        # Start the backend logic in a separate thread
        self.processing_thread = threading.Thread(
            target=run_processing_logic, 
            args=(config, self.log_queue),
            daemon=True # Daemon thread allows the thread to be killed when the main app closes
        )
        self.processing_thread.start()

    def log(self, message: str):
        """Inserts a message into the log textbox and auto-scrolls."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.yview_moveto(1.0) # Auto-scroll to the bottom
        self.log_textbox.configure(state="disabled")

    def update_log_widget(self):
        """Periodically checks the log queue and updates the GUI."""
        try:
            # Empty the queue of all waiting messages
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message == "FINISHED":
                    # Re-enable the button when the processing thread is done
                    self.start_button.configure(state="normal", text="Start Processing")
                else:
                    self.log(message)
        except queue.Empty:
            pass
        
        # Schedule the next check
        self.after(100, self.update_log_widget)

# ==============================================================================
# 3. START APPLICATION
# ==============================================================================

if __name__ == "__main__":
    # Ensure all required libraries are installed
    try:
        import customtkinter as ctk
        from tkinterdnd2 import TkinterDnD
        import google.generativeai
        import pandas
    except ImportError as e:
        print(f"Missing required library: {e}. Please install all dependencies (customtkinter, tkinterdnd2, google-genai, pandas, pillow, python-dotenv).")
    else:
        # TkinterDnD requires the main application class to inherit from it,
        # which is handled by the App class definition.
        app = App()
        app.mainloop()