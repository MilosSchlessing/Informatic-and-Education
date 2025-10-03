# -*- coding: utf-8 -*-
"""
Museum Object Description & Categorization GUI
A user-friendly application for generating object descriptions and primary categories
with drag-and-drop functionality, powered by Gemini AI and a CSV data source.
"""
import os
import csv
import time
import pandas as pd
import threading
import queue
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

# GUI Libraries
import customtkinter as ctk
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

# =======================================================
# HIER IST DER NEUE, ERWEITERTE PROMPT
# =======================================================
CAPTION_PROMPT_TEMPLATE = (
    "As an expert **Museum Curator and Art Historian**, your task is to generate a formal public exhibition label and assign a primary category. "
    "Analyze the provided images and use the factual data from the museum database as the primary source of truth. "
    "\n\n**DATABASE INFORMATION (Source of Truth):**"
    "\n- **Object ID:** {object_id}"
    "\n- **Date:** {date}"
    "\n- **Material:** {material}"
    "\n- **Dimensions:** {dimensions}"
    "\n\n**YOUR TASK (in two parts):**"
    "\n**Part 1: Generate Wall Text**"
    "\n- Synthesize all information to describe the object."
    "\n- Do NOT mention specific measurements (e.g., HxBxT) in the description text."
    "\n- Do NOT contradict the database information."
    "\n\n**Part 2: Categorize the Object**"
    "\n- After the description, add a category tag."
    "\n- Choose the ONE most fitting category from the following list: [Measurement & Testing, Communication & Transmission, Power & Electrical, Audio/Visual, Data Processing, Mechanical Component, Other]"
    "\n\n**Output must be in {language_name} and in the following format:**"
    "\n{headline_tag}: [A short, compelling title]"
    "\n{description_tag}: [The formal museum description, approx. 70-90 words.]"
    "\n{category_tag}: [Your chosen category from the list]"
)
# Add the new CATEGORY tag to the language mapping
LANGUAGE_MAPPING = {
    "Deutsch": ("TITEL", "BESCHREIBUNG", "KATEGORIE"), 
    "English": ("HEADLINE", "DESCRIPTION", "CATEGORY")
}


def load_and_prepare_data(csv_path: str, log_queue: queue.Queue) -> Dict:
    """Loads CSV metadata, cleans 't1' column, and creates a lookup map."""
    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("N/A")
        data_map = {}
        df['cleaned_t1'] = df['t1'].str.strip()
        df['lookup_key'] = df['cleaned_t1'].str.replace('/', '-', regex=False).str.split(' ').str[0]
        df_unique = df.drop_duplicates(subset='lookup_key', keep='first')
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
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key: raise ValueError("API Key not found in environment variables!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite-preview-06-17")

    def generate_description_and_category(self, image_paths: List[str], object_data: Dict[str, str], language: str) -> Tuple[Optional[str], str, str]:
        """Sends images and the formatted prompt to the Gemini API."""
        try:
            images = []
            for p in image_paths:
                img = Image.open(p)
                img.thumbnail((2000, 2000))
                images.append(img)
            
            headline_tag, description_tag, category_tag = LANGUAGE_MAPPING[language]
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language, headline_tag=headline_tag, description_tag=description_tag, category_tag=category_tag,
                **object_data
            )
            
            response = self.model.generate_content([prompt, *images])
            return self._parse_response(response.text, headline_tag, description_tag, category_tag)
        except Exception as e:
            return (None, str(e), "Error")

    @staticmethod
    def _parse_response(text: str, h_tag: str, d_tag: str, c_tag: str) -> Tuple[str, str, str]:
        """Robustly parses the model's response to extract headline, description, and category."""
        headline, description, category, in_description = "", "", "", False
        
        h_prefix = f'{h_tag.upper()}:'
        d_prefix = f'{d_tag.upper()}:'
        c_prefix = f'{c_tag.upper()}:'
        
        for line in text.strip().split('\n'):
            stripped = line.strip()
            if not stripped: continue
            
            if stripped.upper().startswith(h_prefix):
                headline = stripped[len(h_prefix):].strip()
                in_description = False
            elif stripped.upper().startswith(d_prefix):
                description = stripped[len(d_prefix):].strip()
                in_description = True
            elif stripped.upper().startswith(c_prefix):
                category = stripped[len(c_prefix):].strip()
                in_description = False
            elif in_description:
                description += " " + stripped
                
        return headline or "Untitled", description.strip() or "Description not available.", category or "Uncategorized"

def run_processing_logic(config: ProcessingConfig, log_queue: queue.Queue):
    """The main thread function executed separately from the GUI."""
    try:
        log_queue.put("="*50 + "\nStep 1: Loading and processing CSV data...")
        data_map = load_and_prepare_data(config.csv_path, log_queue)
        if not data_map: return

        log_queue.put("\n" + "="*50 + "\nStep 2: Scanning image folder and creating groups...")
        image_groups = defaultdict(list)
        for filename in os.listdir(config.input_path):
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    object_id = '-'.join(filename.split('-')[:4])
                    image_groups[object_id].append(os.path.join(config.input_path, filename))
                except IndexError: continue
        log_queue.put(f"✅ Found {len(image_groups)} object groups.")

        log_queue.put("\n" + "="*50 + "\nStep 3: Generating descriptions and categories with AI...")
        generator = GeminiProcessor()
        total_objects = len(image_groups)
        
        with open(config.output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            # Add the new category column to the header
            writer.writerow(["object_id", "primary_category", "headline", "description", "material", "date", "dimensions"])
            
            for idx, (object_id, image_files) in enumerate(image_groups.items(), 1):
                lookup_key = '-'.join(object_id.split('-')[:3])
                object_data = data_map.get(lookup_key, {})
                object_data['object_id'] = object_id

                files_to_process = sorted(image_files)[:4]
                log_queue.put(f"[{idx}/{total_objects}] Processing ID: {object_id} ({len(files_to_process)} images)...")
                
                result, description, category = generator.generate_description_and_category(files_to_process, object_data, config.language)
                
                if result is None:
                    headline = "Error"
                    log_queue.put(f"  -> ❌ Error: {description}")
                else:
                    headline = result
                    log_queue.put(f"  -> ✅ Category: {category} | Headline: {headline}")

                writer.writerow([
                    object_id, category, headline, description,
                    object_data.get('material', 'N/A'), object_data.get('date', 'N/A'),
                    object_data.get('dimensions', 'N/A')
                ])

                if idx % 5 == 0 and idx < total_objects:
                    log_queue.put(f"  ... short pause (25s) to comply with API rate limits ...")
                    time.sleep(25)
        
        log_queue.put("\n" + "="*50 + f"\n✅ Processing finished! Data saved to '{config.output_path}'.")
    except Exception as e:
        log_queue.put(f"\n❌ A CRITICAL ERROR OCCURRED: {e}")
    finally:
        log_queue.put("FINISHED")


# ==============================================================================
# 2. GRAPHICAL USER INTERFACE (GUI) - REMAINS LARGELY UNCHANGED
# ==============================================================================

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)
        self.title("Museum Object Description & Categorization")
        self.geometry("800x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.log_queue = queue.Queue()
        self.processing_thread = None
        self.create_widgets()
        self.update_log_widget()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)
        ctk.CTkLabel(main_frame, text="1. Image Folder", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.folder_path_entry = ctk.CTkEntry(main_frame, placeholder_text="Drag folder here or click 'Browse'")
        self.folder_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.folder_path_entry)
        ctk.CTkButton(main_frame, text="Browse...", command=self.browse_folder).pack(anchor="e", padx=10, pady=(0, 20))
        ctk.CTkLabel(main_frame, text="2. CSV Data File", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.csv_path_entry = ctk.CTkEntry(main_frame, placeholder_text="Drag CSV file here or click 'Browse'")
        self.csv_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.csv_path_entry)
        ctk.CTkButton(main_frame, text="Browse...", command=self.browse_csv).pack(anchor="e", padx=10, pady=(0, 20))
        ctk.CTkLabel(main_frame, text="3. Language", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.language_menu = ctk.CTkOptionMenu(main_frame, values=["Deutsch", "English"])
        self.language_menu.pack(fill="x", padx=10, pady=(0, 20))
        self.start_button = ctk.CTkButton(main_frame, text="Start Processing", font=("Arial", 16, "bold"), command=self.start_processing)
        self.start_button.pack(fill="x", padx=10, pady=10, ipady=10)
        ctk.CTkLabel(main_frame, text="Live Log", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.log_textbox = ctk.CTkTextbox(main_frame, state="disabled", wrap="word")
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_dnd(self, widget):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, widget))

    def on_drop(self, event, widget):
        path = event.data.replace("{", "").replace("}", "")
        widget.delete(0, "end")
        widget.insert(0, path)

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path: self.folder_path_entry.delete(0, "end"); self.folder_path_entry.insert(0, path)
            
    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path: self.csv_path_entry.delete(0, "end"); self.csv_path_entry.insert(0, path)

    def start_processing(self):
        folder_path = self.folder_path_entry.get()
        csv_path = self.csv_path_entry.get()
        if not folder_path or not os.path.isdir(folder_path): self.log("❌ Please provide a valid image folder."); return
        if not csv_path or not os.path.isfile(csv_path): self.log("❌ Please provide a valid CSV file."); return
        output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")], initialfile="descriptions_categorized.csv", title="Select location to save the result CSV")
        if not output_path: self.log("⚠️ Processing cancelled."); return

        config = ProcessingConfig(input_path=folder_path, csv_path=csv_path, output_path=output_path, language=self.language_menu.get())
        
        self.log_textbox.configure(state="normal"); self.log_textbox.delete("1.0", "end"); self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled", text="Processing in progress...")
        
        self.processing_thread = threading.Thread(target=run_processing_logic, args=(config, self.log_queue), daemon=True)
        self.processing_thread.start()

    def log(self, message: str):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.yview_moveto(1.0)
        self.log_textbox.configure(state="disabled")

    def update_log_widget(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message == "FINISHED":
                    self.start_button.configure(state="normal", text="Start Processing")
                else:
                    self.log(message)
        except queue.Empty: pass
        self.after(100, self.update_log_widget)

# ==============================================================================
# 3. START APPLICATION
# ==============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()