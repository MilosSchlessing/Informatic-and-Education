# -*- coding: utf-8 -*-
"""
Museum Object Description GUI
Eine benutzerfreundliche Anwendung zur Erstellung von Objektbeschreibungen
mit Drag-and-Drop, basierend auf Gemini AI und einer CSV-Datenquelle.
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

# GUI-Bibliotheken
import customtkinter as ctk
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import filedialog

# KI und Bildverarbeitung
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# ==============================================================================
# 1. BACKEND-LOGIK (Unsere bisherigen Skript-Teile)
# ==============================================================================

@dataclass
class ProcessingConfig:
    input_path: str
    csv_path: str
    output_path: str
    language: str

# Der Prompt und das Sprach-Mapping bleiben gleich
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
    "\n3. **Do NOT contradict the database information.**"
    "\n4. If a database field is marked as 'N/A', do not invent information for it."
    "\n\n**Output must be in {language_name}.**"
    "\n\nProvide your response in the following format:"
    "\n{headline_tag}: [A short, compelling title]"
    "\n{description_tag}: [The formal museum description, approx. 70-90 words.]"
)
LANGUAGE_MAPPING = {"Deutsch": ("TITEL", "BESCHREIBUNG"), "English": ("HEADLINE", "DESCRIPTION")}

def load_and_prepare_data(csv_path: str, log_queue: queue.Queue) -> Dict:
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
        log_queue.put(f"✅ CSV-Daten erfolgreich für {len(data_map)} Objekte geladen.")
        return data_map
    except Exception as e:
        log_queue.put(f"❌ FEHLER beim Verarbeiten der CSV: {e}")
        return {}

class GeminiProcessor:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key: raise ValueError("API Key nicht gefunden!")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-2.5-flash-lite-preview-06-17")

    def generate_description(self, image_paths, object_data, language):
        try:
            images = []
            for p in image_paths:
                img = Image.open(p)
                img.thumbnail((2000, 2000))
                images.append(img)
            
            headline_tag, description_tag = LANGUAGE_MAPPING[language]
            prompt = CAPTION_PROMPT_TEMPLATE.format(
                language_name=language, headline_tag=headline_tag, description_tag=description_tag,
                **object_data
            )
            response = self.model.generate_content([prompt, *images])
            return self._parse_response(response.text, headline_tag, description_tag)
        except Exception as e:
            # Gib den Fehler zurück, damit er im Log angezeigt werden kann
            return (None, str(e))

    @staticmethod
    def _parse_response(text, h_tag, d_tag):
        headline, description, in_description = "", "", False
        for line in text.strip().split('\n'):
            stripped = line.strip()
            if not stripped: continue
            if stripped.upper().startswith(f'{h_tag.upper()}:'):
                headline = stripped[len(h_tag)+1:].strip()
                in_description = False
            elif stripped.upper().startswith(f'{d_tag.upper()}:'):
                description = stripped[len(d_tag)+1:].strip()
                in_description = True
            elif in_description:
                description += " " + stripped
        return headline or "Untitled", description.strip() or "Description not available."

def run_processing_logic(config: ProcessingConfig, log_queue: queue.Queue):
    try:
        # 1. Daten laden
        log_queue.put("="*50 + "\nSchritt 1: Lade und verarbeite CSV-Datei...")
        data_map = load_and_prepare_data(config.csv_path, log_queue)
        if not data_map: return

        # 2. Bilder gruppieren
        log_queue.put("\n" + "="*50 + "\nSchritt 2: Scanne Bilder-Ordner und erstelle Gruppen...")
        image_groups = defaultdict(list)
        for filename in os.listdir(config.input_path):
            if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png"}:
                try:
                    object_id = '-'.join(filename.split('-')[:4])
                    image_groups[object_id].append(os.path.join(config.input_path, filename))
                except IndexError: continue
        log_queue.put(f"✅ {len(image_groups)} Objektgruppen gefunden.")

        # 3. KI-Verarbeitung
        log_queue.put("\n" + "="*50 + "\nSchritt 3: Generiere Beschreibungen mit der KI...")
        generator = GeminiProcessor()
        total_objects = len(image_groups)
        
        with open(config.output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["object_id", "headline", "description", "material", "date", "dimensions"])
            
            for idx, (object_id, image_files) in enumerate(image_groups.items(), 1):
                lookup_key = '-'.join(object_id.split('-')[:3])
                object_data = data_map.get(lookup_key, {})
                object_data['object_id'] = object_id

                files_to_process = sorted(image_files)[:4]
                log_queue.put(f"[{idx}/{total_objects}] Verarbeite ID: {object_id} ({len(files_to_process)} Bilder)...")
                
                result = generator.generate_description(files_to_process, object_data, config.language)
                
                if result[0] is None: # Fehlerfall
                    headline, description = "Error", result[1]
                    log_queue.put(f"  -> ❌ Fehler: {description}")
                else:
                    headline, description = result
                    log_queue.put(f"  -> ✅ Headline: {headline}")

                writer.writerow([
                    object_id, headline, description,
                    object_data.get('material', 'N/A'), object_data.get('date', 'N/A'),
                    object_data.get('dimensions', 'N/A')
                ])

                if idx % 5 == 0 and idx < total_objects:
                    log_queue.put(f"  ... kurze Pause (25s) zur Einhaltung der API-Limits ...")
                    time.sleep(25)
        
        log_queue.put("\n" + "="*50 + f"\n✅ Verarbeitung abgeschlossen! Daten in '{config.output_path}' gespeichert.")
    except Exception as e:
        log_queue.put(f"\n❌ EIN KRITISCHER FEHLER IST AUFGETRETEN: {e}")
    finally:
        log_queue.put("FINISHED")


# ==============================================================================
# 2. GRAFISCHE BENUTZEROBERFLÄCHE (GUI)
# ==============================================================================

class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)

        self.title("Museum Object Description Generator")
        self.geometry("800x700")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.log_queue = queue.Queue()
        self.processing_thread = None

        # --- Widgets erstellen ---
        self.create_widgets()
        
        # Starte den Log-Updater
        self.update_log_widget()

    def create_widgets(self):
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        # Eingabefelder
        ctk.CTkLabel(main_frame, text="1. Bilder-Ordner", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.folder_path_entry = ctk.CTkEntry(main_frame, placeholder_text="Ordner hierher ziehen oder auf 'Durchsuchen' klicken")
        self.folder_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.folder_path_entry)
        ctk.CTkButton(main_frame, text="Durchsuchen...", command=self.browse_folder).pack(anchor="e", padx=10, pady=(0, 20))

        ctk.CTkLabel(main_frame, text="2. CSV-Datendatei", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.csv_path_entry = ctk.CTkEntry(main_frame, placeholder_text="CSV-Datei hierher ziehen oder auf 'Durchsuchen' klicken")
        self.csv_path_entry.pack(fill="x", padx=10, pady=(0, 5))
        self.setup_dnd(self.csv_path_entry)
        ctk.CTkButton(main_frame, text="Durchsuchen...", command=self.browse_csv).pack(anchor="e", padx=10, pady=(0, 20))
        
        # Einstellungen
        ctk.CTkLabel(main_frame, text="3. Sprache", font=("Arial", 14, "bold")).pack(anchor="w", padx=10)
        self.language_menu = ctk.CTkOptionMenu(main_frame, values=["Deutsch", "English"])
        self.language_menu.pack(fill="x", padx=10, pady=(0, 20))

        # Start-Button
        self.start_button = ctk.CTkButton(main_frame, text="Verarbeitung starten", font=("Arial", 16, "bold"), command=self.start_processing)
        self.start_button.pack(fill="x", padx=10, pady=10, ipady=10)

        # Log-Ausgabe
        ctk.CTkLabel(main_frame, text="Live-Protokoll", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.log_textbox = ctk.CTkTextbox(main_frame, state="disabled", wrap="word")
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=10)

    def setup_dnd(self, widget):
        widget.drop_target_register(DND_FILES)
        widget.dnd_bind('<<Drop>>', lambda e: self.on_drop(e, widget))

    def on_drop(self, event, widget):
        # Entfernt eventuelle geschweifte Klammern, die bei manchen Systemen auftreten
        path = event.data.replace("{", "").replace("}", "")
        widget.delete(0, "end")
        widget.insert(0, path)

    def browse_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.folder_path_entry.delete(0, "end")
            self.folder_path_entry.insert(0, path)
            
    def browse_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path_entry.delete(0, "end")
            self.csv_path_entry.insert(0, path)

    def start_processing(self):
        folder_path = self.folder_path_entry.get()
        csv_path = self.csv_path_entry.get()
        
        if not folder_path or not os.path.isdir(folder_path):
            self.log("❌ Bitte einen gültigen Bilder-Ordner angeben.")
            return
        if not csv_path or not os.path.isfile(csv_path):
            self.log("❌ Bitte eine gültige CSV-Datei angeben.")
            return
            
        output_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="descriptions_enriched.csv",
            title="Speicherort für die Ergebnis-CSV wählen"
        )
        if not output_path:
            self.log("⚠️ Verarbeitung abgebrochen, da kein Speicherort gewählt wurde.")
            return

        config = ProcessingConfig(
            input_path=folder_path,
            csv_path=csv_path,
            output_path=output_path,
            language=self.language_menu.get()
        )
        
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

        self.start_button.configure(state="disabled", text="Verarbeitung läuft...")
        
        # Starte die Verarbeitung in einem separaten Thread, um die GUI nicht zu blockieren
        self.processing_thread = threading.Thread(
            target=run_processing_logic, 
            args=(config, self.log_queue),
            daemon=True
        )
        self.processing_thread.start()

    def log(self, message):
        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", message + "\n")
        self.log_textbox.yview_moveto(1.0) # Auto-scroll
        self.log_textbox.configure(state="disabled")

    def update_log_widget(self):
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if message == "FINISHED":
                    self.start_button.configure(state="normal", text="Verarbeitung starten")
                else:
                    self.log(message)
        except queue.Empty:
            pass
        
        # Überprüfe alle 100ms erneut
        self.after(100, self.update_log_widget)

# ==============================================================================
# 3. ANWENDUNG STARTEN
# ==============================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()