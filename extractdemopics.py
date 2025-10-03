import os
import shutil
from collections import defaultdict

# ==============================================================================
# 1. KONFIGURATION
# ==============================================================================

# Name des Quellordners (muss neben dem Skript liegen)
SOURCE_FOLDER = "final_pictures"

# Name des Zielordners (wird automatisch erstellt)
DESTINATION_FOLDER = "demo_pictures"

# Anzahl der Objekte, die extrahiert werden sollen
NUMBER_OF_OBJECTS = 10

# ==============================================================================
# 2. SKRIPT-LOGIK (Keine Änderungen hier nötig)
# ==============================================================================

def prepare_demo_files_dynamically():
    """
    Findet die ersten 10 einzigartigen Objekt-IDs in einem Ordner und kopiert
    für jede dieser IDs die ersten 4 Bilder in einen neuen Demo-Ordner.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, SOURCE_FOLDER)
    destination_path = os.path.join(script_dir, DESTINATION_FOLDER)

    if not os.path.isdir(source_path):
        print(f"❌ FEHLER: Der Quellordner '{SOURCE_FOLDER}' wurde nicht gefunden.")
        return

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
        print(f"✅ Zielordner '{DESTINATION_FOLDER}' wurde erstellt.")

    print(f"\nSchritt 1: Scanne '{SOURCE_FOLDER}' nach Objekt-IDs...")

    # Finde alle einzigartigen Objekt-IDs im Ordner
    all_object_ids = set()
    for filename in os.listdir(source_path):
        if filename.lower().endswith(".jpg"):
            try:
                # Extrahiere die ID aus den ersten vier Teilen des Namens
                object_id = '-'.join(filename.split('-')[:4])
                all_object_ids.add(object_id)
            except IndexError:
                # Ignoriere Dateien, die nicht dem Namensschema entsprechen
                continue

    if not all_object_ids:
        print("❌ FEHLER: Keine gültigen Objekt-IDs im Quellordner gefunden.")
        return

    # Sortiere die IDs alphabetisch und wähle die ersten 10 aus
    sorted_ids = sorted(list(all_object_ids))
    ids_for_demo = sorted_ids[:NUMBER_OF_OBJECTS]

    print(f"✅ {len(ids_for_demo)} einzigartige Objekt-IDs für die Demo ausgewählt.")
    print("\nSchritt 2: Kopiere die zugehörigen Bilder...")

    # Hole eine aktuelle Liste aller Dateien für eine effiziente Suche
    all_source_files = os.listdir(source_path)
    total_copied = 0

    # Kopiere die Bilder für die ausgewählten IDs
    for obj_id in ids_for_demo:
        matching_files = sorted([
            f for f in all_source_files
            if f.startswith(obj_id) and f.lower().endswith(".jpg")
        ])
        
        files_to_copy = matching_files[:4]
        
        if files_to_copy:
            print(f"  -> Kopiere {len(files_to_copy)} Bilder für ID '{obj_id}'...")
            for filename in files_to_copy:
                shutil.copy2(
                    os.path.join(source_path, filename),
                    os.path.join(destination_path, filename)
                )
                total_copied += 1
        else:
            print(f"  -> ⚠️ Warnung: Für die ID '{obj_id}' wurden keine Bilder gefunden.")


    print("\n" + "="*50)
    print("✅ KOPIERVORGANG ABGESCHLOSSEN")
    print(f"Insgesamt wurden {total_copied} Bilder in den Ordner '{DESTINATION_FOLDER}' kopiert.")
    print("="*50)

if __name__ == "__main__":
    prepare_demo_files_dynamically()