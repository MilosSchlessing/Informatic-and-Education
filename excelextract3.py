import pandas as pd
from pathlib import Path

# --- Konfiguration ---

# 1. Korrigierte Liste der Excel-Dateinamen
input_files = [
    'non_empty_501_1000.xlsx',
    'non_empty_601_900.xlsx'
]

# 2. Name der Ausgabedatei (bleibt eine CSV)
output_filename = 'cleaned_data.csv'

# 3. Name der Spalte, die das Objekt eindeutig identifiziert
id_column = 't1'

# --- Skript-Logik ---

script_dir = Path(__file__).parent
output_file_path = script_dir / output_filename

all_dataframes = []

print("Beginne mit dem Bereinigungsprozess...")

# Schritt 1: Alle Excel-Dateien laden
for file_name in input_files:
    input_file_path = script_dir / file_name
    try:
        # HIER IST DIE ÄNDERUNG: pd.read_excel() statt pd.read_csv()
        df = pd.read_excel(input_file_path)
        all_dataframes.append(df)
        print(f"✔️ Datei '{file_name}' erfolgreich geladen ({len(df)} Zeilen).")
    except FileNotFoundError:
        print(f"⚠️ FEHLER: Datei nicht gefunden -> '{file_name}'.")
    except Exception as e:
        print(f"Ein Fehler ist beim Lesen der Datei {file_name} aufgetreten: {e}")

if not all_dataframes:
    print("Es wurden keine Daten geladen. Das Skript wird beendet.")
else:
    # Schritt 2: Alle Daten zu einem DataFrame kombinieren
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\\nGesamtzahl der Zeilen vor der Bereinigung: {len(combined_df)}")

    # Schritt 3: Duplikate zusammenfassen
    if id_column not in combined_df.columns:
        print(f"⚠️ FEHLER: Die ID-Spalte '{id_column}' konnte nicht gefunden werden.")
    else:
        combined_df.dropna(subset=[id_column], inplace=True)
        print(f"Fasse Duplikate basierend auf der ID '{id_column}' zusammen...")
        cleaned_df = combined_df.groupby(id_column, as_index=False).agg('first')

        # Schritt 4: Das Ergebnis als CSV speichern
        cleaned_df.to_csv(output_file_path, index=False, encoding='utf-8')

        print(f"\\nGesamtzahl der Zeilen nach der Bereinigung: {len(cleaned_df)}")
        print(f"✅ Erfolg! Die bereinigten Daten wurden in '{output_filename}' gespeichert.")