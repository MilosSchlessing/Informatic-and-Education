import pandas as pd

# Liste der Dateinamen Ihrer CSV-Dateien
files = ['non_empty_501_1000.xlsx - Sheet1.csv', 'non_empty_601_900.xlsx - Sheet1.csv']

all_object_ids = []

# Geht jede Datei in der Liste durch
for file in files:
    try:
        # Liest die CSV-Datei
        df = pd.read_csv(file)
        
        # Überprüft, ob die Spalte 't1' (die die Objekt-IDs enthält) vorhanden ist
        if 't1' in df.columns:
            # Fügt die Objekt-IDs zur Liste hinzu
            all_object_ids.extend(df['t1'].tolist())
        else:
            print(f"Die Spalte 't1' wurde in der Datei {file} nicht gefunden.")
            
    except FileNotFoundError:
        print(f"Fehler: Die Datei {file} wurde nicht gefunden.")
    except Exception as e:
        print(f"Beim Verarbeiten der Datei {file} ist ein Fehler aufgetreten: {e}")

# Entfernt doppelte Objekt-IDs
unique_object_ids = sorted(list(set(all_object_ids)))

# Erstellt einen neuen DataFrame für die eindeutigen Objekt-IDs
unique_object_ids_df = pd.DataFrame(unique_object_ids, columns=['Object ID'])

# Speichert die eindeutigen Objekt-IDs in einer neuen CSV-Datei
unique_object_ids_df.to_csv('unique_object_ids.csv', index=False)

print("Die eindeutigen Objekt-IDs wurden erfolgreich in der Datei 'unique_object_ids.csv' gespeichert.")