import pandas as pd
import os
import shutil

# --- BITTE HIER NUR DIE ORDNERNAMEN ANPASSEN ---
ordnername_1 = '1996'  # Ersetze mit deinem ersten Ordnernamen
ordnername_2 = '2024'  # Ersetze mit deinem zweiten Ordnernamen
zielordner_name = 'final_pictures'  # Ersetze mit deinem gewünschten Zielordnernamen
# --- ENDE DER ANPASSUNGEN ---

# Automatisches Setup der Pfade
script_ordner = os.getcwd()
csv_dateipfad = os.path.join(script_ordner, 'cleaned_data.csv')
quellordner_1 = os.path.join(script_ordner, ordnername_1)
quellordner_2 = os.path.join(script_ordner, ordnername_2)
zielordner = os.path.join(script_ordner, zielordner_name)

# --- Kurzer Check, ob die Ordner da sind ---
for pfad, name in [(quellordner_1, ordnername_1), (quellordner_2, ordnername_2)]:
    if not os.path.exists(pfad):
        print(f"❌ FEHLER: Der Quellordner '{name}' wurde nicht gefunden. Bitte Namen prüfen.")
        exit()

if not os.path.exists(zielordner):
    os.makedirs(zielordner)

# Schritt 1: Korrekte Extraktion der Objekt-Identifier aus der CSV
objekt_identifier = set()
try:
    df = pd.read_csv(csv_dateipfad, on_bad_lines='skip')
    if 'T13' not in df.columns:
        print("❌ FEHLER: Spalte 'T13' nicht in der CSV gefunden.")
        exit()

    for eintrag in df['T13'].dropna():
        for pfad in eintrag.split('\n'):
            # --- DER ENTSCHEIDENDE FIX ---
            # Ersetze Windows-Backslashes durch normale Slashes
            korrigierter_pfad = pfad.strip().replace('\\', '/')
            # Extrahiere jetzt den echten Dateinamen
            bildname = os.path.basename(korrigierter_pfad)
            
            if bildname:
                try:
                    teile = bildname.split('-')
                    identifier = '-'.join(teile[:3])
                    objekt_identifier.add(identifier.lower())
                except IndexError:
                    pass

except FileNotFoundError:
    print(f"❌ FEHLER: Die CSV-Datei '{csv_dateipfad}' wurde nicht gefunden.")
    exit()

print(f"{len(objekt_identifier)} einzigartige Objekt-Identifier wurden aus der CSV extrahiert.")

# Schritt 2: Quellordner durchsuchen und alle passenden Bilder kopieren
gefundene_bilder_zaehler = 0
kopierte_dateien = set()

print("Starte Kopiervorgang...")
for quellordner in [quellordner_1, quellordner_2]:
    for dateiname in os.listdir(quellordner):
        dateiname_klein = dateiname.lower()
        for identifier in objekt_identifier:
            if dateiname_klein.startswith(identifier):
                if dateiname not in kopierte_dateien:
                    quellpfad = os.path.join(quellordner, dateiname)
                    zielpfad = os.path.join(zielordner, dateiname)
                    if os.path.isfile(quellpfad):
                        shutil.copy2(quellpfad, zielpfad)
                        gefundene_bilder_zaehler += 1
                        kopierte_dateien.add(dateiname)

# Schritt 3: Zusammenfassung
print("\n" + "="*30)
print("        KOPIERVORGANG ABGESCHLOSSEN")
print("="*30)
print(f"✅ Erfolgreich kopiert: {gefundene_bilder_zaehler} Bilder.")
print("="*30)