import pandas as pd
import numpy as np

def extract_501_1000_non_empty_columns():
    input_file = 'Liste2.xls'  # oder .csv
    output_file = 'non_empty_501_1000.xlsx'
    
    try:
        # Versuche verschiedene Encodings für CSV
        if input_file.endswith('.csv'):
            encodings = ['latin1', 'iso-8859-1', 'cp1252', 'utf-8']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(input_file, encoding=encoding)
                    print(f"✅ Erfolg mit Encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                # Fallback: read with error handling
                df = pd.read_csv(input_file, encoding='latin1', errors='ignore')
        else:
            # Für Excel-Dateien
            df = pd.read_excel(input_file)
        
        print(f"📋 Original: {len(df)} Zeilen, {len(df.columns)} Spalten")
        
        # Zeilen 501-1000 extrahieren
        extracted_data = df.iloc[500:1000]
        
        # Spalten filtern: nur die, die nicht komplett leer/NaN sind
        non_empty_columns = []
        empty_columns = []
        
        for column in extracted_data.columns:
            # Prüfe ob die Spalte in den extrahierten Zeilen nicht-leere Werte hat
            column_has_data = (
                not extracted_data[column].isna().all() and 
                not (extracted_data[column].astype(str).str.strip() == '').all() and
                not (extracted_data[column].astype(str).str.strip() == 'nan').all()
            )
            
            if column_has_data:
                non_empty_columns.append(column)
            else:
                empty_columns.append(column)
        
        # Nur nicht-leere Spalten behalten
        filtered_data = extracted_data[non_empty_columns]
        
        # In neue Datei schreiben
        filtered_data.to_excel(output_file, index=False)
        
        print(f"✅ Erfolg: Zeilen 501-1000 mit nicht-leeren Spalten gespeichert")
        print(f"📊 Behalten: {len(non_empty_columns)} nicht-leere Spalten")
        print(f"🗑️  Entfernt: {len(empty_columns)} leere Spalten")
        
        if empty_columns:
            print(f"❌ Entfernte Spalten: {empty_columns}")
        
        print(f"📝 Behaltene Spalten: {non_empty_columns}")
        print(f"🔢 Finale Daten: {len(filtered_data)} Zeilen × {len(filtered_data.columns)} Spalten")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")

if __name__ == "__main__":
    extract_501_1000_non_empty_columns()