import pandas as pd
import numpy as np
import os # Import os for potential future path/file checks (though not strictly used here)

def extract_601_900_non_empty_columns():
    """
    Loads data from 'Liste1.xls', extracts rows 601-900, filters out 
    columns that are entirely empty (NaN, empty string, or 'nan' string) 
    within this subset, and saves the result to a new Excel file.
    """
    
    # --- Configuration ---
    input_file = 'Liste1.xls'  # Expected input file name (can be .xls, .xlsx, or .csv)
    output_file = 'non_empty_601_900.xlsx' # Desired output Excel file name
    
    try:
        # ======================================================================
        # 1. Data Loading with Encoding Handling
        # ======================================================================
        
        # Check file extension to determine the correct read function
        if input_file.lower().endswith('.csv'):
            # Try common encodings for CSV files
            encodings = ['latin1', 'iso-8859-1', 'cp1252', 'utf-8']
            df = None
            for encoding in encodings:
                try:
                    # Attempt to read the CSV with the current encoding
                    df = pd.read_csv(input_file, encoding=encoding)
                    print(f"✅ Success with Encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    # Continue to the next encoding if a decode error occurs
                    continue
            if df is None:
                # Fallback: Read with error handling (replacing bad bytes)
                df = pd.read_csv(input_file, encoding='latin1', errors='ignore')
                print("⚠️  Warning: Used Latin1 fallback with error ignoring.")
        
        elif input_file.lower().endswith(('.xls', '.xlsx')):
            # For Excel files, use pd.read_excel
            df = pd.read_excel(input_file)
        
        else:
            # Handle unknown file types
            raise ValueError(f"Unsupported file format: {input_file}. Requires .csv, .xls, or .xlsx.")
            
        print(f"📋 Original Data: {len(df)} rows, {len(df.columns)} columns")
        
        # ======================================================================
        # 2. Subsetting Rows
        # ======================================================================
        
        # Extract rows 601-900 (using 0-based indexing: index 600 up to, but not including, 900)
        extracted_data = df.iloc[600:900]
        
        # ======================================================================
        # 3. Filtering Non-Empty Columns
        # ======================================================================
        
        # Lists to track column status
        non_empty_columns = []
        empty_columns = []
        
        # Iterate over all column names in the subset
        for column in extracted_data.columns:
            # Check for non-empty data in the column subset:
            column_has_data = (
                # 1. Not all values are NaN
                not extracted_data[column].isna().all() and 
                # 2. Not all values, when converted to string and stripped, are empty ('')
                not (extracted_data[column].astype(str).str.strip() == '').all() and
                # 3. Not all values, when converted to string and stripped, are the string 'nan'
                not (extracted_data[column].astype(str).str.strip() == 'nan').all()
            )
            
            if column_has_data:
                non_empty_columns.append(column)
            else:
                empty_columns.append(column)
        
        # Keep only the columns that were found to contain data in the 601-900 range
        filtered_data = extracted_data[non_empty_columns]
        
        # ======================================================================
        # 4. Save Output and Print Summary
        # ======================================================================
        
        # Write the resulting DataFrame to the specified Excel file
        # index=False prevents writing the DataFrame index to the file
        filtered_data.to_excel(output_file, index=False)
        
        print(f"✅ Success: Rows 601-900 with non-empty columns saved to '{output_file}'")
        print(f"📊 Retained: {len(non_empty_columns)} non-empty columns")
        print(f"🗑️  Removed: {len(empty_columns)} empty columns")
        
        if empty_columns:
            print(f"❌ Removed Columns: {empty_columns}")
        
        print(f"📝 Retained Columns: {non_empty_columns}")
        print(f"🔢 Final Data Shape: {len(filtered_data)} rows × {len(filtered_data.columns)} columns")
        
    except Exception as e:
        # Print a detailed error message if any step fails
        print(f"❌ Error during processing: {e}")

if __name__ == "__main__":
    # Execute the main function when the script is run directly
    extract_601_900_non_empty_columns()