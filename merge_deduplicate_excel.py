import pandas as pd
from pathlib import Path

# --- Configuration Section ---

# 1. Corrected list of input Excel filenames.
# These files are expected to be the output from the previous filtering steps.
input_files = [
    'non_empty_501_1000.xlsx',
    'non_empty_601_900.xlsx'
]

# 2. Name of the final output file (saved as a CSV).
output_filename = 'cleaned_data.csv'

# 3. Name of the column that uniquely identifies the object (the key for grouping/deduplication).
id_column = 't1'

# --- Script Logic ---

# Get the directory where the script is located (for relative file paths).
script_dir = Path(__file__).parent
# Construct the full path for the output file.
output_file_path = script_dir / output_filename

# Initialize an empty list to hold DataFrames loaded from the input files.
all_dataframes = []

print("Starting the data cleaning and merging process...")

# ==============================================================================
# Step 1: Load All Excel Files
# ==============================================================================
for file_name in input_files:
    input_file_path = script_dir / file_name
    try:
        # Load the data using pd.read_excel() since the input files are .xlsx.
        df = pd.read_excel(input_file_path)
        all_dataframes.append(df)
        print(f"✔️ Successfully loaded file '{file_name}' ({len(df)} rows).")
    except FileNotFoundError:
        print(f"⚠️ ERROR: File not found -> '{file_name}'. Skipping this file.")
    except Exception as e:
        print(f"An error occurred while reading file {file_name}: {e}")

# Check if any data was successfully loaded before proceeding.
if not all_dataframes:
    print("No data was loaded. Script terminating.")
else:
    # ==============================================================================
    # Step 2: Combine All Data into One DataFrame
    # ==============================================================================
    # Concatenate all DataFrames in the list vertically.
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nTotal number of rows before cleaning: {len(combined_df)}")

    # ==============================================================================
    # Step 3: Deduplicate Based on ID Column
    # ==============================================================================
    if id_column not in combined_df.columns:
        print(f"⚠️ ERROR: The specified ID column '{id_column}' could not be found.")
    else:
        # Drop rows where the unique ID is missing (required for effective grouping).
        combined_df.dropna(subset=[id_column], inplace=True)
        
        print(f"Deduplicating based on the ID column '{id_column}'...")
        
        # Group by the ID column and aggregate using the 'first' non-NaN value found
        # for each column within the group. 'as_index=False' keeps 't1' as a column.
        cleaned_df = combined_df.groupby(id_column, as_index=False).agg('first')

        # ==============================================================================
        # Step 4: Save the Result as CSV
        # ==============================================================================
        # Write the deduplicated DataFrame to the output CSV file.
        cleaned_df.to_csv(output_file_path, index=False, encoding='utf-8')

        print(f"\nTotal number of rows after deduplication: {len(cleaned_df)}")
        print(f"✅ Success! The cleaned data has been saved to '{output_filename}'.")