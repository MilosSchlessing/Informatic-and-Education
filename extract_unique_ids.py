import pandas as pd

# List of the filenames for the CSV files to be processed.
# These files are assumed to contain object metadata, including their IDs.
files = ['non_empty_501_1000.xlsx - Sheet1.csv', 'non_empty_601_900.xlsx - Sheet1.csv']

# Initialize an empty list to store all object IDs collected from all files.
all_object_ids = []

# ==============================================================================
# 1. Iterate Through Files and Extract IDs
# ==============================================================================

# Loops through each filename in the 'files' list.
for file in files:
    try:
        # Reads the CSV file into a Pandas DataFrame.
        df = pd.read_csv(file)
        
        # Checks if the column 't1' (which is assumed to contain the Object IDs) is present.
        if 't1' in df.columns:
            # Adds the object IDs from the 't1' column to the master list.
            # .tolist() converts the Pandas Series into a standard Python list.
            all_object_ids.extend(df['t1'].tolist())
        else:
            # Prints a warning if the expected ID column is missing.
            print(f"Column 't1' was not found in file: {file}.")
            
    except FileNotFoundError:
        # Handles the case where a file in the list cannot be found in the directory.
        print(f"Error: The file {file} was not found.")
    except Exception as e:
        # Catches any other unexpected errors during file processing (e.g., parsing issues).
        print(f"An error occurred while processing file {file}: {e}")

# ==============================================================================
# 2. Deduplication and Export
# ==============================================================================

# Removes duplicate Object IDs:
# 1. Converts the list to a 'set' to automatically remove duplicates.
# 2. Converts the set back to a 'list'.
# 3. Sorts the list for a clean, ordered output.
unique_object_ids = sorted(list(set(all_object_ids)))

# Creates a new DataFrame from the list of unique IDs.
# The single column is explicitly named 'Object ID'.
unique_object_ids_df = pd.DataFrame(unique_object_ids, columns=['Object ID'])

# Saves the DataFrame containing the unique Object IDs to a new CSV file.
# 'index=False' prevents the DataFrame's internal index from being written to the file.
unique_object_ids_df.to_csv('unique_object_ids.csv', index=False)

# Confirmation message for successful completion.
print("The unique object IDs have been successfully saved to 'unique_object_ids.csv'.")