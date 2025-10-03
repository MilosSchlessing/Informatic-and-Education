import pandas as pd
import os
import shutil

# ==============================================================================
# --- CONFIGURATION (ADJUST FOLDER NAMES HERE) ---
# ==============================================================================
folder_name_1 = '1996'  # Replace with your first source folder name
folder_name_2 = '2024'  # Replace with your second source folder name
target_folder_name = 'final_pictures'  # Replace with your desired target folder name
# --- END OF CONFIGURATION ---

# Automatic path setup based on the script's current directory
script_folder = os.getcwd()
csv_filepath = os.path.join(script_folder, 'cleaned_data.csv')
source_folder_1 = os.path.join(script_folder, folder_name_1)
source_folder_2 = os.path.join(script_folder, folder_name_2)
target_folder = os.path.join(script_folder, target_folder_name)

# --- Initial Folder Check ---
for path, name in [(source_folder_1, folder_name_1), (source_folder_2, folder_name_2)]:
    if not os.path.exists(path):
        print(f"❌ ERROR: The source folder '{name}' was not found. Please check the name.")
        exit()

# Create the target folder if it does not exist
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# ==============================================================================
# Step 1: Extract Object Identifiers from CSV
# ==============================================================================
# A set is used to store unique identifiers, automatically handling deduplication
object_identifiers = set()
try:
    # Read the CSV, skipping malformed lines
    df = pd.read_csv(csv_filepath, on_bad_lines='skip')
    
    # The column 'T13' is assumed to contain file paths/names
    if 'T13' not in df.columns:
        print("❌ ERROR: Column 'T13' not found in the CSV file.")
        exit()

    # Iterate through all non-missing entries in the 'T13' column
    for entry in df['T13'].dropna():
        # Handle multiple paths or file names separated by newlines
        for path_entry in entry.split('\n'):
            # --- THE CRUCIAL FIX ---
            # Replace Windows backslashes with standard forward slashes for cross-platform compatibility
            corrected_path = path_entry.strip().replace('\\', '/')
            # Extract the actual file name from the potentially full path
            file_name = os.path.basename(corrected_path)
            
            if file_name:
                try:
                    # The unique object identifier is assumed to be the first three parts
                    # of the file name, separated by hyphens (e.g., 'A-B-C-D.jpg' -> 'A-B-C')
                    parts = file_name.split('-')
                    identifier = '-'.join(parts[:3])
                    # Store the identifier in lowercase for case-insensitive matching later
                    object_identifiers.add(identifier.lower())
                except IndexError:
                    # Ignore entries that don't follow the expected naming convention
                    pass

except FileNotFoundError:
    print(f"❌ ERROR: The CSV file '{csv_filepath}' was not found.")
    exit()

print(f"{len(object_identifiers)} unique object identifiers extracted from the CSV.")

# ==============================================================================
# Step 2: Search Source Folders and Copy Matching Images
# ==============================================================================
found_images_counter = 0
# A set to prevent copying the same file name multiple times (if it exists in both source folders)
copied_files = set() 

print("Starting copy operation...")
# Iterate over both configured source folders
for source_folder in [source_folder_1, source_folder_2]:
    # Iterate over all files/folders in the source directory
    for file_name in os.listdir(source_folder):
        file_name_lower = file_name.lower()
        
        # Check if the file name starts with any of the required identifiers
        for identifier in object_identifiers:
            if file_name_lower.startswith(identifier):
                # Ensure the file hasn't been copied already
                if file_name not in copied_files:
                    source_path = os.path.join(source_folder, file_name)
                    target_path = os.path.join(target_folder, file_name)
                    
                    # Verify it is actually a file before attempting to copy
                    if os.path.isfile(source_path):
                        # Use shutil.copy2 to preserve metadata (timestamps, etc.)
                        shutil.copy2(source_path, target_path)
                        found_images_counter += 1
                        copied_files.add(file_name)
                        # Break the identifier loop once a match is found for the current file
                        break 

# ==============================================================================
# Step 3: Summary
# ==============================================================================
print("\n" + "="*40)
print("        COPY OPERATION COMPLETED")
print("="*40)
print(f"✅ Successfully copied: {found_images_counter} images.")
print(f"Target folder: {target_folder_name}")
print("="*40)