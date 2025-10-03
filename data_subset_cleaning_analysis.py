import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Data Loading and Subsetting
# ==============================================================================

# Loads the two source CSV files (which appear to be Excel exports)
# into separate Pandas DataFrames.
df1 = pd.read_csv('Liste1.xls - CSV-Export.csv')
df2 = pd.read_csv('Liste2.xls - CSV-Export.csv')

# --- Data Subsetting and Merging ---

# Selects specific row ranges from the loaded DataFrames.
# df1: Rows 601-900 (corresponds to 0-based index positions 600 to 899).
df1_subset = df1.iloc[600:900]
# df2: Rows 501-1000 (corresponds to 0-based index positions 500 to 999).
df2_subset = df2.iloc[500:1000]

# Concatenates the two subsets vertically.
# 'ignore_index=True' ensures the new DataFrame has a continuous index.
df = pd.concat([df1_subset, df2_subset], ignore_index=True)

# --- Data Cleaning and Preprocessing Setup ---

# Defines the relevant source columns (e.g., t1, T2) and
# maps them to clearer, descriptive target names.
relevant_columns = {
    't1': 'ID',
    'T2': 'Manufacturer',
    'T3': 'Material',
    'T5': 'Dimensions',
    'T8': 'Location',
    'T9': 'Department',
    't12': 'URL',
    'T13': 'Image_Path',
    'T14': 'Year'
}

# Filters the DataFrame to only include columns that actually exist in the source data.
# This prevents errors if one of the expected source columns is missing.
existing_columns = {k: v for k, v in relevant_columns.items() if k in df.columns}
df_filtered = df[list(existing_columns.keys())].copy()

# Renames the selected columns to their clearer names.
df_filtered.rename(columns=existing_columns, inplace=True)

# ==============================================================================
# 2. Data Cleaning Functions
# ==============================================================================

def clean_year(year_str):
    """
    Extracts the first 4-digit year found in a string (e.g., '1985' from 'ca. 1985').
    """
    if isinstance(year_str, str):
        # Finds a 4-digit number surrounded by word boundaries (e.g., '2023').
        years = re.findall(r'\b\d{4}\b', year_str)
        if years:
            # Returns the first found year as an integer.
            return int(years[0])
    # Returns NaN (Not a Number) if no year is found or the value isn't a string.
    return np.nan

# Applies the cleaning function to the 'Year' column and stores the result
# in a new, clean column 'Year_Cleaned'.
df_filtered['Year_Cleaned'] = df_filtered['Year'].apply(clean_year)


def extract_dimensions(dim_str):
    """
    Extracts Mass (kg), Height (H), Width (W), and Depth (D) (or Length/Depth)
    from the 'Dimensions' string using specific regex patterns (HxBxT, LxBxH).
    """
    # Returns four NaN values if the input is not a string.
    if not isinstance(dim_str, str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm'])

    # Pattern 1: Extract Mass (kg)
    # Searches for 'Masse:' followed by a number (with comma or period) and 'kg'.
    mass_match = re.search(r'Masse:\s*([\d,\.]+)\s*kg', dim_str, re.IGNORECASE)
    # Converts to float, replacing commas with periods, or sets to NaN if not found.
    mass = float(mass_match.group(1).replace(',', '.')) if mass_match else np.nan

    # Pattern 2: Extract HxBxT (Height, Width, Depth) in mm
    h_match = re.search(r'HxBxT:\s*([\d,\.]+) x', dim_str, re.IGNORECASE)
    w_match = re.search(r'x\s*([\d,\.]+) x', dim_str, re.IGNORECASE)
    d_match = re.search(r'x\s*([\d,\.]+)\s*mm', dim_str, re.IGNORECASE)

    height = float(h_match.group(1).replace(',', '.')) if h_match else np.nan
    width = float(w_match.group(1).replace(',', '.')) if w_match else np.nan
    depth = float(d_match.group(1).replace(',', '.')) if d_match else np.nan

    # Pattern 3: Tries LxBxH (Length, Width, Height) if HxBxT failed
    if pd.isna(height) and pd.isna(width) and pd.isna(depth):
        # In LxBxH, L is often interpreted as Depth, B as Width, and H as Height.
        h_match_lbh = re.search(r'LxBxH:\s*[\d,\.]+ x [\d,\.]+ x ([\d,\.]+)\s*mm', dim_str, re.IGNORECASE)
        w_match_lbh = re.search(r'LxBxH:\s*[\d,\.]+ x ([\d,\.]+) x', dim_str, re.IGNORECASE)
        d_match_lbh = re.search(r'LxBxH:\s*([\d,\.]+) x', dim_str, re.IGNORECASE)

        height = float(h_match_lbh.group(1).replace(',', '.')) if h_match_lbh else height
        width = float(w_match_lbh.group(1).replace(',', '.')) if w_match_lbh else width
        depth = float(d_match_lbh.group(1).replace(',', '.')) if d_match_lbh else depth

    # Returns the extracted values as a Pandas Series to be added as separate columns.
    return pd.Series([mass, height, width, depth], index=['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm'])

# Applies the dimension extraction function and adds the new numerical columns.
df_filtered[['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm']] = df_filtered['Dimensions'].apply(extract_dimensions)

# ==============================================================================
# 3. Exploratory Data Analysis (EDA) & Visualization
# ==============================================================================

# Clean the Manufacturer name: Takes only the first entry (before a newline)
# for a consistent count.
df_filtered['Manufacturer_Cleaned'] = df_filtered['Manufacturer'].astype(str).str.split('\n').str[0]

# --- 3.1 Year Distribution ---
plt.figure(figsize=(10, 6))
# Creates a histogram of the cleaned year data, dropping NaN values.
df_filtered['Year_Cleaned'].dropna().hist(bins=30, edgecolor='black')
plt.title('Distribution of Manufacturing Years (New Subset)')
plt.xlabel('Year')
plt.ylabel('Number of Objects')
plt.grid(False)
# Saves the chart as an image file.
plt.savefig('year_distribution_new.png')

# --- 3.2 Top 10 Manufacturers ---
plt.figure(figsize=(12, 8))
# Counts the occurrences of the cleaned manufacturers and selects the top 10.
top_manufacturers = df_filtered['Manufacturer_Cleaned'].value_counts().nlargest(10)
# Creates a horizontal bar chart, sorting values for better readability.
top_manufacturers.sort_values().plot(kind='barh')
plt.title('Top 10 Manufacturers by Number of Objects (New Subset)')
plt.xlabel('Number of Objects')
plt.ylabel('Manufacturer')
plt.tight_layout() # Adjusts margins to prevent labels from being cut off.
plt.savefig('top_manufacturers_new.png')

# --- 3.3 Mass Distribution ---
plt.figure(figsize=(10, 6))
# Creates a histogram of the cleaned mass values.
df_filtered['Mass_kg'].dropna().hist(bins=30, edgecolor='black')
plt.title('Distribution of Object Mass (kg) (New Subset)')
plt.xlabel('Mass (kg)')
plt.ylabel('Frequency')
plt.savefig('mass_distribution_new.png')

# ==============================================================================
# 4. Save Processed Data
# ==============================================================================

# Defines the columns for the final CSV output.
final_columns = [
    'ID', 'Manufacturer_Cleaned', 'Year_Cleaned', 
    'Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm', 
    'Material', 'Location', 'URL', 'Image_Path'
]

# Ensures only columns that exist in the processed DataFrame are selected.
final_columns_exist = [col for col in final_columns if col in df_filtered.columns]
processed_df = df_filtered[final_columns_exist]

# Saves the cleaned and augmented DataFrame to a new CSV file.
# 'index=False' prevents the DataFrame index from being written to the file.
processed_df.to_csv('processed_museum_data_new.csv', index=False)

# Confirmation message.
print("Analysis complete! Three image files and one CSV file have been saved to your directory.")