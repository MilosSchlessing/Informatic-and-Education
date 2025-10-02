import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Load the new datasets
df1 = pd.read_csv('Liste1.xls - CSV-Export.csv')
df2 = pd.read_csv('Liste2.xls - CSV-Export.csv')

# --- Data Subsetting and Merging ---

# Subset the dataframes according to the user's request (using 0-based indexing)
# table 1 (df1): rows 601-900 -> index 600 to 899
# table 2 (df2): rows 501-1000 -> index 500 to 999
df1_subset = df1.iloc[600:900]
df2_subset = df2.iloc[500:1000]

# Concatenate the subsets
df = pd.concat([df1_subset, df2_subset], ignore_index=True)

# --- Data Cleaning and Preprocessing ---

# Define the relevant columns and their new, clearer names
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

# Filter the dataframe to only include columns that actually exist in the loaded data
existing_columns = {k: v for k, v in relevant_columns.items() if k in df.columns}
df_filtered = df[list(existing_columns.keys())].copy()
df_filtered.rename(columns=existing_columns, inplace=True)

# Clean 'Year' column
def clean_year(year_str):
    if isinstance(year_str, str):
        years = re.findall(r'\b\d{4}\b', year_str)
        if years:
            return int(years[0])
    return np.nan

df_filtered['Year_Cleaned'] = df_filtered['Year'].apply(clean_year)

# Clean 'Dimensions' to extract Mass, Height, Width, Depth
def extract_dimensions(dim_str):
    if not isinstance(dim_str, str):
        return pd.Series([np.nan, np.nan, np.nan, np.nan], index=['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm'])

    mass_match = re.search(r'Masse:\s*([\d,\.]+)\s*kg', dim_str, re.IGNORECASE)
    mass = float(mass_match.group(1).replace(',', '.')) if mass_match else np.nan

    # Pattern: HxBxT
    h_match = re.search(r'HxBxT:\s*([\d,\.]+) x', dim_str, re.IGNORECASE)
    w_match = re.search(r'x\s*([\d,\.]+) x', dim_str, re.IGNORECASE)
    d_match = re.search(r'x\s*([\d,\.]+)\s*mm', dim_str, re.IGNORECASE)

    height = float(h_match.group(1).replace(',', '.')) if h_match else np.nan
    width = float(w_match.group(1).replace(',', '.')) if w_match else np.nan
    depth = float(d_match.group(1).replace(',', '.')) if d_match else np.nan

    # Pattern: LxBxH (if HxBxT is not found)
    if pd.isna(height) and pd.isna(width) and pd.isna(depth):
        h_match_lbh = re.search(r'LxBxH:\s*[\d,\.]+ x [\d,\.]+ x ([\d,\.]+)\s*mm', dim_str, re.IGNORECASE)
        w_match_lbh = re.search(r'LxBxH:\s*[\d,\.]+ x ([\d,\.]+) x', dim_str, re.IGNORECASE)
        d_match_lbh = re.search(r'LxBxH:\s*([\d,\.]+) x', dim_str, re.IGNORECASE)
        height = float(h_match_lbh.group(1).replace(',', '.')) if h_match_lbh else np.nan
        width = float(w_match_lbh.group(1).replace(',', '.')) if w_match_lbh else np.nan
        depth = float(d_match_lbh.group(1).replace(',', '.')) if d_match_lbh else np.nan
        
    return pd.Series([mass, height, width, depth], index=['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm'])

df_filtered[['Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm']] = df_filtered['Dimensions'].apply(extract_dimensions)

# --- Exploratory Data Analysis & Visualization ---

# 1. Year Distribution
plt.figure(figsize=(10, 6))
df_filtered['Year_Cleaned'].dropna().hist(bins=30, edgecolor='black')
plt.title('Distribution of Manufacturing Years (New Subset)')
plt.xlabel('Year')
plt.ylabel('Number of Objects')
plt.grid(False)
plt.savefig('year_distribution_new.png')

# 2. Top 10 Manufacturers
plt.figure(figsize=(12, 8))
df_filtered['Manufacturer_Cleaned'] = df_filtered['Manufacturer'].astype(str).str.split('\n').str[0]
top_manufacturers = df_filtered['Manufacturer_Cleaned'].value_counts().nlargest(10)
top_manufacturers.sort_values().plot(kind='barh')
plt.title('Top 10 Manufacturers by Number of Objects (New Subset)')
plt.xlabel('Number of Objects')
plt.ylabel('Manufacturer')
plt.tight_layout()
plt.savefig('top_manufacturers_new.png')

# 3. Mass Distribution (Histogram)
plt.figure(figsize=(10, 6))
df_filtered['Mass_kg'].dropna().hist(bins=30, edgecolor='black')
plt.title('Distribution of Object Mass (kg) (New Subset)')
plt.xlabel('Mass (kg)')
plt.ylabel('Frequency')
plt.savefig('mass_distribution_new.png')

# --- Save the Processed Data ---

# Select columns for the final output file
final_columns = [
    'ID', 'Manufacturer_Cleaned', 'Year_Cleaned', 
    'Mass_kg', 'Height_mm', 'Width_mm', 'Depth_mm', 
    'Material', 'Location', 'URL', 'Image_Path'
]
# Ensure all columns exist in the dataframe before trying to save
final_columns_exist = [col for col in final_columns if col in df_filtered.columns]
processed_df = df_filtered[final_columns_exist]
processed_df.to_csv('processed_museum_data_new.csv', index=False)

print("Analysis complete! Three image files and one CSV file have been saved to your directory.")