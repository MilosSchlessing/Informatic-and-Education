import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

def analyze_and_categorize_collection(file_path):
    """
    Counts unique objects in the CSV and categorizes them by
    primary material and decade.
    """
    print("Starting analysis and categorization...")

    # --- 1. Load and Clean Data ---
    try:
        df = pd.read_csv(file_path, dtype=str).fillna("N/A")
        # Standardize column names
        df.columns = df.columns.str.strip().str.lower()
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{file_path}' was not found.")
        return

    # --- 2. Identify Unique Objects ---
    # Create the unique lookup key from the 't1' column
    df['lookup_key'] = df['t1'].str.strip().str.replace('/', '-', regex=False).str.split(' ').str[0]
    
    # Create a DataFrame containing only the unique objects (first entry for each ID)
    unique_objects_df = df.drop_duplicates(subset='lookup_key', keep='first')
    unique_object_count = len(unique_objects_df)

    print("\n" + "="*50)
    print("      Total Number of Unique Objects")
    print("="*50)
    print(f"✅ Your collection contains {unique_object_count} unique objects.")
    print("="*50)


    # --- 3. Create Categories ---
    
    # --- Category 1: By Primary Material ---
    # Extract the first listed material and capitalize it
    unique_objects_df['material_category'] = unique_objects_df['t3'].str.split(r'[,;]').str[0].str.strip().str.capitalize()
    material_counts = unique_objects_df['material_category'].value_counts().nlargest(10)

    print("\n" + "="*50)
    print("      Categorization by Primary Material (Top 10)")
    print("="*50)
    print(material_counts)
    print("="*50)

    # Visualization for Material
    plt.figure(figsize=(12, 8))
    sns.barplot(x=material_counts.values, y=material_counts.index, palette="crest")
    plt.title('Top 10 Material Categories', fontsize=16)
    plt.xlabel('Number of Unique Objects', fontsize=12)
    plt.ylabel('Material', fontsize=12)
    plt.tight_layout()
    materials_chart_path = 'category_by_material.png'
    plt.savefig(materials_chart_path)
    print(f"\n✅ Chart for material categories saved to '{materials_chart_path}'")


    # --- Category 2: By Decade ---
    def get_decade(date_string):
        if not isinstance(date_string, str): return "Unknown"
        # Find the first 4-digit year
        match = re.search(r'\b(\d{4})\b', date_string)
        if match:
            year = int(match.group(1))
            # Create decade string (e.g., 1940 -> "1940s")
            return f"{(year // 10) * 10}s"
        return "Unknown"

    unique_objects_df['decade_category'] = unique_objects_df['t14'].apply(get_decade)
    decade_counts = unique_objects_df['decade_category'].value_counts().nlargest(15)
    # Exclude 'Unknown' from the top list if it's there, unless it's the only category
    if "Unknown" in decade_counts.index and len(decade_counts) > 1:
        decade_counts = decade_counts.drop("Unknown")
        
    decade_counts = decade_counts.sort_index() # Sort decades chronologically

    print("\n" + "="*50)
    print("        Categorization by Decade")
    print("="*50)
    print(decade_counts)
    print("="*50)

    # Visualization for Decade
    plt.figure(figsize=(14, 7))
    sns.barplot(x=decade_counts.index, y=decade_counts.values, palette="magma")
    plt.title('Object Distribution by Decade', fontsize=16)
    plt.xlabel('Decade', fontsize=12)
    plt.ylabel('Number of Unique Objects', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    decades_chart_path = 'category_by_decade.png'
    plt.savefig(decades_chart_path)
    print(f"✅ Chart for decade categories saved to '{decades_chart_path}'")


# --- Run the Analysis ---
if __name__ == "__main__":
    analyze_and_categorize_collection('cleaned_data.csv')