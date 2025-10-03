import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud, STOPWORDS

# ==============================================================================
# CONFIGURATION
# ==============================================================================
CSV_FILE_PATH = 'descriptions_categorized.csv'
# English stopwords for the Word Cloud
STOPWORDS_EN = set(STOPWORDS)
STOPWORDS_EN.update([
    "object", "component", "device", "unit", "part", "system", "collection",
    "design", "construction", "features", "likely", "suggests", "use",
    "application", "applications", "designed", "function", "purpose"
])


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_year(date_str):
    """Extracts the first year from a date string like '1920 - 1940' or 'c. 1995'."""
    if not isinstance(date_str, str):
        return None
    # Find all four-digit numbers in the string
    numbers = re.findall(r'\b\d{4}\b', date_str)
    if numbers:
        # Return the first year found as an integer
        return int(numbers[0])
    return None

def plot_and_save(plot_function, filename, title):
    """Wrapper to create, style, and save plots."""
    plt.figure(figsize=(12, 8))
    plot_function()
    plt.title(title, fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Plot saved as: {filename}")


# ==============================================================================
# MAIN ANALYSIS SCRIPT
# ==============================================================================
def analyze_data():
    """Runs a comprehensive analysis of the categorized descriptions CSV."""
    
    # --- 1. Load and Prepare Data ---
    print("="*60)
    print("1. Loading and inspecting the dataset...")
    print("="*60)
    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: The file '{CSV_FILE_PATH}' was not found.")
        return

    print("Dataset Info:")
    df.info()
    print("\nFirst 5 rows of data:")
    print(df.head())

    # --- 2. Category Analysis ---
    print("\n" + "="*60)
    print("2. Analyzing object categories...")
    print("="*60)
    category_counts = df['primary_category'].value_counts()
    print("Number of objects per category:")
    print(category_counts)
    
    # Plot for category distribution
    def plot_categories():
        sns.barplot(x=category_counts.index, y=category_counts.values, palette='viridis')
        plt.ylabel('Number of Objects', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        plt.xticks(rotation=45, ha='right')
    plot_and_save(plot_categories, '1_category_distribution.png', 'Distribution of Objects by Category')

    # --- 3. Date & Timeline Analysis ---
    print("\n" + "="*60)
    print("3. Analyzing the distribution over time...")
    print("="*60)
    df['year'] = df['date'].apply(extract_year)
    df_time = df.dropna(subset=['year'])
    print(f"{len(df_time)} of {len(df)} objects could be assigned to a specific year.")
    
    # Plot for timeline distribution
    def plot_timeline():
        sns.histplot(df_time['year'], bins=30, kde=True, color='navy')
        plt.ylabel('Number of Objects', fontsize=12)
        plt.xlabel('Year', fontsize=12)
    plot_and_save(plot_timeline, '2_timeline_distribution.png', 'Historical Distribution of Objects')

    # --- 4. Description Text Analysis ---
    print("\n" + "="*60)
    print("4. Analyzing the generated descriptions...")
    print("="*60)
    df['description_length_words'] = df['description'].str.split().str.len()
    print("Statistics for description length (in words):")
    print(df['description_length_words'].describe())
    
    # Create Word Cloud
    print("\nGenerating a Word Cloud from all descriptions...")
    text = " ".join(desc for desc in df['description'].dropna())
    wordcloud = WordCloud(
        width=1200, height=800,
        background_color='white',
        stopwords=STOPWORDS_EN,
        min_font_size=10,
        colormap='cividis'
    ).generate(text)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig('3_description_wordcloud.png')
    plt.close()
    print("✅ Word Cloud saved as: 3_description_wordcloud.png")

    print("\n" + "="*60)
    print("✅ Analysis complete!")
    print("="*60)

# Run the script
if __name__ == "__main__":
    analyze_data()