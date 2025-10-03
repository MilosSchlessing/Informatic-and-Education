# Museum Catalogue Automation Project (V2.0 - Gemini/CSV-Enriched)

## Background
This project was developed to support museum cataloguing using **Google Gemini AI** and **Python-based tools**. It combines advanced image description generation, enriched with factual data from existing museum records, with robust data cleaning and file management utilities. The primary goal is to **accelerate and standardize the documentation** of technical heritage items while keeping human curators in the loop for validation and final polish.

The system features a **standalone Graphical User Interface (GUI)** for ease of use, enabling non-technical staff to load image folders and master data to produce high-quality, fact-checked narrative catalogue entries.

---

## Code and System Architecture

### Files and Functionality 🛠️

The project evolved from simple scripts to a fully integrated application with five key components.

| Filename | Description (Functionality) | Final Version Role |
| :--- | :--- | :--- |
| **`data_subset_filter_nonempty.py`** | Cleans Excel/CSV files (`Liste1.xls` / `Liste2.xls`) by extracting specific row ranges and removing columns that are entirely empty in that range. | **Data Preparation** |
| **`merge_deduplicate_excel.py`** | **Data Integration:** Merges and dedupicates cleaned data (based on the `t1` ID column) from different source files into a single, clean `cleaned_data.csv`. | **Data Integration** |
| **`copy_images_by_id.py`** | **File Management:** Copies required image files from various source directories into a dedicated folder, based on the `T13` file paths listed in the `cleaned_data.csv`. | **File Preparation** |
| **`gemini_csv_enriched_generator.py`** | **AI Core Logic:** Generates museum-quality headlines and descriptions by fusing **visual analysis** of up to 4 images with **factual metadata** (Material, Date, Dimensions) retrieved from the CSV. | **Integrated** |
| **`gemini_museum_gui.py`** | **Main Application:** A cross-platform **CustomTkinter GUI** utilizing a multi-threaded architecture (TkinterDnD) to manage the entire process, including CSV loading, Gemini API calls, and live logging, without freezing the user interface. | **Main Application** |


### System Requirements
- Python 3.9 or higher
- Libraries:
  - google-generativeai
  - pandas
  - Pillow
  - python-dotenv
  - numpy
  - openpyxl
- Google Gemini API key (stored in `.env`)
- Input formats: `.jpg`, `.jpeg`, `.png` for images; `.xls`, `.xlsx`, `.csv` for tabular data
- Outputs: `caption.csv` (captions) and `non_empty_501_1000.xlsx` (cleaned data)

---

## Data

### Data Description
The initial dataset consisted of **300 records**, but these included:
- Duplicate entries (exact copies of the same item).
- Fragmented records, where a single object’s details were spread across multiple rows.

After data cleaning, the analysis focused on **32 unique technical items**, ensuring only distinct objects were included.

- **Total Unique Items:** 32
- **Time Span:** 1889–1980
- **Manufacturer Concentration:** 56.3% of items from AEG-Telefunken AG
- **Material Composition:** 53.1% of items made from Kunststoff (Plastic)

### Manufacturer / Collection Breakdown
- AEG-Telefunken AG — 18
- Allgemeine Elektricitäts-Gesellschaft AEG AG — 9
- No information yet — 3
- AEG-Aus- und Weiterbildungszentrum Berlin — 2

### Material Breakdown
- Kunststoff (Plastic) — 17
- Metall (Metal) — 6
- Glas (Glass) — 4
- Eisenblech (Sheet metal) — 2
- Kunststoff (u.a. Tenacit) — 1
- Textilgewebe (Textile) — 1
- Metall (Messing vernickelt) (Nickel-plated brass) — 1

### Collection Summary (Presentation Paragraph)
This collection of 70 unique technical objects, derived from an initial dataset of 300 records (containing extensive duplication and fragmentation), offers a focused look into a specific industrial history. Spanning nearly a century, from 1889 to 1980, the collection is overwhelmingly concentrated on a single manufacturer: AEG-Telefunken AG, which accounts for over 56% of the items. Furthermore, the material composition is dominated by Kunststoff (Plastic), representing over 53% of the primary materials,...
