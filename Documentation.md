# Museum Catalogue Automation Project

## Background
This project was developed to support museum cataloguing using machine learning and Python-based tools. It combines image captioning powered by Google Gemini AI with data cleaning utilities for structured records. The main aim is to accelerate and standardize the documentation of technical heritage items while keeping human curators in the loop for validation.

The system helps museums like the Technikmuseum manage large depots of inventoried objects, producing both structured metadata and narrative catalogue entries.

---

## Code

### Files
1. **Experiment1.py**
   - Uses Google Gemini AI to generate descriptive captions for images.
   - Processes images from a folder or ZIP archive.
   - Produces structured outputs in CSV format.

2. **excelextract2.py**
   - Cleans Excel/CSV files by filtering row ranges and removing empty columns.
   - Outputs ready-to-use cleaned Excel files for catalogue records.

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
