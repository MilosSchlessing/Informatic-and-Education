Project: Image Caption Generator & Excel Extractor

Step #1: pip install -r requirements.txt

This repository contains two Python utilities:
Experiment1.py – Machine learning pipeline using Google Gemini AI to generate captions for images.
excelextract1.py/excelextract2.py – Data-cleaning tool to extract non-empty rows/columns from Excel or CSV files.


File 1: Experiment1.py

Purpose:
Generates descriptive catalogue captions for images (e.g., museum items).

Features:

- Uses Google Gemini AI via the google-generativeai library.
- Processes folders or ZIP archives of images (.jpg, .jpeg, .png).
- Produces captions in headline + description format.
- Cleans unwanted phrasing and provides fallback captions when AI output fails.
- Outputs results into a structured CSV (headline, caption, image_file).
- Respects API rate limits to avoid overload.

Usage:
        Run in terminal:

python gemini_musuem_gui.py

The script will prompt you for:

Path to a folder or ZIP file of images.

Output CSV filename (default: caption.csv).

Dependencies:

Python >= 3.9
google-generativeai
pandas
Pillow
python-dotenv

Setup:

Install required dependencies:

pip install -r requirements.txt

Add your Google API key in a .env file:
GOOGLE_API_KEY=your_api_key_here


File 2: excelextract1.py/excelextract2.py

Purpose:
Extracts rows 501–1000 from an Excel/CSV file and filters out empty columns.

Features:
- Reads .xls, .xlsx, or .csv files with multiple encoding fallbacks.
- Retains only non-empty columns in the given row range.
- Saves cleaned output to non_empty_501_1000.xlsx.
- Prints summary of kept vs. removed columns.

Usage:
Run in terminal:
python excelextract2.py

Dependencies:
pandas
numpy
openpyxl

Example Outputs

Experiment1.py:
Produces caption.csv with structure:
headline, caption, image_file
"Industrial Machine Part", "A detailed description of materials and condition...", "object1.jpg"

excelextract2.py:
Produces non_empty_501_1000.xlsx with only useful non-empty columns.

Notes:
1. Images should be reasonably sized (<2000×2000px).
2. Captions generated should be reviewed and edited by curators for accuracy.
3. API usage with Google Gemini may incur costs depending on the chosen plan.
