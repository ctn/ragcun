# Data Directory

This directory contains all data files used by RAGCUN.

## Structure

- `raw/` - Raw input documents (PDFs, text files, etc.)
- `processed/` - Processed and cleaned documents
- `embeddings/` - Stored vector embeddings for documents

## Usage

### In Google Colab

When working in Google Colab, you can:

1. **Upload files directly:**
```python
from google.colab import files
uploaded = files.upload()
```

2. **Mount Google Drive:**
```python
from google.colab import drive
drive.mount('/content/drive')
```

3. **Download from URL:**
```python
!wget -O data/raw/document.pdf https://example.com/document.pdf
```

### Local Development

Simply place your documents in the appropriate subdirectories:
- Raw documents → `data/raw/`
- Processed documents → `data/processed/`

## Gitignore

Note: The actual data files are not tracked in git (see .gitignore).
Only the directory structure and documentation are versioned.
