Here's a draft of the README file for your repository:

---

# Civil Archive - Named Entity Recognition (NER) Hybrid Model

## Overview

This repository is dedicated to the Civil Archive project, focusing on Named Entity Recognition (NER) for Hebrew-language testimonies related to the October 7th incidents. The project employs a **hybrid model** that integrates multiple advanced techniques to enhance entity extraction:

1. **Hebrew NER Model**: Using the `he_ner_news_trf` model from the [HebSafeHarbor project](https://github.com/8400TheHealthNetwork/HebSafeHarbor) for accurate extraction of persons, locations, and organizations from the text.
   
2. **Segmentation Model**: Leveraging the **DictaBERT segmentation model** ([dicta-il/dictabert-seg](https://huggingface.co/dicta-il/dictabert-seg)) to handle the morphological complexity of Hebrew, segmenting words into meaningful components.

3. **Rule-Based Methods & Synonyms**: Applying rule-based approaches and a collection of synonyms maintained in a database, improving entity recognition accuracy through custom rules and pre-defined synonym mappings.

---

## File Structure

### Main Script: `classify_ner.py`
- This is the primary file used to classify entities within a CSV file.
- **Usage**: You need to specify the path to the CSV file you want to classify. The model will process the file and save the results in the `results` folder.
- The results include detailed summaries of the entities found, categorized by entity type (persons, locations, organizations).

### Results Folder: `results/`
- Contains the final output files from the classification process.
- A summary file is generated, showing the counts of entities found per entity type (e.g., person, location, organization).

### Google Sheets Integration
- The database for rules and synonyms is maintained in a Google Sheet, and the system connects to this sheet via **Google Cloud Platform (GCP)** credentials.
- **Note**: Please contact the repository owner to obtain the necessary GCP credentials for accessing the Google Sheets database.
- Relevant code for Google Sheets access is in the file `gcp_service_account.py`.

For more details on the project, you can refer to the [detailed report](https://drive.google.com/file/d/1nG3YjFcVaCNNzDjI6V9snKZhQi8KGM-t/view?usp=sharing).
---

## Synonyms Database

- The synonyms used in the rule-based approach are stored in a Google Sheet. You can find the database here:
  [Synonyms Google Sheet](https://docs.google.com/spreadsheets/d/1cdfq43IcbOaj7mUquVeiRWEBg_O6IsWOVO5SVeRmMlM/edit?gid=854247877#gid=854247877).

---

## Requirements

- **Python 3.x**
- **HuggingFace Transformers**: For using the NER and segmentation models.
- **Google Cloud SDK**: To connect to the Google Sheets for the rule-based and synonym database.

### Installation

```bash
pip install -r requirements.txt
```

---

## Running the Model

1. Add the path to your CSV file in `classify_ner.py`.
2. Run the classification process:
   
   ```bash
   python classify_ner.py
   ```

3. The classified entities and summary files will be saved in the `results` folder.

- For faster processing, it is recommended to run the model on a computer equipped with a GPU.

---

## Contact

For access to GCP credentials or any other inquiries, please contact the repository owner.

---
