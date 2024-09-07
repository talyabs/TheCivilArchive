Here's a draft of the README file for your repository:

---

# Civil Archive - Named Entity Recognition (NER) Hybrid Model

## Overview

This repository is dedicated to the Civil Archive project, focusing on Named Entity Recognition (NER) for Hebrew-language testimonies related to the October 7th incidents. The project employs a **hybrid model** that integrates multiple advanced techniques to enhance entity extraction:

1. **Hebrew NER Model**: Using the `he_ner_news_trf` model from the [HebSafeHarbor project](https://github.com/8400TheHealthNetwork/HebSafeHarbor) for accurate extraction of persons, locations, and organizations from the text.
   
2. **Segmentation Model**: Leveraging the **DictaBERT segmentation model** ([dicta-il/dictabert-seg](https://huggingface.co/dicta-il/dictabert-seg)) to handle the morphological complexity of Hebrew, segmenting words into meaningful components.

3. **Rule-Based Methods & Synonyms**: Applying rule-based approaches and a collection of synonyms maintained in a database, improving entity recognition accuracy through custom rules and pre-defined synonym mappings.

- For more details on the project, you can refer to the [detailed report](https://drive.google.com/file/d/1nG3YjFcVaCNNzDjI6V9snKZhQi8KGM-t/view?usp=sharing).

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

---

## Synonyms Database

- The synonyms used in the rule-based approach are stored in a Google Sheet. You can find the database here:
  [Synonyms Google Sheet](https://docs.google.com/spreadsheets/d/1cdfq43IcbOaj7mUquVeiRWEBg_O6IsWOVO5SVeRmMlM/edit?gid=854247877#gid=854247877). The code reads automatically from this file. Make sure you have the credentials file.
- In the sheet you find a file for each entity type: Location, Organization, Person (in this format location_synonyms, etc). If you want to add synonyms just add them in the relevant row (if exists), or in a new line. Each entity you will add under the `name` column will be added as entity in the "rulebased" phase of the algorithm (For ex. if you add ״ארגון צער בעלי חיים״ each testimony with this phrase weill get not this as ORG entity)
- In the same sheet you can find the summary files: `person/location/organizations_ids`, there you can find an id to each entity.
- To improve the algorithm there are "cleaning" files, for each entity type - Here you can mark `not` under `true_entity` if this entity should not classified as the entity type. You can also add new row if you saw a wrong entity.

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

3. The classified entities and summary files with entities id, will be saved as `{entity}_summary.csv` file.

- For faster processing, it is recommended to run the model on a computer equipped with a GPU.

---


## Model Output

The model identifies and extracts various types of entities from the testimonies, including both URL-based and Named Entity Recognition (NER) entities:

- **URL-Based Entities**:
  - **URL Domain**: Identifies the domain of the URL, which can provide insights into the "Publisher."
  - **URL Title**: Extracts the title from the URL (possible only in a few cases).
  - **URL Profile**: Captures the profile or account associated with a social media URL.
  - **URL Content Type**: If possible, extracts the type of content (e.g., video, post, reel, etc.).

- **Language Detection**:
  - **Language**: Identifies the language of the testimony.
  - **Language_he**: Specifically detects Hebrew or English (עברית/ אנגלית).

- **NER Entities**:
  - **ORG**: Extracts organization entities from the title, subtitle, and testimony text.
  - **ORG_TITLE**: Extracts organization entities only from the title, which can be useful to identify the source of the testimony.
  - **PERS**: Extracts person entities from the title, subtitle, and testimony text.
  - **PERS_TITLE**: Extracts person entities only from the title, which can be useful to identify the agent.
  - **LOC**: Extracts location entities from the title, subtitle, and testimony text.
  - **DATE**: Extracts date entities from the title, subtitle, and testimony text.
  - **TIME**: Extracts time entities from the title, subtitle, and testimony text.

- **Unique IDs**:
  - For each LOC, PERS, and ORG entity, the model assigns a unique ID. 
  - **Important Note**: Currently, each run generates new IDs. Future updates will include support to retrieve existing IDs from the database and only assign new IDs to previously unlisted entities.

- **Keywords**:
  - Based on a collected file of keywords, the model searches each testimony for matching keywords and adds them as an additional field.

---

## Contact

For access to GCP credentials or any other inquiries, please contact the repository owner - talyatalyab@gmail.com

---
