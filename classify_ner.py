import spacy
import pandas as pd
import re

# Load the SpaCy model
nlp = spacy.load("he_ner_news_trf")

# Load the DataFrame
df = pd.read_csv("/data/talya/TheCivilArchive/data_archive - main.csv")
df = df[:100]
relevant_column = "תוכן העדות בכתב (העתק או תמלול)"
df.rename(columns={relevant_column: "testimony"}, inplace=True)

# Initialize lists to store entities by type
org_entities = []
pers_entities = []
date_entities = []
time_entities = []
loc_entities = []
gpe_entities = []
duc_entities = []

# Define exclusion lists for each entity type
exclusions_orgs = ['שייטת גבעתי', 'שב״כ', 'הכי חם בגיהנום']
exclusions_place = ['בית קמה', 'באר שבע', 'בארי']
exclusions_pers = ['הרבי מלובביץ', 'לילך', 'ליאת', 'הילה', 'ליאם']
exclusions_dates = []
exclusions_times = []

def normalize_person_name(name, exclusions_list=[]):
    if any(exclusion in name for exclusion in exclusions_list):
        return name
    name = re.sub(r'^(כש|ו)', '', name)
    return name

def normalize_location(name, exclusions_list=[]):
    if any(exclusion in name for exclusion in exclusions_list):
        return name
    name = re.sub(r'-(?=\S)', ' ', name)  # Replace dashes with spaces
    name = re.sub(r'^(ב|מ|מה)', '', name)
    return name.strip()

def normalize_org(name, exclusions_list=[]):
    if any(exclusion in name for exclusion in exclusions_list):
        return name
    name = re.sub(r'^(מה|כשה|ל|ב)', '', name)
    return name

def normalize_time(name, exclusions_list=[]):
    name = re.sub(r'^(ב|בשעה|בשעות|בשעת)', '', name)
    return name

def merge_entities(entities, normalize_func, exclusions=[]):
    entity_dict = {}
    for name, confidence in entities:
        normalized_name = normalize_func(name, exclusions)
        if normalized_name in entity_dict:
            if confidence > entity_dict[normalized_name]:
                entity_dict[normalized_name] = confidence
        else:
            entity_dict[normalized_name] = confidence
    return list(entity_dict.items())

def merge_names(entities, exclusions=[]):
    full_name_dict = {}
    surname_dict = {}
    first_name_dict = {}

    for name, confidence in entities:
        normalized_name = normalize_person_name(name, exclusions)
        parts = normalized_name.split()
        if len(parts) == 2:
            first_name, surname = parts
            full_name_dict[normalized_name] = max(confidence, full_name_dict.get(normalized_name, 0))
            first_name_dict[first_name] = max(confidence, first_name_dict.get(first_name, 0))
        elif len(parts) == 1:
            single_name = parts[0]
            if single_name in first_name_dict:
                first_name_dict[single_name] = max(confidence, first_name_dict.get(single_name, 0))
            else:
                surname_dict[single_name] = max(confidence, surname_dict.get(single_name, 0))

    for full_name in list(full_name_dict.keys()):
        first_name, surname = full_name.split()
        if surname in surname_dict:
            full_name_dict[full_name] = max(full_name_dict[full_name], surname_dict[surname])
            del surname_dict[surname]
        if first_name in first_name_dict:
            full_name_dict[full_name] = max(full_name_dict[full_name], first_name_dict[first_name])
            del first_name_dict[first_name]

    merged_entities = list(full_name_dict.items()) + list(surname_dict.items()) + list(first_name_dict.items())
    return merged_entities

# Iterate over each row in the DataFrame
for i, row in df.iterrows():
    if i % 100 == 0:
        print(f"Processing row {i}...")
    title = row.get("Title") if not pd.isna(row.get("Title")) else ""
    short_title = row.get("Short title") if not pd.isna(row.get("Short title")) else ""
    testimony = row.get("testimony") if not pd.isna(row.get("testimony")) else ""
    if title == short_title:
        short_title = ""

    text = title + short_title + testimony
    if text.strip() == "":
        org_entities.append([])
        pers_entities.append([])
        date_entities.append([])
        time_entities.append([])
        loc_entities.append([])
        gpe_entities.append([])
        duc_entities.append([])
        continue

    doc = nlp(text)
    orgs = []
    pers = []
    dates = []
    times = []
    locs = []
    gpes = []
    ducs = []

    for entity in doc.ents:
        entity_info = (entity.text, round(entity._.confidence_score, 2))
        if entity.label_ == "ORG":
            orgs.append(entity_info)
        elif entity.label_ == "PERS":
            pers.append(entity_info)
        elif entity.label_ == "DATE":
            dates.append(entity_info)
        elif entity.label_ == "TIME":
            times.append(entity_info)
        elif entity.label_ == "LOC":
            locs.append(entity_info)
        elif entity.label_ == "GPE":
            gpes.append(entity_info)
        elif entity.label_ == "DUC":
            ducs.append(entity_info)

    pers = merge_names(pers, exclusions=exclusions_pers)
    locs = merge_entities(locs, normalize_location, exclusions=exclusions_place)
    orgs = merge_entities(orgs, normalize_org, exclusions=exclusions_orgs)
    dates = merge_entities(dates, normalize_time, exclusions=exclusions_dates)
    times = merge_entities(times, normalize_time, exclusions=exclusions_times)

    org_entities.append(orgs)
    pers_entities.append(pers)
    date_entities.append(dates)
    time_entities.append(times)
    loc_entities.append(locs)
    gpe_entities.append(gpes)
    duc_entities.append(ducs)

# Add the entities lists as new columns to the DataFrame
df["ORG"] = org_entities
df["PERS"] = pers_entities
df["DATE"] = date_entities
df["TIME"] = time_entities
df["LOC"] = loc_entities
df["GPE"] = gpe_entities
df["DUC"] = duc_entities

# Optionally save the updated DataFrame to a new CSV file
df.to_csv("/data/talya/TheCivilArchive/data_archive_with_entities.csv", index=False)

# Print the top 20 rows to verify the new columns
print(df.head(20))
