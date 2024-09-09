import spacy
import pandas as pd
from gcp_service_account import read_sheet
from text_utils import *
from constants import *
import time
import datetime

# current date format dd-mm-yyyy
now = datetime.datetime.now()
current_date = now.strftime("%d-%m-%Y")

start = time.time()
nlp = spacy.load("he_ner_news_trf")

SCORE_THRESHOLD = 0.6


def organize_df(path):
    df = pd.read_csv(path)
    testimony_col = "תוכן העדות בכתב (העתק או תמלול)"
    url_col = "קישור למקור (URL) / Identifier"
    url2 = "קישור לאתר / ערוץ (URL)"
    df.rename(
        columns={testimony_col: "testimony", url_col: "url", url2: "url2"},
        inplace=True,
    )
    df = df.replace("״", '"', regex=False)

    df["text"] = df.apply(
        lambda row: (
            (row["Title"] + " " if not pd.isna(row["Title"]) else "")
            + (
                ""
                if row["Title"] == row["Short title"]
                else row["Short title"] + " "
                if not pd.isna(row["Short title"])
                else ""
            )
            + (row["testimony"] if not pd.isna(row["testimony"]) else "")
        ).lower(),
        axis=1,
    )
    # Extract entities from Title + Short title
    df["title_text"] = df.apply(
        lambda row: (
            (row["Title"] + " " if not pd.isna(row["Title"]) else "")
            + (
                (row["Short title"] + " ")
                if not pd.isna(row["Short title"])
                and row["Title"] != row["Short title"]
                else ""
            )
        )
        .strip()
        .lower(),
        axis=1,
    )

    return df


def find_entities(df):
    org_entities = []
    orgs_title_entities = []
    pers_entities = []
    pers_title_entities = []
    date_entities = []
    time_entities = []
    loc_entities = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing row {i}...")

        if row["text"].strip() == "":
            org_entities.append([])
            pers_entities.append([])
            pers_title_entities.append([])
            orgs_title_entities.append([])
            date_entities.append([])
            time_entities.append([])
            loc_entities.append([])
            continue

        pers_from_title = []
        org_from_title = []
        # Entities from title only (For Agent, Publisher)
        if row["title_text"].strip():
            title_doc = nlp(row["title_text"])
            for entity in title_doc.ents:
                score = round(entity._.confidence_score, 2)
                if entity.label_ == "PERS":
                    entity_name = segment_preposition_letters(entity.text)
                    if len(entity_name.split()) >= 2 and score >= SCORE_THRESHOLD:
                        pers_from_title.append(entity_name)
                if entity.label_ == "ORG":
                    entity_name = segment_preposition_letters(entity.text)
                    if score >= SCORE_THRESHOLD:
                        org_from_title.append(entity_name)

        # Process the entire text (title + testimony)
        doc = nlp(row["text"])
        orgs = []
        pers = []
        dates = []
        times = []
        locs = []

        # Entites from the entire text (including title)
        for entity in doc.ents:
            score = round(entity._.confidence_score, 2)
            if entity.label_ in ["PERS", "ORG", "LOC"]:
                entity_name = segment_preposition_letters(entity.text)
            else:
                entity_name = entity.text

            if entity.label_ == "ORG" and score >= SCORE_THRESHOLD:
                orgs.append(entity_name)
            elif entity.label_ == "PERS" and score >= SCORE_THRESHOLD:
                pers.append(entity_name)
            elif entity.label_ == "DATE" and score >= SCORE_THRESHOLD:
                dates.append(entity_name)
            elif entity.label_ == "TIME" and score >= SCORE_THRESHOLD:
                times.append(entity_name)
            elif entity.label_ == "LOC" and score >= SCORE_THRESHOLD:
                locs.append(entity_name)

        pers_title_entities.append(pers_from_title)
        orgs_title_entities.append(org_from_title)

        org_entities.append(orgs)
        pers_entities.append(pers)
        date_entities.append(dates)
        time_entities.append(times)
        loc_entities.append(locs)

    # Fields from the URL
    df[["URL Domain", "URL Title", "URL Profile", "URL Content Type"]] = df.apply(
        combine_url_info, axis=1
    )
    df["Language"] = df.apply(lambda row: detect_language(row["testimony"]), axis=1)
    df["Language_he"] = df["Language"].map(language_mapping)
    df["ORG"] = org_entities
    df["ORG_TITLE"] = orgs_title_entities
    df["PERS"] = pers_entities
    df["PERS_TITLE"] = pers_title_entities
    df["LOC"] = loc_entities
    df["DATE"] = date_entities
    df["DATE_CLEAN"] = df["DATE"].apply(lambda dates: extract_valid_dates(dates))
    df["TIME"] = time_entities

    return df


def synonyms_handeling(df):
    print("Reading organizations, locations, and person sheets...")
    org_df = read_sheet("organizations_synonyms")
    loc_df = read_sheet("locations_synonyms")
    pers_df = read_sheet("person_synonyms")

    print("Adding synonyms to the relevant entity columns...")
    add_synonyms_to_entity_columns(df, "ORG", org_df)
    add_synonyms_to_entity_columns(df, "ORG_TITLE", org_df, "title_text")
    add_synonyms_to_entity_columns(df, "LOC", loc_df)
    add_synonyms_to_entity_columns(df, "PERS", pers_df)
    add_synonyms_to_entity_columns(df, "PERS_TITLE", pers_df, "title_text")

    print("Applying synonyms to the DataFrame...")
    df = replace_synonyms(df, "ORG", org_df)
    df = replace_synonyms(df, "ORG_TITLE", org_df)
    df = replace_synonyms(df, "LOC", loc_df)
    df = replace_synonyms(df, "PERS", pers_df)
    df = replace_synonyms(df, "PERS_TITLE", pers_df)

    return df


def rulebased_entities(df):
    print("Applying rule-based entities...")
    df = add_entities(df, is_loc, is_person, is_org)
    df = apply_blacklist(df, not_loc, not_pers, not_org)
    return df


def clean_text_add_id(df, recreate_ids=False):
    df = final_cleaning(df)
    df["PERS"] = df["PERS"].apply(
        lambda pers_list: [
            entity
            for entity in pers_list
            if (len(entity.split()) >= 2 and entity not in specific_single_token_names)
        ]
        if pers_list is not None
        else []  # Handle NoneType by returning an empty list
    )

    df = df.rename(
        columns={
            "DATE_CLEAN": "DATE",
            "DATE": "DATE_FULL",
        }
    )
    df.drop(columns=["DATE_FULL"], inplace=True)

    # TODO: Here we create each time new ids
    # Need to support cases where id exists already
    df = add_unique_ids_and_summary(df)

    return df


def add_keywords(df):
    print("Adding keywords to the DataFrame...")
    keywords_list = get_unique_keywords(df)
    df["KEYWORDS"] = df["text"].apply(
        lambda x: extract_keywords_from_text(x, keywords_list)
    )
    return df


def hybrid_model(csv_path):
    output_path = f"data_archive_with_entities_{current_date}.csv"
    df = organize_df(csv_path)
    df = find_entities(df)
    df = synonyms_handeling(df)
    df = rulebased_entities(df)
    df = clean_text_add_id(df)
    df = add_keywords(df)
    df = df.drop(columns=["text", "Unnamed: 33", "title_text"])

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print(f"Saving the DataFrame to a CSV file {output_path}...")
    df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    path = "data/data_archive - main.csv"
    df_output = hybrid_model(path)


# TODO: If entity exist - do not add id
