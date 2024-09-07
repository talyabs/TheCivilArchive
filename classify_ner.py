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


def organize_df(path):
    # Load the DataFrame
    df = pd.read_csv(path)
    df = df[:100]
    relevant_column = "תוכן העדות בכתב (העתק או תמלול)"
    url_col = "קישור למקור (URL) / Identifier"
    url2 = "קישור לאתר / ערוץ (URL)"
    df.rename(
        columns={relevant_column: "testimony", url_col: "url", url2: "url2"},
        inplace=True,
    )
    df = df.replace("״", '"', regex=False)

    # Iterate over each row in the DataFrame
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
    # Initialize lists to store entities by type
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
        if row["title_text"].strip():
            title_doc = nlp(row["title_text"])
            for entity in title_doc.ents:
                score = round(entity._.confidence_score, 2)
                if entity.label_ == "PERS":
                    # entity_name = segment_preposition_letters(entity.text)
                    entity_name = entity.text
                    if len(entity_name.split()) >= 2 and score >= 0.6:
                        pers_from_title.append(entity_name)
                if entity.label_ == "ORG":
                    # entity_name = segment_preposition_letters(entity.text)
                    entity_name = entity.text
                    if score >= 0.6:
                        org_from_title.append(entity_name)

        # Process the entire text (title + testimony)
        doc = nlp(row["text"])
        orgs = []
        pers = []
        dates = []
        times = []
        locs = []

        for entity in doc.ents:
            score = round(entity._.confidence_score, 2)
            if entity.label_ in ["PERS", "ORG", "LOC"]:
                # entity_name = segment_preposition_letters(entity.text)
                entity_name = entity.text
            else:
                entity_name = entity.text

            if entity.label_ == "ORG" and score >= 0.6:
                orgs.append(entity_name)
            elif entity.label_ == "PERS" and score >= 0.6:
                pers.append(entity_name)
            elif entity.label_ == "DATE" and score >= 0.6:
                dates.append(entity_name)
            elif entity.label_ == "TIME" and score >= 0.6:
                times.append(entity_name)
            elif entity.label_ == "LOC" and score >= 0.6:
                locs.append(entity_name)

        # Add PERS from title to both PERS_TITLE and PERS
        pers_title_entities.append(pers_from_title)
        orgs_title_entities.append(org_from_title)

        # Append entities to their corresponding lists
        org_entities.append(orgs)
        pers_entities.append(pers)
        date_entities.append(dates)
        time_entities.append(times)
        loc_entities.append(locs)

    # Make sure all lists are the same length as df
    assert len(pers_entities) == len(df)
    assert len(org_entities) == len(df)

    # Set the extracted entities into the DataFrame
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


def add_synonyms_to_entity_columns(df, entity_type, synonyms_df, field="text"):
    """
    Check if any of the names or synonyms in the synonyms_df are present in each row of the text column
    and add the corresponding name to the relevant entity column (ORG, LOC, PERS).

    Parameters:
    - df: The DataFrame containing the text data.
    - entity_type: The entity type column (e.g., "ORG", "LOC", "PERS").
    - synonyms_df: The DataFrame containing the names and synonyms for the entities.
    """
    for _, row in synonyms_df.iterrows():
        name = row["name"]
        synonyms = (
            row.dropna().tolist()
        )  # Get all non-null values including the name itself
        synonyms = [syn for syn in synonyms if syn]

        # Check each row in the DataFrame
        for index, text in df[field].iteritems():
            if any(syn in text for syn in synonyms):
                if name not in df.at[index, entity_type]:
                    df.at[index, entity_type].append(name)


def synonyms_handeling(df):
    print("Reading organizations, locations, and person sheets...")
    org_df = read_sheet("organizations")
    loc_df = read_sheet("locations")
    pers_df = read_sheet("person")

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
            if (len(entity.split()) >= 2 and entity not in specific_names)
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
    df = organize_df(csv_path)
    df = find_entities(df)
    df = synonyms_handeling(df)
    df = rulebased_entities(df)
    df = clean_text_add_id(df)
    df = add_keywords(df)
    df = df.drop(columns=["text", "Unnamed: 33"])
    end = time.time()
    print(f"Time taken: {end - start} seconds")
    return df


if __name__ == "__main__":
    path = "data/data_archive - main.csv"
    output_path = f"results/data_archive_with_entities_{current_date}.csv"
    df_output = hybrid_model(path)

    print(df_output.head(20))
    print(f"Saving the DataFrame to a CSV file {output_path}...")
    df_output.to_csv(output_path, index=False)


# TODO: If entity exist - do not add id
