import string
import ast
import pandas as pd
import re
import langid
import hashlib

from constants import *
from transformers import AutoModel, AutoTokenizer

from gcp_service_account import read_sheet

tokenizer = AutoTokenizer.from_pretrained("dicta-il/dictabert-seg")
model = AutoModel.from_pretrained(
    "dicta-il/dictabert-seg", trust_remote_code=True, max_length=1200
)
model.eval()


def normalize_person_name(name, exclusions_list=[]):
    # Return None if the name is in not_pers (assuming not_pers is defined elsewhere)
    if name in not_pers:
        return None

    # Check if name contains any exclusion or starts with any exclusion
    if any(exclusion in name for exclusion in exclusions_list) or name.startswith(
        tuple(exclusions_list)
    ):
        return name

    # Remove specific Hebrew letters from the start of the name
    name = re.sub(r"^(ל|כש|ו)", "", name)
    return name


def normalize_location(name, exclusions_list=[]):
    # for exclusion in exclusions_list:
    #     if exclusion in name:
    #         return exclusion
    if any(exclusion in name for exclusion in exclusions_list) or name.startswith(
        tuple(exclusions_list)
    ):
        return name
    name = re.sub(r"-(?=\S)", " ", name)  # Replace dashes with spaces
    name = re.sub(r"^(|ו|ל|ב|מ|מה)", "", name)
    return name.strip()


def normalize_org(name, exclusions_list=[]):
    if name in not_org:
        return
    for exclusion in exclusions_list:
        if exclusion in name:
            return exclusion
    name = re.sub(r"^(מה|וש|כשה|ל|ול|ו)", "", name)
    return name


def normalize_time(name, exclusions_list=[]):
    name = re.sub(r"^(ב|בשעה|בשעות|בשעת)", "", name)
    return name


def segment_text(text_):
    # Assuming predict method is correctly applied to segment text
    segmented_text = model.predict([text_], tokenizer)
    return segmented_text


def segment_preposition_letters(entity_text):
    """Clean entity by removing prepositional prefixes if they are at the start and ensuring single-letter prepositions are attached to the next word."""
    entity_text = entity_text.replace("-", " ").replace("׳", "'")
    entity_text = re.sub(r"[\\/\*\%\$\#\!\?\^]", " ", entity_text)
    entity_text = re.sub(r"\s+", " ", entity_text).strip()

    segmented = segment_text(entity_text)

    # Flatten the segmented output
    segmented_tokens = [token for sublist in segmented[0] for token in sublist]
    segmented_tokens = [
        token for token in segmented_tokens if token not in ("[CLS]", "[SEP]")
    ]

    # Rebuild the entity, ensuring single-letter prepositions are handled correctly
    cleaned_tokens = []
    i = 0
    while i < len(segmented_tokens):
        token = segmented_tokens[i]

        if i == 0 and token in HEBREW_PREPOSITIONS:
            # Skip the first token if it's a preposition (e.g., 'ב', 'ל', 'כ')
            i += 1
            continue

        if (
            len(token) == 1
            and token in HEBREW_PREPOSITIONS
            and i < len(segmented_tokens) - 1
        ):
            # Attach single-letter prepositions to the next token
            next_token = segmented_tokens[i + 1]
            cleaned_tokens.append(token + next_token)
            i += 1  # Skip the next token as it has been combined
        elif len(token) == 1 and token == "ה" and i < len(segmented_tokens) - 1:
            # If 'ה' is a standalone token, attach it to the next token
            next_token = segmented_tokens[i + 1]
            cleaned_tokens.append(token + next_token)
            i += 1  # Skip the next token as it has been combined
        elif token == "״":
            # Handle case where "״" is between two tokens, connect the previous and next token
            if i > 0 and i < len(segmented_tokens) - 1 and cleaned_tokens:
                prev_token = cleaned_tokens.pop()  # Remove the last token
                next_token = segmented_tokens[i + 1]
                cleaned_tokens.append(prev_token + token + next_token)
                i += 1  # Skip the next token as it has been combined
            else:
                cleaned_tokens.append(token)
        elif token == "'":
            # Handle cases where geresh (') is between two tokens (middle of a word)
            if i > 0 and i < len(segmented_tokens) - 1 and cleaned_tokens:
                prev_token = cleaned_tokens.pop()
                next_token = segmented_tokens[i + 1]
                cleaned_tokens.append(prev_token + token + next_token)
                i += 1  # Skip the next token
            else:
                cleaned_tokens.append(token)
        elif token.endswith("'") or token.endswith("״") or token.endswith('"'):
            # Handle cases where the token ends with geresh (') or gereshayim (״)
            if i < len(segmented_tokens) - 1:
                next_token = segmented_tokens[i + 1]
                cleaned_tokens.append(token + next_token)
                i += 1  # Skip the next token as it has been combined
            else:
                cleaned_tokens.append(token)
        elif token.startswith("'") or token.startswith("״"):
            # Handle cases where the token starts with geresh (' or ") and attach to the previous one
            if cleaned_tokens:
                prev_token = cleaned_tokens.pop()  # Remove the last token
                cleaned_tokens.append(
                    prev_token + token
                )  # Attach it to the current token
        else:
            cleaned_tokens.append(token)

        i += 1

    # Join tokens while making sure there are no extra spaces
    cleaned_entity = " ".join(cleaned_tokens).replace("־", "").strip()

    # Further remove any double spaces that may have been introduced
    cleaned_entity = " ".join(cleaned_entity.split())

    # If the entity ends with " ׳", attach it to the last word
    if cleaned_entity.endswith(" '"):
        parts = cleaned_entity.rsplit(" ", 1)
        if len(parts) == 2:
            cleaned_entity = parts[0] + "׳" + parts[1]

    return cleaned_entity.strip()


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
        if not normalized_name:  # Check if normalized_name is None or empty
            continue
        parts = normalized_name.split()
        if len(parts) == 2:
            first_name, surname = parts
            full_name_dict[normalized_name] = max(
                confidence, full_name_dict.get(normalized_name, 0)
            )
            first_name_dict[first_name] = max(
                confidence, first_name_dict.get(first_name, 0)
            )
        elif len(parts) == 1:
            single_name = parts[0]
            if single_name in first_name_dict:
                first_name_dict[single_name] = max(
                    confidence, first_name_dict.get(single_name, 0)
                )
            else:
                surname_dict[single_name] = max(
                    confidence, surname_dict.get(single_name, 0)
                )

    for full_name in list(full_name_dict.keys()):
        first_name, surname = full_name.split()
        if surname in surname_dict:
            full_name_dict[full_name] = max(
                full_name_dict[full_name], surname_dict[surname]
            )
            del surname_dict[surname]
        if first_name in first_name_dict:
            full_name_dict[full_name] = max(
                full_name_dict[full_name], first_name_dict[first_name]
            )
            del first_name_dict[first_name]

    merged_entities = (
        list(full_name_dict.items())
        + list(surname_dict.items())
        + list(first_name_dict.items())
    )
    return merged_entities


def extract_valid_dates(dates):
    standardized_dates = set()

    for date_str in dates:
        standardized_date = standardize_date(date_str)
        if standardized_date:
            standardized_dates.add(standardized_date)

    return sorted(standardized_dates)


def extract_info_url(url):
    if not isinstance(url, str):  # Check if the URL is not a string
        return None, None, None, None

    # Adjusted pattern to capture the second-level domain
    domain_pattern = r"https?://(?:www\.|m\.)?([^/]+)"
    domain_match = re.search(domain_pattern, url)
    domain = domain_match.group(1) if domain_match else None
    domain = domain.split(".")[0] if domain else None

    # Special handling for Telegram URLs
    if "t.me" in url:
        domain = "telegram"
        profile_pattern = r"t\.me/([^/]+)/"
        profile_match = re.search(profile_pattern, url)
        profile = profile_match.group(1) if profile_match else None
    else:
        profile = None

    # Title extraction
    title = None
    url_content_type = None

    # Extract title for specific domains like Mako
    if domain == "mako":
        title_pattern = r"Article-([^/]+)\.htm$"
        title_match = re.search(title_pattern, url)
        title = title_match.group(1).replace("-", " ") if title_match else None

    # Handle general cases and exclude certain domains
    if domain not in [
        "facebook",
        "youtube",
        "ynet",
        "apple",
        "telegram",
        "haaretz",
        "glz",
        "kan",
        "calcalist",
        "instagram",
        "tiktok",
    ]:
        title_pattern = r"/([^/?]+)$"
        title_match = re.search(title_pattern, url)
        title = (
            title_match.group(1).replace("-", " ")
            if title_match and not re.search(r"\.\w{2,4}$", title_match.group(1))
            else title
        )

        # Return None if the title is numeric
        if title and title.isdigit():
            title = None

    # Specific handling for social media and content types
    if "facebook.com" in url:
        if "/posts/" in url:
            url_content_type = "posts"
        else:
            url_content_type = None
    elif domain == "youtube":
        url_content_type = "video"
    elif domain == "tiktok":
        profile_pattern = r"tiktok\.com/@([^/]+)/?"
        profile_match = re.search(profile_pattern, url)
        profile = profile_match.group(1) if profile_match else profile
        if "/video/" in url:
            url_content_type = "video"
    elif domain == "instagram":
        profile_pattern = r"instagram\.com/([^/]+)/?"
        profile_match = re.search(profile_pattern, url)
        profile = profile_match.group(1) if profile_match else profile

    # Profile extraction and filtering for other URLs
    if domain not in ["telegram"]:
        profile_pattern = r"twitter\.com/([^/]+)/|facebook\.com/([^/]+)/|youtube\.com/([^/]+)/|instagram\.com/([^/]+)/"
        profile_match = re.search(profile_pattern, url)

        if profile_match:
            profile = next(
                p for p in profile_match.groups() if p
            )  # Get the non-None value

        # Exclude unwanted profiles and set url_content_type
        if profile in [
            "watch",
            "photo",
            "search",
            "reel",
            "plugins",
            "groups",
            "share",
        ]:
            url_content_type = profile  # Save the profile to url_content_type column
            profile = None  # Set profile to None after storing in url_content_type
        elif profile and profile.isdigit():
            profile = None
        elif profile:
            # Capitalize the first letters and replace '.' with a space
            profile = " ".join([part.capitalize() for part in profile.split(".")])

    if domain == "fb":
        domain = "facebook"

    return domain, title, profile, url_content_type


def detect_language(text):
    if (
        not isinstance(text, str) or not text.strip()
    ):  # Check if text is not a string or is empty
        return None
    lang, _ = langid.classify(text)
    return lang


def remove_date_from_name(name):
    # Split the name by '|' and remove any part that looks like a date
    parts = name.split("|")
    cleaned_parts = []

    for part in parts:
        if not re.match(
            r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", part
        ):  # Check if the part is not a date
            cleaned_parts.append(part.strip())  # Add non-date parts to cleaned_parts

    name = " ".join(cleaned_parts)
    cleaned_name = name.replace("\u200f", "").strip()
    return cleaned_name


def clean_text(text):
    if isinstance(text, list):
        # Normalize quotation marks and remove dates
        text = [remove_date_from_name(word.replace('"', "״")) for word in text]
        text = [word.strip("״") for word in text]  # Remove leading or trailing ״
        text = [word.replace("'", "׳") for word in text]  # Standardize single quotes
        # Remove duplicates by converting to a set and back to a list
        text = list(set(text))
    return text


def standardize_date(date_str):
    # Mapping special phrases to specific dates

    # Check if the date_str matches a special mapping
    if date_str in special_mappings:
        return special_mappings[date_str]

    # Remove unnecessary characters like extra spaces or commas
    date_str = date_str.strip().lower()

    # Check for exact dates like '11/12/2023' and convert to '11.12.2023'
    if re.match(r"\d{1,2}[./]\d{1,2}[./]\d{4}", date_str):
        return date_str.replace("/", ".")

    # Check for short dates like '7.10.23' and convert to '7.10.2023'
    match = re.match(r"(\d{1,2})[./](\d{1,2})[./](\d{2,4})", date_str)
    if match:
        day, month, year = match.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{day}.{month}.{year}"

    # Filter out time formats (e.g., '17:30', '17.30')
    if re.match(r"^\d{1,2}[:.]\d{2}$", date_str):
        return None

    # Convert Hebrew date strings like '7 אוקטובר 2023' to '7.10.2023'

    match = re.match(r"(\d{1,2}) ([א-ת]+) (\d{4})", date_str)
    if match:
        day, month_name, year = match.groups()
        if month_name in hebrew_months and year in ["2023", "2024"]:
            return f"{day}.{hebrew_months[month_name]}.{year}"

    # Convert Hebrew short date strings like '7 באוקטובר' to '7.10.2023'
    match = re.match(r"(\d{1,2}) ב([א-ת]+)", date_str)
    if match:
        day, month_name = match.groups()
        if month_name in hebrew_months:
            return f"{day}.{hebrew_months[month_name]}.2023"

    match = re.match(r"(\w+) (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})?", date_str)
    if match:
        month_name, day, year = match.groups()
        if month_name in english_months:
            if not year:
                year = "2023"  # Default to 2023 if no year is provided
            return f"{day}.{english_months[month_name]}.{year}"

    # Convert short dates like '7.10' to '7.10.2023'
    match = re.match(r"(\d{1,2})[./](\d{1,2})", date_str)
    if match:
        day, month = match.groups()
        return f"{day}.{month}.2023"

    # Exclude invalid dates that do not match any of the patterns above
    return None


def extract_info_url_safe(url):
    if url:
        result = extract_info_url(url)
        if len(result) == 4:
            return result
        else:
            return result + (None,) * (4 - len(result))
    else:
        return (None, None, None, None)


# First pass: extract from 'url'
# Apply extract_info_url_safe to 'url' and 'url2' and combine the results
def combine_url_info(row):
    url_info = extract_info_url_safe(row["url"])
    url2_info = extract_info_url_safe(row["url2"])

    combined_info = []
    for info1, info2 in zip(url_info, url2_info):
        combined_info.append(info1 if info1 else info2)

    return pd.Series(combined_info)


def add_unique_ids_and_summary(df):
    unique_ids_pers = {}
    unique_ids_loc = {}
    unique_ids_org = {}  # Dictionary for ORG IDs
    pers_summary = []
    loc_summary = []
    org_summary = []  # Summary for ORG entities

    # Explode columns once
    exploded_pers = df["PERS"].explode()
    exploded_loc = df["LOC"].explode()
    exploded_org = df["ORG"].explode()

    # Generate unique IDs for PERS
    for person in exploded_pers.unique():
        if pd.notna(person):
            unique_id = generate_short_id(person)
            unique_ids_pers[person] = unique_id
            pers_summary.append(
                {
                    "Person": person,
                    "Count": (exploded_pers == person).sum(),
                    "ID": unique_id,
                }
            )

    # Generate unique IDs for LOC
    for loc in exploded_loc.unique():
        if pd.notna(loc):
            unique_id = generate_short_id(loc)
            unique_ids_loc[loc] = unique_id
            loc_summary.append(
                {
                    "Location": loc,
                    "Count": (exploded_loc == loc).sum(),
                    "ID": unique_id,
                }
            )

    # Generate unique IDs for ORG
    for org in exploded_org.unique():
        if pd.notna(org):
            unique_id = generate_short_id(org)
            unique_ids_org[org] = unique_id
            org_summary.append(
                {
                    "Organization": org,
                    "Count": (exploded_org == org).sum(),
                    "ID": unique_id,
                }
            )

    # Create summary dataframes
    pers_summary_df = pd.DataFrame(pers_summary)
    loc_summary_df = pd.DataFrame(loc_summary)
    org_summary_df = pd.DataFrame(org_summary)

    # Save summary dataframes to CSV
    print("Saving summary dataframes to CSV, find it in results/ folder...")
    pers_summary_df.to_csv("results/person_summary.csv", index=False)
    loc_summary_df.to_csv("results/location_summary.csv", index=False)
    org_summary_df.to_csv("results/organization_summary.csv", index=False)

    # Add unique IDs for PERS
    df["PERS_IDS"] = df["PERS"].apply(
        lambda pers_list: [
            unique_ids_pers[person] for person in pers_list if pd.notna(person)
        ]
        if isinstance(pers_list, list)
        else []
    )

    # Add unique IDs for LOC
    df["LOC_IDS"] = df["LOC"].apply(
        lambda loc_list: [unique_ids_loc[loc] for loc in loc_list if pd.notna(loc)]
        if isinstance(loc_list, list)
        else []
    )

    # Add unique IDs for ORG
    df["ORG_IDS"] = df["ORG"].apply(
        lambda org_list: [unique_ids_org[org] for org in org_list if pd.notna(org)]
        if isinstance(org_list, list)
        else []
    )

    return df


def generate_short_id(name):
    # Generate a short ID using the first 8 characters of an MD5 hash
    md5_hash = hashlib.md5(name.encode("utf-8")).hexdigest()
    return md5_hash[:10]


def replace_synonyms(df, column_name, synonyms_file):
    # Load synonyms data from the provided CSV file
    syn_df = synonyms_file

    # Normalize the synonyms to lowercase, preserving Hebrew punctuation
    syn_df = syn_df.applymap(
        lambda x: x.strip().lower().replace("״", '"').replace("׳", "'")
        if isinstance(x, str)
        else x
    )

    # Flatten the synonyms DataFrame for easier searching
    synonym_columns = syn_df.columns.drop("name")
    syn_dict = pd.melt(syn_df, id_vars=["name"], value_vars=synonym_columns)
    syn_dict = syn_dict.dropna().set_index("value")["name"].to_dict()

    # Add synonyms with common Hebrew connectors
    connectors = ["ל", "ש", "ב"]
    expanded_syn_dict = syn_dict.copy()
    for key, value in syn_dict.items():
        for connector in connectors:
            expanded_syn_dict[connector + key] = value

    def clean_entity(entity):
        # Normalize the entity name by removing punctuation and converting to lowercase
        entity_cleaned = entity.strip().lower().replace("״", '"').replace("׳", "'")
        entity_cleaned = re.sub(r"\s+", " ", entity_cleaned)  # Normalize spaces
        entity_cleaned = entity_cleaned.translate(
            str.maketrans("", "", string.punctuation)
        )
        entity_cleaned = entity_cleaned.strip()
        entity_cleaned = re.sub(r"\s+", " ", entity_cleaned).strip()

        replacement = expanded_syn_dict.get(entity_cleaned)
        return replacement if replacement else entity

    df[column_name] = df[column_name].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )

    def process_entities(entities):
        cleaned_entities = []
        for entity in entities:
            if entity:
                cleaned_entity = clean_entity(entity)
                cleaned_entities.append(cleaned_entity)
        return list(set(cleaned_entities))

    df[column_name] = df[column_name].apply(process_entities)

    return df


def remove_date_from_name(text):
    # This regex removes common date patterns like 23.04.24, 23 / 04 / 24, etc.
    return re.sub(r"\b\d{1,2}[\./\-]\d{1,2}[\./\-]\d{2,4}\b", "", text)


def clean_text_exclude_non_entities(text, type, exclusion_list=[]):
    if isinstance(text, list):
        # Normalize quotation marks and remove dates
        text = [remove_date_from_name(word.replace('"', "״")) for word in text]
        text = [word.strip("״") for word in text]  # Remove leading or trailing ״
        text = [word.replace("'", "׳") for word in text]  # Standardize single quotes

        if type == "PERS":
            # Remove numbers and punctuations that are not - ' " ׳ ״
            text = [re.sub(r"[^א-תA-Za-z -׳״]", "", word) for word in text]
            # Remove standalone numbers and punctuations that resemble dates
            text = [remove_date_from_name(word) for word in text]

        # Remove words in the exclusion list
        text = [word for word in text if word not in exclusion_list]
        # Remove duplicates by converting to a set and back to a list
        text = list(set(text))
    return text


def read_exclusions_from_sheet_and_constants(not_org, not_pers, not_loc):

    clean_organization_df = read_sheet("clean organization")
    not_org_sheet = clean_organization_df[clean_organization_df["true_entity"] == "not"]
    not_org_sheet = not_org_sheet["Organization"].tolist()

    clean_person_df = read_sheet("clean person")
    not_pers_sheet = clean_person_df[clean_person_df["true_entity"] == "not"]
    not_pers_sheet = not_pers_sheet["Person"].tolist()

    clean_location_df = read_sheet("clean location")
    not_loc_sheet = clean_location_df[clean_location_df["true_entity"] == "not"]
    not_loc_sheet = not_loc_sheet["Location"].tolist()

    not_org.extend(not_org_sheet)
    not_pers.extend(not_pers_sheet)
    not_loc.extend(not_loc_sheet)

    not_loc = list(set(not_loc))
    not_pers = list(set(not_pers))
    not_org = list(set(not_org))

    return not_org, not_pers, not_loc


def add_entities(df, is_loc, is_person, is_org):
    # Function to update a specific column with entities from the list
    def update_column(row, entities, column_name, special_cases=None):
        text = row["text"]
        if text:
            # Handle special cases if provided
            if special_cases and any(
                phrase in text.lower() for phrase in special_cases
            ):
                row[column_name] = list(
                    set(special_cases.values())
                )  # Use the special case replacement
                return row[column_name]

            # Find entities in the text
            found_entities = [entity for entity in entities if entity in text]
            if found_entities:
                # Append to existing list or create a new one
                if isinstance(row[column_name], list):
                    row[column_name].extend(found_entities)
                else:
                    row[column_name] = found_entities

                # Remove duplicates
                row[column_name] = list(set(row[column_name]))
            return row[column_name]

    # Apply the function to update each column
    df["LOC"] = df.apply(
        lambda row: update_column(row, is_loc, "LOC", loc_special_cases), axis=1
    )
    df["PERS"] = df.apply(lambda row: update_column(row, is_person, "PERS"), axis=1)
    df["ORG"] = df.apply(lambda row: update_column(row, is_org, "ORG"), axis=1)

    return df


def apply_blacklist(df, not_loc, not_pers, not_org):
    not_org, not_pers, not_loc = read_exclusions_from_sheet_and_constants(
        not_org, not_pers, not_loc
    )

    # Function to filter out entities based on the blacklist
    def filter_column(row, column_name, exclusion_list):
        if isinstance(row[column_name], list):
            row[column_name] = [
                entity for entity in row[column_name] if entity not in exclusion_list
            ]
        return row[column_name]

    # Apply the blacklist to each column
    df["LOC"] = df.apply(lambda row: filter_column(row, "LOC", not_loc), axis=1)
    df["PERS"] = df.apply(lambda row: filter_column(row, "PERS", not_pers), axis=1)
    df["ORG"] = df.apply(lambda row: filter_column(row, "ORG", not_org), axis=1)

    return df


def final_cleaning(df):
    # Apply the cleaning function after updating entities
    df["ORG"] = df["ORG"].apply(
        lambda x: clean_text_exclude_non_entities(x, "ORG", not_org)
    )
    df["ORG_TITLE"] = df["ORG_TITLE"].apply(
        lambda x: clean_text_exclude_non_entities(x, "ORG", not_org)
    )
    df["PERS"] = df["PERS"].apply(
        lambda x: clean_text_exclude_non_entities(x, "PERS", not_pers)
    )
    df["PERS_TITLE"] = df["PERS_TITLE"].apply(
        lambda x: clean_text_exclude_non_entities(x, "PERS_TITLE", not_pers)
    )
    df["LOC"] = df["LOC"].apply(
        lambda x: clean_text_exclude_non_entities(x, "LOC", not_loc)
    )
    return df


def get_unique_keywords(df):
    column_name = "מילות מפתח"

    # Step 1: Drop rows where the column is NaN
    keywords = df[df[column_name].notna()][column_name]

    # Step 2: Split the strings by comma and flatten the list
    all_keywords = keywords.str.split(",").explode()

    # Step 3: Remove leading/trailing whitespace from each keyword
    all_keywords = all_keywords.str.strip()

    # Step 4: Get the unique values
    unique_keywords = all_keywords.drop_duplicates()

    # Step 5: Convert back to a list and return
    unique_keywords_list = unique_keywords.tolist()
    return unique_keywords_list


def extract_keywords_from_text(text, keywords):
    # Find the keywords that exist in the text
    found_keywords = [word for word in keywords if word in text]
    return (
        found_keywords if found_keywords else None
    )  # Return None if no keywords are found
