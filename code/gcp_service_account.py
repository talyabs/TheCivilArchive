import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd


def read_sheet(sheet_name):
    # Define the scope
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]

    # Add your service account credentials
    creds = ServiceAccountCredentials.from_json_keyfile_name(
        "/data/talya/TheCivilArchive/credentials/oct-7th-the-civil-archive-fbd889171246.json",
        scope,
    )

    # Authorize the client
    client = gspread.authorize(creds)

    # Open the Google Sheet by name
    sheet = client.open("NER Test").worksheet(sheet_name)

    # Get all values from the sheet
    data = sheet.get_all_records()

    df = pd.DataFrame(data[1:], columns=data[0])

    # Display the DataFrame
    return df
