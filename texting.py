import os
import sys
import pickle
import base64

from email.mime.text import MIMEText

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import os
SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
load_dotenv()
phone = os.getenv('PHONE')
def send_text(
    phone_number: str= phone,
    subject: str="Atom: ",
    body: str = "You've Got a message",
    credentials_path=r'C:\Users\jordd\OneDrive\Desktop\Code\StockBot\NettedBot\Scalping\client_secret_550034098476-scso2t8qse1dfu4ri60nq38c6ho4le3s.apps.googleusercontent.com.json',
    token_path="token.pickle",
    fixed_port=8080
):
    """
    Sends an SMS message to a Verizon phone by emailing <number>@vtext.com via the Gmail API.
    
    :param phone_number: 10-digit phone number (as a string), e.g. '3366023500'
    :param subject: Subject line (will sometimes appear in the SMS body)
    :param body: The text message body
    :param credentials_path: Path to your OAuth2 client 'credentials.json'
    :param token_path: Where to store the OAuth token (refresh/access)
    :param fixed_port: Port number for run_local_server (8080 or any free port)
    :return: The Gmail API response dict
    """
    # 1) Load existing creds (if any)
    creds = None
    if os.path.exists(token_path):
        with open(token_path, "rb") as token_file:
            creds = pickle.load(token_file)

    # 2) If no valid creds, do OAuth
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            try:
                # Use a fixed port to avoid ephemeral-port issues
                creds = flow.run_local_server(port=fixed_port)
            except KeyboardInterrupt:
                print("\n[!] Keyboard interrupt detected; exiting.")
                sys.exit(1)

        # Save creds for next time
        with open(token_path, "wb") as token_file:
            pickle.dump(creds, token_file)

    # 3) Build the Gmail service
    service = build("gmail", "v1", credentials=creds)

    # 4) Create the message (MIMEText)
    msg = MIMEText(body)
    msg["to"] = f"{phone_number}@vtext.com"
    msg["from"] = "me"  # 'me' is recognized by Gmail API as the authenticated user
    msg["subject"] = subject

    # 5) Encode and send
    raw_msg = base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")
    response = service.users().messages().send(
        userId="me",
        body={"raw": raw_msg}
    ).execute()

    print(f"Sent to {phone_number}@vtext.com (Message ID: {response.get('id')})")
    return response

# Optional: You could also define similar helper functions for other carriers:
# e.g., send_att_text(), send_tmobile_text(), etc., changing the "@vtext.com" domain
# to "@txt.att.net" or "@tmomail.net".
