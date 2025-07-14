from typing import List, Optional
from pydantic import BaseModel, Field
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import os
import pickle
import base64
import email
from email.mime.text import MIMEText
from langchain_core.tools import StructuredTool
# OAuth 2.0 scopes for Gmail API
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly',
          'https://www.googleapis.com/auth/gmail.compose']

def get_gmail_service():
    """Initialize and return Gmail API service"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    return build('gmail', 'v1', credentials=creds)

class EmailSummaryInput(BaseModel):
    max_results: int = Field(
        default=5,
        description="Maximum number of emails to summarize"
    )

class DraftEmailInput(BaseModel):
    to: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")

def get_email_summaries(max_results: int = 5) -> str:
    """Get summaries of recent emails from Primary inbox"""
    service = get_gmail_service()
    
    # Get messages from Primary category
    results = service.users().messages().list(
        userId='me',
        maxResults=max_results,
        labelIds=['CATEGORY_PERSONAL', 'INBOX']  # This filters for Primary inbox
    ).execute()
    messages = results.get('messages', [])
    
    if not messages:
        return "No emails found in Primary inbox."
    
    summaries = []
    for message in messages:
        msg = service.users().messages().get(
            userId='me', id=message['id'], format='full').execute()
        
        headers = msg['payload']['headers']
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'No Date')
        
        # Get email body
        body = ''
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    body = base64.urlsafe_b64decode(part['body']['data']).decode()
                    # Truncate body if too long
                    body = body[:200] + '...' if len(body) > 200 else body
        elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
            body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode()
            body = body[:200] + '...' if len(body) > 200 else body
        
        summary = f"""
From: {sender}
Date: {date}
Subject: {subject}
Preview: {body}
-------------------"""
        summaries.append(summary)
    
    return "\n".join(summaries)

def draft_email(to: str, subject: str, body: str) -> str:
    """Create a draft email"""
    service = get_gmail_service()
    
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    
    encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    
    draft = service.users().drafts().create(
        userId='me',
        body={'message': {'raw': encoded_message}}
    ).execute()
    
    return f"Draft email created successfully. Draft ID: {draft['id']}"

# Define the tools list for LangChain
gmail_tools = [
    StructuredTool(
        name="get_email_summaries",
        description="Get summaries of recent emails from Gmail inbox",
        func=get_email_summaries,
        args_schema=EmailSummaryInput
    ),
    StructuredTool(
        name="draft_email",
        description="Create a draft email in Gmail",
        func=draft_email,
        args_schema=DraftEmailInput
    )
]
