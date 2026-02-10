"""
Gmail utility functions for reading emails and downloading attachments.

This module provides:
- Searching Gmail for application emails with attachments
- Reading email content and metadata
- Downloading PDF attachments
- Extracting text from PDFs

Author: Junior Dev (under Senior Dev review)
"""

import os
import logging
from typing import List, Dict, Tuple
from langchain_google_community import GmailToolkit
from config import api_resource
import base64
import PyPDF2

logger = logging.getLogger(__name__)

toolkit = GmailToolkit(api_resource=api_resource)

# ============================================================================
# GMAIL SEARCH & READ FUNCTIONS
# ============================================================================

def search_gmail() -> List[str]:
    """
    Search Gmail for emails with subject 'application' and attachments.
    
    Uses the Gmail API search tool to find emails matching:
    - Subject contains: "application"
    - Has attachment: yes
    
    Returns:
        List[str]: List of message IDs found
        
    Example:
        >>> msg_ids = search_gmail()
        >>> print(f"Found {len(msg_ids)} emails")
    """
    logger.info("Starting Gmail search for application emails with attachments")
    
    try:
        tools = toolkit.get_tools()
        search_tool = next(tool for tool in tools if tool.name == "search_gmail")
        
        query = "subject:application has:attachment"
        search_res = search_tool.invoke({"query": query})
        
        msg_ids = [msg['id'] for msg in search_res]
        logger.info(f"Found {len(msg_ids)} emails matching criteria")
        
        return msg_ids
        
    except Exception as e:
        logger.error(f"Error during Gmail search: {e}")
        return []


def get_email_body(payload: Dict) -> str:
    """
    Recursively extract plain text body from email payload.
    
    Gmail emails can have complex structures with multiple parts.
    This function searches for the text/plain MIME type.
    
    Args:
        payload (Dict): Email payload from Gmail API
        
    Returns:
        str: Decoded email body text, or empty string if not found
        
    Note:
        Prefers text/plain over HTML for simplicity
    """
    try:
        if payload.get('body') and payload['body'].get('data'):
            return base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        
        if payload.get('parts'):
            for part in payload['parts']:
                if part['mimeType'] == 'text/plain':
                    return base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    
        return ""
        
    except Exception as e:
        logger.warning(f"Error extracting email body: {e}")
        return ""


def read_email(msg_ids: List[str]) -> List[Dict]:
    """
    Read multiple emails and extract their content and attachments.
    
    For each message ID, fetches:
    - Sender email address
    - Subject line
    - Body text
    - Attachment metadata (filename, attachment_id)
    
    Args:
        msg_ids (List[str]): List of Gmail message IDs
        
    Returns:
        List[Dict]: List of email data dictionaries with keys:
            - id: message ID
            - sender: from address
            - subject: email subject
            - body: email text content
            - attachments: list of {filename, attachment_id}
            
    Example:
        >>> emails = read_email(['msg123', 'msg456'])
        >>> print(emails[0]['subject'])
    """
    logger.info(f"Reading {len(msg_ids)} emails")
    content = []
    
    for msg_id in msg_ids:
        try:
            message = api_resource.users().messages().get(userId='me', id=msg_id).execute()
            payload = message.get('payload', {})
            headers = payload.get('headers', [])
            parts = payload.get('parts', [])

            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "No Subject")
            sender = next((h['value'] for h in headers if h['name'] == 'From'), "Unknown")

            body = get_email_body(payload)

            
            files_found = []
            for part in parts:
                if part.get('filename'):
                    files_found.append({
                        "filename": part['filename'],
                        "attachment_id": part['body'].get('attachmentId')
                    })

            content.append({
                "id": msg_id,
                "sender": sender,
                "subject": subject,
                "body": body,
                "attachments": files_found
            })
            
            logger.debug(f"Read email {msg_id} from {sender}: {subject}")
            
        except Exception as e:
            logger.error(f"Error reading email {msg_id}: {e}")
            continue

    logger.info(f"Successfully read {len(content)} emails")
    return content


# ============================================================================
# PDF DOWNLOAD & EXTRACTION FUNCTIONS
# ============================================================================

def save_pdf(contents: List[Dict], save_path: str = ".\downloads") -> Tuple[str, str]:
    """
    Download PDF attachments from emails to local directory.
    
    For each email in contents, downloads all PDF attachments.
    Creates the save directory if it doesn't exist.
    
    Args:
        contents (List[Dict]): List of email data from read_email()
        save_path (str): Directory to save PDFs (default: ./downloads/)
        
    Returns:
        Tuple[str, str]: (path_to_last_pdf, email_body_of_last_email)
        
    Raises:
        Exception: If no attachments found or download fails
        
    Note:
        Only processes the FIRST attachment from the FIRST email
        This matches the original code behavior
    """
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Downloading PDFs to {save_path}")
    
    if not contents:
        logger.error("No email contents provided for PDF download")
        raise ValueError("No email contents to process")
    
    content = contents[0]
    msg_id = content['id']
    email = content['sender']
    body = content['body']
    
    if not content['attachments']:
        logger.error(f"No attachments found in email {msg_id}")
        raise ValueError("No attachments in email")
    
    attachment = content['attachments'][0]
    file_name = attachment['filename']
    attachment_id = attachment['attachment_id']
    
    logger.info(f"Downloading {file_name} from {email}")

    try:
        att = api_resource.users().messages().attachments().get(
            userId='me', 
            messageId=msg_id, 
            id=attachment_id
        ).execute()

        file_data = base64.urlsafe_b64decode(att['data'].encode('UTF-8'))

        path = os.path.join(save_path, file_name)
        with open(path, 'wb') as file:
            file.write(file_data)
        
        logger.info(f"Successfully saved PDF to {path}")
        return path, body
        
    except Exception as e:
        logger.error(f"Failed to download attachment: {e}")
        raise


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract all text content from a PDF file.
    
    Reads all pages from the PDF and combines them into a single
    text string. Removes extra whitespace for cleaner output.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content, or error message if failed
        
    Example:
        >>> text = extract_text_from_pdf("./downloads/resume.pdf")
        >>> print(f"Extracted {len(text)} characters")
    """
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    try:
        reader = PyPDF2.PdfReader(pdf_path)
        full_txt = ""

        logger.debug(f"Processing {len(reader.pages)} pages")

        for page in reader.pages:
            full_txt += page.extract_text() + "\n"
        
        clean_text = " ".join(full_txt.split())

        logger.info(f"Extracted {len(clean_text)} characters from PDF")
        return clean_text
    
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        return f"Error: PDF file not found at {pdf_path}"
    
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        return f"Error reading PDF: {str(e)}"
