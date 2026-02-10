"""
Main entry point for Gmail Reader Agent.

This agent:
1. Searches Gmail for application emails with attachments
2. Downloads PDF attachments
3. VALIDATES if PDF is actually a resume
4. Extracts candidate data using LLM (only for valid resumes)
5. Saves data to Google Sheets
6. Logs rejected applications separately

Runs continuously every 1 hour.

Usage:
    python main.py
"""

import logging
from gmail_func import search_gmail, read_email, save_pdf, extract_text_from_pdf
from llm_func import extract_llm, save_to_sheets, save_rejected_to_sheets, agent_runner

logger = logging.getLogger(__name__)


@agent_runner
def main():
    """
    Main agent workflow for processing intern applications.
    
    Validates resumes before processing to avoid junk data.
    
    Workflow:
    1. Search Gmail for emails with subject 'application' and attachments
    2. Read email content and metadata
    3. Download PDF attachments
    4. Extract text from PDFs
    5. VALIDATE if PDF is actually a resume
    6. Use LLM to parse candidate data (only if valid resume)
    7. Save to Google Sheets (or rejection log)
    
    This function is wrapped with @agent_runner decorator which:
    - Runs it every 1 hour
    - Retries 3 times on failure
    - Logs all activity
    
    Returns:
        None
        
    Raises:
        Any exception raised will be caught by agent_runner decorator
    """
    logger.info("Starting main agent workflow")
    
    # Step 1: Search for emails
    msg_ids = search_gmail()

    if not msg_ids:
        logger.info("No new application emails found. Exiting.")
        return
    
    # Step 2: Read email content
    content = read_email(msg_ids)
    
    if not content:
        logger.warning("Failed to read any emails. Exiting.")
        return
    
    # Process each email (in case of multiple applications)
    for email_data in content:
        try:
            logger.info(f"Processing email from: {email_data['sender']}")
            
            # Step 3: Download PDF attachment
            if not email_data['attachments']:
                logger.warning(f"No attachments in email from {email_data['sender']}")
                continue
            
            # Process first attachment only (I can modify to handle multiples)
            try:
                path, body = save_pdf([email_data])
            except Exception as e:
                logger.error(f"Failed to download PDF: {e}")
                continue
            
            # Step 4: Extract text from PDF
            pdf_txt = extract_text_from_pdf(path)
            
            if pdf_txt.startswith("Error"):
                logger.error(f"PDF extraction failed: {pdf_txt}")
                continue
            
            # Step 5 & 6: Validate + Extract structured data using LLM
            data = extract_llm(body, pdf_txt)
            
            if not data:
                # Document was rejected (not a resume or low confidence)
                logger.warning(
                    f"Application from {email_data['sender']} rejected  "
                    f"not a valid resume"
                )
                
                # Log to rejected applications sheet
                save_rejected_to_sheets(
                    email_sender=email_data['sender'],
                    document_type="non-resume",
                    reason="Failed resume validation check"
                )
                
                continue
            
            # Step 7: Convert to dict and save to sheets
            dict_data = data.model_dump()
            
            try:
                save_to_sheets(dict_data,"candidates")
                logger.info(
                    f"Successfully processed application from {data.name} "
                    f"({email_data['sender']})"
                )
            except Exception as e:
                logger.error(f"Failed to save to Google Sheets: {e}")
                raise
        
        except Exception as e:
            logger.error(
                f"Error processing email from {email_data.get('sender', 'unknown')}: {e}",
                exc_info=True
            )
            continue
    
    logger.info("Completed processing all emails in this batch")


if __name__ == "__main__":
    """
    Entry point when running the script directly.
    
    Starts the agent in continuous mode (runs forever).
    """
    logger.info("="*60)
    logger.info("Gmail Reader Agent Starting (WITH RESUME VALIDATION)")
    logger.info("="*60)
    logger.info("Configuration:")
    logger.info(f"  - Search query: subject:application has:attachment")
    logger.info(f"  - Run interval: 1 hour")
    logger.info(f"  - Retry attempts: 3")
    logger.info(f"  - Download path: ./downloads/")
    logger.info(f"  - Valid resumes -> Sheet: intern_can")
    logger.info(f"  - Rejected docs -> Sheet: rejected_applications")
    logger.info(f"  - Resume validation: ENABLED")
    logger.info("="*60)
    
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Agent stopped by user (Ctrl+D)")
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        raise
