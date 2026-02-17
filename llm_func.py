"""
LLM extraction and Google Sheets integration functions.

This module provides:
- Resume validation (filters non-resume PDFs)
- LLM-based resume parsing using Groq
- Structured data extraction from email + PDF
- Google Sheets writing functionality
- Agent runner decorator for scheduled execution

"""

import logging
import functools
import time
from datetime import datetime
from typing import Optional, List, Dict, Literal
from config import settings, client
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import gspread

logger = logging.getLogger(__name__)

api_key = settings.groq_api_key

# ============================================================================
# LLM SETUP
# ============================================================================
"""
I made model here because if we want to change model we can from top we don't
had to find in between functions.
"""
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=api_key,
    max_retries=3
)

logger.info("LLM initialized: llama-3.1-8b-instant")

# ============================================================================
# DATA MODELS
# ============================================================================

class ResumeValidation(BaseModel):
    """
    Model for validating if a document is actually a resume.
    
    Used as a pre-check before attempting full data extraction.
    
    Attributes:
        is_resume: Whether the document appears to be a resume
        confidence: Confidence level (high/medium/low)
        document_type: What type of document it actually is
        reason: Brief explanation of the classification
    """
    is_resume: bool = Field(description="True if document is a resume/CV, False otherwise")
    confidence: Literal['high', 'medium', 'low'] = Field(description="Confidence in classification")
    document_type: str = Field(description="Type of document: resume, cover_letter, portfolio, invoice, other, junk")
    reason: str = Field(description="Brief reason for classification (1 sentence)")


class Interns(BaseModel):
    """
    Structured data model for intern candidate information.
    
    This schema is used by the LLM to extract and validate
    candidate data from resumes and application emails.
    
    Attributes:
        name: Full name of candidate
        email: Email address (or 'N/A' if not found)
        phone: Phone number as string (or 'N/A' if not found)
        skills: List of top 5 technical skills
        exp: Total years of experience (e.g., '2 years', 'Fresher')
        status: Application status (always 'New' for incoming)
        summary: 2-sentence candidate profile summary
        question: add context-aware candidate assessment engine for automated technical screening
    """
    name: str = Field(description="Full name of the candidate found in resume or email")
    email: str = Field(description="Email address. If not found, return 'N/A'")
    phone: str = Field(description="Phone number as a string (e.g., '+91-999...'). If not found, return 'N/A'") 
    skills: List[str] = Field(description="List of top 5 technical skills mentioned")
    exp: str = Field(description="Total years of experience (e.g., '2 years', 'Fresher'). Default to '0 years'")
    status: Literal['New', 'Old'] = Field(description="Always return 'New' for incoming applications")
    summary: str = Field(description="A concise 2-sentence summary of the candidate's profile")
    question: str = Field(description="3 to 5 question for asking from candidate from its weakness")


# ============================================================================
# DECORATOR FOR AGENT SCHEDULING
# ============================================================================

def agent_runner(func):
    """
    Decorator to run agent function on a schedule with retry logic.
    
    Runs the wrapped function every 1 hour indefinitely.
    Includes 3 retry attempts with 10-second delays on failure.
    
    Args:
        func: Function to run on schedule
        
    Returns:
        Wrapped function that runs in infinite loop
        
    Example:
        @agent_runner
        def my_agent():
            print("Running agent task")
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        while True:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            logger.info(f"========== Agent Scan Started: {timestamp} ==========")
            
            success = False
            for attempt in range(1, 4):
                try:
                    func(*args, **kwargs)
                    logger.info("Agent task completed successfully")
                    success = True
                    break

                except Exception as e:
                    logger.error(f"Attempt {attempt}/3 failed: {e}", exc_info=True)
                    if attempt < 3:
                        time.sleep(10)
            
            if not success:
                logger.error("All 3 attempts failed. Skipping to next cycle.")

            logger.info("Agent going to sleep for 1 hour")
            time.sleep(3600)
            
    return wrapper


# ============================================================================
# RESUME VALIDATION FUNCTION
# ============================================================================

def validate_resume(pdf_text: str, email_body: str = "") -> ResumeValidation:
    """
    Validate if the PDF is actually a resume before extracting data.
    
    Uses LLM to classify the document type and prevent processing
    of non-resume files (cover letters, invoices, portfolios, etc.)
    
    Args:
        pdf_text (str): Extracted text from PDF
        email_body (str): Email content (optional, provides context)
        
    Returns:
        ResumeValidation: Classification result with confidence level
        
    Example:
        >>> validation = validate_resume(pdf_text)
        >>> if validation.is_resume and validation.confidence == 'high':
        >>>     # Proceed with extraction
    """
    logger.info("Validating if document is a resume")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a document classifier. Your job is to determine if a document is a RESUME/CV.
        
        A RESUME typically contains:
        - Personal information (name, contact)
        - Work experience or education history
        - Skills section
        - Professional summary or objective
        
        NOT a resume:
        - Cover letters (focus on why applying, company-specific)
        - Portfolios (collection of work samples, projects)
        - Invoices, receipts, financial documents
        - Company brochures or marketing materials
        - Random PDFs, junk documents
        
        Be strict: If it doesn't clearly look like a resume, mark is_resume=False.
        """),
        ("user", """
        Classify this document:
        
        === EMAIL CONTEXT ===
        {email_content}
        
        === DOCUMENT TEXT (first 1000 chars) ===
        {document_preview}
        
        Is this a resume/CV?
        """)
    ])

    structured_llm = llm.with_structured_output(ResumeValidation)
    chain = prompt | structured_llm

    try:
        # Use first 1000 chars to save tokens
        preview = pdf_text[:1000] if len(pdf_text) > 1000 else pdf_text
        
        result = chain.invoke({
            "email_content": email_body[:500],  # Brief context
            "document_preview": preview
        })
        
        logger.info(
            f"Validation result: is_resume={result.is_resume}, "
            f"confidence={result.confidence}, type={result.document_type}"
        )
        logger.debug(f"Reason: {result.reason}")
        
        return result
        
    except Exception as e:
        logger.error(f"Resume validation failed: {e}", exc_info=True)
        # On error, assume it's NOT a resume (safe default)
        return ResumeValidation(
            is_resume=False,
            confidence='low',
            document_type='unknown',
            reason=f"Validation error: {str(e)}"
        )


# ============================================================================
# LLM EXTRACTION FUNCTION
# ============================================================================

def extract_llm(email_body: str, pdf_text: str, skip_validation: bool = False) -> Optional[Interns]:
    """
    Extract structured candidate data using LLM.
    
    NEW: Now includes resume validation by default.
    Will reject non-resume documents automatically.
    
    Sends email body and resume text to Groq LLM for parsing.
    Uses structured output to ensure data matches Interns schema.
    
    Args:
        email_body (str): Text content of application email
        pdf_text (str): Extracted text from resume PDF
        skip_validation (bool): Skip resume validation (default: False)
        
    Returns:
        Interns: Structured candidate data object, or None if:
            - Document is not a resume
            - Validation confidence is low
            - Extraction fails
        
    Example:
        >>> data = extract_llm("Applying for role", "John Doe, Python Dev...")
        >>> if data:
        >>>     print(data.name)  # Only if valid resume
    """
    logger.info("Starting LLM extraction for candidate data")
    
    # STEP 1: Validate it's actually a resume
    if not skip_validation:
        validation = validate_resume(pdf_text, email_body)
        
        if not validation.is_resume:
            logger.warning(
                f"Document rejected - Not a resume. "
                f"Type: {validation.document_type}, Reason: {validation.reason}"
            )
            return None
        
        if validation.confidence == 'low':
            logger.warning(
                f"Document rejected - Low confidence validation. "
                f"Type: {validation.document_type}"
            )
            return None
        
        logger.info(f"Resume validation passed with {validation.confidence} confidence")
    
    # STEP 2: Extract candidate data
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an expert Resume Parser for a hiring agency. 
        Your job is to extract structured candidate data from their Email and Resume.
        
        RULES:
        1. Prioritize information found in the RESUME. Use EMAIL content as backup.
        2. If a field is missing, use the default value specified in the schema.
        3. For 'skills', extract the most relevant technical skills (Python, AI, Agents, etc.).
        4. Always set 'status' to 'New'.
        5. Make 3 to 5 interview questions that are the weakness of candidate that recuritor can ask while taking interview from candidate  
        """),
        ("user", """
        Here is the candidate data:
        
        === EMAIL BODY ===
        {email_content}
        
        === RESUME TEXT ===
        {resume_content}
        """)
    ])

    structured_llm = llm.with_structured_output(Interns)
    chain = prompt | structured_llm

    try:
        result = chain.invoke({
            "email_content": email_body, 
            "resume_content": pdf_text
        })
        
        logger.info(f"Successfully extracted data for candidate: {result.name}")
        logger.debug(f"Extracted skills: {', '.join(result.skills)}")
        
        return result
        
    except Exception as e:
        logger.error(f"LLM extraction failed: {e}", exc_info=True)
        return None


# ============================================================================
# GOOGLE SHEETS INTEGRATION
# ============================================================================

def save_to_sheets(data: Dict, file_name: str = "intern_can"):
    """
    Save candidate data to Google Sheets.
    
    Appends a new row to the spreadsheet. If the sheet doesn't exist,
    creates it with headers and shares it with the configured email.
    
    Args:
        data (Dict): Candidate data dictionary with keys matching Interns model
        file_name (str): Name of Google Sheet (default: 'intern_can')
        
    Raises:
        Exception: If sheet creation or writing fails
        
    Note:
        Creates sheet with columns: Name, Email, Phone, Skills, Experience, Status, Summary
        Shares new sheets with: mdhananjay776@gmail.com
    """
    logger.info(f"Saving candidate data to Google Sheet: {file_name}")
    
    name = data.get('name', "")
    email = data.get('email', "")
    phone = data.get('phone', "")
    skills = data.get('skills', [])
    exp = data.get('exp', "")
    status = data.get('status', "")
    summary = data.get('summary', "")
    question = data.get('question',"")

    
    if isinstance(skills, list):
        skills_str = ", ".join(skills)
    else:
        skills_str = str(skills)

    row = [name, email, phone, skills_str, exp, status, summary, question]
    
    logger.debug(f"Row to insert: {row}")

    try:
        # Try to open existing sheet
        sh = client.open(file_name)
        logger.info(f"Found existing sheet: '{file_name}'")
        
        ws = sh.sheet1
        ws.append_row(row)

        logger.info(f"Successfully added row for {name}")

    except gspread.exceptions.SpreadsheetNotFound:
        # Create new sheet if not found
        logger.warning(f"Sheet not found. Creating new sheet: '{file_name}'")
        
        sh = client.create(file_name)
        sh.share('mdhananjay776@gmail.com', perm_type='user', role='writer')

        ws = sh.sheet1
        
        # Add headers
        headers = ["Name", "Email", "Phone", "Skills", "Experience", "Status", "Summary","questions"]
        ws.append_row(headers)
        
        logger.info(f"Created new sheet with headers: {headers}")

        # Add data row
        ws.append_row(row)
        logger.info(f"Successfully added first row for {name}")


def save_rejected_to_sheets(email_sender: str, document_type: str, reason: str, file_name: str = "rejected_applications"):
    """
    Save rejected applications to a separate tracking sheet.
    
    Useful for auditing and understanding what types of files are being sent.
    
    Args:
        email_sender (str): Email address of sender
        document_type (str): Type of document detected (cover_letter, invoice, etc.)
        reason (str): Rejection reason from validation
        file_name (str): Sheet name (default: 'rejected_applications')
    """
    logger.info(f"Logging rejected application from {email_sender}")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, email_sender, document_type, reason]
    
    try:
        sh = client.open(file_name)
        ws = sh.sheet1
        ws.append_row(row)
        logger.info(f"Logged rejection for {email_sender}")
        
    except gspread.exceptions.SpreadsheetNotFound:
        logger.info(f"Creating rejected applications tracking sheet")
        sh = client.create(file_name)
        sh.share('mdhananjay776@gmail.com', perm_type='user', role='writer')
        ws = sh.sheet1
        ws.append_row(["Timestamp", "Sender", "Document Type", "Rejection Reason"])
        ws.append_row(row)
        logger.info("Created rejection tracking sheet")



# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    """
    Test the LLM extraction and validation with fake data.
    
    Run this file directly to test without email/PDF:
    python llm_func.py
    """
    logger.info("Running test mode with fake data")
    
    print("\n" + "="*60)
    print("TEST 1: Valid Resume")
    print("="*60)
    
    fake_email = "Hi, applying for the intern role. Please find my resume attached."
    fake_resume = """
    Dhananjay Kumar
    Email: dhananjay@example.com
    Phone: +91-999-888-7777
    
    EXPERIENCE:
    Python Developer - 2 years
    Built 3 Agentic AI Systems using LangChain
    
    SKILLS:
    Python, AI, Machine Learning, LangChain, FastAPI
    
    EDUCATION:
    B.Tech Computer Science
    """
    
    data = extract_llm(fake_email, fake_resume)
    
    if data:
        print("✅ VALID RESUME - Data Extracted:")
        print(data.model_dump_json(indent=2))
    else:
        print("❌ Resume rejected")
    
    print("\n" + "="*60)
    print("TEST 2: Cover Letter (Should Reject)")
    print("="*60)
    
    fake_cover_letter = """
    Dear Hiring Manager,
    
    I am writing to express my strong interest in the Software Engineering
    position at your company. With my background in computer science and
    passion for technology, I believe I would be a great fit for your team.
    
    I have always admired your company's commitment to innovation...
    
    Thank you for considering my application.
    
    Sincerely,
    John Doe
    """
    
    data2 = extract_llm(fake_email, fake_cover_letter)
    
    if data2:
        print("❌ ERROR: Cover letter was accepted (should reject)")
    else:
        print("✅ CORRECT: Cover letter rejected")
    
    print("\n" + "="*60)
    print("TEST 3: Invoice (Should Reject)")
    print("="*60)
    
    fake_invoice = """
    INVOICE #12345
    Date: 2024-01-15
    
    Bill To:
    Company XYZ
    123 Street
    
    Services Rendered:
    - Web Development: $500
    - Consulting: $300
    
    Total: $800
    """
    
    data3 = extract_llm(fake_email, fake_invoice)
    
    if data3:
        print("❌ ERROR: Invoice was accepted (should reject)")
    else:
        print("✅ CORRECT: Invoice rejected")
    
    print("\n" + "="*60)
