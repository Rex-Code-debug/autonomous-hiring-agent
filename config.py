"""
Configuration module for Gmail Reader Agent.

This module sets up:
- Google API credentials (Gmail + Sheets)
- Application settings from environment variables
- Logging configuration
"""

import sys
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_google_community.gmail.utils import build_gmail_service, get_google_credentials
import gspread

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """
    Configure logging for the application.
    
    Creates a logs directory and sets up both file and console logging.
    Log file: logs/gmail_agent.log
    Format: timestamp - level - module - message
    """
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "gmail_agent.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

# ============================================================================
# GOOGLE API CREDENTIALS
# ============================================================================

scopes = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

try:
    credentials = get_google_credentials(
        token_file="token.json",
        scopes=scopes,
        client_secrets_file="credentials.json",
    )
    logger.info("Google credentials loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Google credentials: {e}")
    raise

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

class Settings(BaseSettings):
    """
    Application configuration settings.
    
    Loads from .env file:
    - groq_api_key: API key for Groq LLM
    - google_api_key: Google API key (if needed separately)
    - max_retries: Number of retry attempts for failed operations
    - debug_mode: Enable debug logging
    """
    app_name: str = "gmail_reader_agent"
    groq_api_key: str
    google_api_key: str
    max_retries: int = 3
    debug_mode: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
logger.info(f"Settings loaded: {settings.app_name}, debug_mode={settings.debug_mode}")

# ============================================================================
# API RESOURCES
# ============================================================================

try:
    api_resource = build_gmail_service(credentials=credentials)
    logger.info("Gmail API service built successfully")
except Exception as e:
    logger.error(f"Failed to build Gmail service: {e}")
    raise

try:
    client = gspread.authorize(credentials)
    logger.info("Google Sheets client authorized successfully")
except Exception as e:
    logger.error(f"Failed to authorize Sheets client: {e}")
    raise
