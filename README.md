# 🤖 Autonomous Hiring Agent

> **An AI-powered ETL pipeline that automates the recruitment process by extracting structured data from resumes in Gmail and syncing it to Google Sheets.**

## 🚀 Overview
Reviewing hundreds of resumes manually is slow and error-prone. This agent acts as a **24/7 Digital Recruiter**. It monitors a Gmail inbox for job applications, downloads PDF resumes, validates them, and uses **Llama-3 (via Groq)** to extract key candidate details (Skills, Experience, Contact Info) into a structured Google Sheet.

## 🛠️ Tech Stack
* **Core:** Python 3.10+
* **AI/LLM:** Llama-3-8b-instant (via Groq API)
* **Frameworks:** LangChain, Pydantic (Data Validation)
* **Integrations:** Gmail API (Google Workspace), Google Sheets API
* **PDF Processing:** PyPDF2

## ✨ Key Features
* **Auto-Detection:** Polls Gmail every 60 minutes for emails with subject `"Application"`.
* **Smart Validation:** Uses a classifier agent to distinguish between valid Resumes and random documents (Invoices, Cover Letters).
* **Structured Extraction:** Converts messy PDF text into clean JSON (Name, Email, Skills, Exp) using `Pydantic`.
* **Robust Architecture:** Includes auto-retry logic, error logging, and continuous daemon mode.

## ⚙️ Setup & Usage

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/Rex-Code-debug/autonomous-hiring-agent.git](https://github.com/Rex-Code-debug/autonomous-hiring-agent.git)
    cd autonomous-hiring-agent
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure Environment:**
    * Create a `.env` file and add your `GROQ_API_KEY`.
    * Add your Google `credentials.json` to the root folder.

4.  **Run the Agent:**
    ```bash
    python main.py
    ```

## 📊 Output
**Google Sheet Example:**
| Name | Email | Skills | Experience | Status |
| :--- | :--- | :--- | :--- | :--- |
| Rahul Kumar | rahul@email.com | Python, LangChain, SQL | 2 Years | New |
---
in output add summary feature also

*Built by Dhananjay Mishra - Open for AI Engineering Roles*
