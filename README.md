# Meta-Feedback UI

A minimal web interface for collecting and generating process oriented meta-feedback for algorithm assignments.  

## Features
- Simple sign-up and login system (SQLite-based)
- Students can draft and submit design plans, code, and tests
- Real-time AI-generated feedback (via OpenAI API)
- Admins can view and trigger feedback for each student
- Lightweight — no external framework required for frontend


## Prerequisites

- Python 3.9+
- An OpenAI API key

## Setup Instructions

### 1. Clone the repository
```bash
pip install fastapi uvicorn "sqlalchemy>=2" passlib[bcrypt] python-multipart openai python-dotenv
```

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-xxxx
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
DATABASE_URL=sqlite:///./meta_feedback.db
```

## Run

```bash
python server.py
```

Then open in your browser:

```
http://localhost:8000/app/index.html
```
## Share Publicly

```bash
ngrok http 8000
```
This provides a temporary public URL  

