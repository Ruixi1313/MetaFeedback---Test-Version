# Meta-Feedback UI

A minimal FastAPI + SQLite application for collecting and generating process-oriented meta-feedback for programming assignments. Students draft plans/code/tests inside the browser, receive AI guidance, and instructors can review the full history.

## Features
- SQLite-backed auth with JWT-based sessions
- Drafting surface for plans, code, and tests with version snapshots
- Real-time OpenAI feedback generation with configurable model
- Admin-only panels for triggering feedback, inspecting logs, and pre/post tests
- Pure HTML/CSS/JS frontend served by FastAPI static files (no build step)

## Tech Stack
- FastAPI, Pydantic, SQLAlchemy 2.x, Passlib (bcrypt), python-jose
- SQLite by default (switchable via SQLAlchemy URL)
- OpenAI API for feedback generation
- Optional `ngrok` tunnel for sharing with remote testers

## Quick Start

### Prerequisites
- Python 3.9 or newer
- `pip` and (optionally) `python -m venv`
- An OpenAI API key with access to the configured model
- (Optional) `ngrok` account for public tunneling

### 1. Clone the repository
```bash
git clone <repo-url> meta-feedback-ui
cd meta-feedback-ui
```

### 2. Create & activate a virtual environment (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# or, if you prefer manual installs:
# pip install fastapi uvicorn "sqlalchemy>=2" passlib[bcrypt] python-multipart openai python-dotenv python-jose[cryptography]
```

### 4. Configure environment variables
Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=sk-your-key
OPENAI_MODEL=gpt-4o-mini        # Optional override
ADMIN_USERNAME=admin            # Change for production
ADMIN_PASSWORD=admin123         # Change for production
DATABASE_URL=sqlite:///./meta_feedback.db
UI_DIR=./                       # Optional; override static asset directory
```

Additional flags used by `server.py` (set if you need overrides):
- `SECRET_KEY` – JWT signing key (auto-generated if omitted)
- `ACCESS_TOKEN_EXPIRE_MINUTES` – default 1440 (24h)
- `OPENAI_MAX_TOKENS`, `OPENAI_TEMPERATURE`, etc. – see `server.py`

### 5. Initialize / inspect the database (optional)
The first start creates `meta_feedback.db` automatically. Delete the file to reset or point `DATABASE_URL` at Postgres/MySQL as needed.

### 6. Run the development server
```bash
python server.py
# or
# uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

### 7. Open the UI
Visit `http://localhost:8000/app/index.html`. The FastAPI app serves static assets from `UI_DIR`, so adjust that env var if you keep the frontend elsewhere.

## Sharing with Others

Once the server is running locally:
```bash
ngrok http 8000
```
Copy the forwarded HTTPS URL and share `https://<random>.ngrok-free.app/app/index.html`. Remember that anyone with this link can reach your instance—never expose real admin credentials or unmasked API keys.

## Project Structure (trimmed)
```
.
├── server.py          # FastAPI application (REST + static serving)
├── README.md
├── requirements.txt
├── index.html / app/  # Frontend assets served under /app
├── meta_feedback.db   # SQLite db (created at runtime)
└── .env               # Not committed; holds secrets
```

## Troubleshooting
- **401s after login** – ensure `SECRET_KEY` stays consistent between restarts and that the server time is correct.
- **OpenAI errors** – verify `OPENAI_API_KEY`, model availability, and network egress.
- **Static files not updating** – set `UI_DIR` to the absolute path of your frontend build or clear browser cache.
- **Different database** – update `DATABASE_URL` (e.g., `postgresql+psycopg://user:pass@host/db`) and install the matching driver.

## Security Checklist
- Replace default admin credentials before exposing to students.
- Store `.env` outside version control and rotate credentials regularly.
- Restrict ngrok links to trusted testers; stop the tunnel when finished.

## Contributing
1. Fork and branch from `main`.
2. Make changes, add/adjust tests if applicable.
3. Run `ruff`/`pytest` (if configured) and open a PR summarizing changes + screenshots of UI updates.

