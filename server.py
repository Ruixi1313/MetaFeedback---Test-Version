# server.py
# FastAPI backend with SQLite + persistent drafts & feedback history
# Run: pip install fastapi uvicorn "openai>=1.40.0" pydantic python-jose bcrypt python-multipart sqlalchemy
# Then: uvicorn server:app --reload --port 8000
# Env: export OPENAI_API_KEY=sk-... ; export SECRET_KEY= ;

import os
from datetime import datetime, timedelta
import pytz
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pydantic import BaseModel
from jose import JWTError, jwt
from openai import OpenAI
import bcrypt

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DATABASE_URL = "sqlite:///./meta_feedback.db"

# PST timezone helper
PST = pytz.timezone('America/Los_Angeles')

def get_pst_now():
    """Get current time in PST timezone (stored as naive local PST)."""
    return datetime.now(PST).replace(tzinfo=None)

def get_utc_now():
    """Get current time in UTC (standard for database storage)"""
    return datetime.utcnow()

def to_pst_string(dt):
    """Format datetime as PST string. If naive, assume it's already PST.
    If timezone-aware, convert to PST before formatting.
    """
    if dt is None:
        return ""
    if dt.tzinfo is None:
        return dt.strftime('%m/%d/%Y, %H:%M:%S')
    pst_dt = dt.astimezone(PST)
    return pst_dt.strftime('%m/%d/%Y, %H:%M:%S')

client = None  # Will be initialized when needed
security = HTTPBearer()

# DB
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    feedback_enabled = Column(Boolean, default=False)
    feedback_count = Column(Integer, default=0)  # Track number of feedbacks received
    feedback_assignment = Column(String, nullable=True)  # Assignment where feedback is allowed
    created_at = Column(DateTime, default=get_pst_now)

class ConfidenceIn(BaseModel):
    assignment: str
    domain: str
    confidence_level: int  # 0..100


class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    assignment = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    plan = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    tests = Column(Text, nullable=True)
    confidence_level = Column(Integer, nullable=True)  # 0-100 percentage
    # Per-part evaluator results
    a_correct = Column(Boolean, nullable=True)
    b_correct = Column(Boolean, nullable=True)
    c_correct = Column(Boolean, nullable=True)
    a_reason = Column(Text, nullable=True)
    b_reason = Column(Text, nullable=True)
    c_reason = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=get_pst_now)

class Draft(Base):
    __tablename__ = "drafts"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    assignment = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    plan = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    tests = Column(Text, nullable=True)
    feedback_md = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=get_pst_now, onupdate=get_pst_now)
    __table_args__ = ({"sqlite_autoincrement": True},)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    content_md = Column(Text)
    source = Column(String(32), default="instant")  
    assignment_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=get_pst_now)

class DraftHistory(Base):
    __tablename__ = "draft_history"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    assignment = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    plan = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    tests = Column(Text, nullable=True)
    feedback_md = Column(Text, nullable=True)
    version_tag = Column(String(32), nullable=True) 
    created_at = Column(DateTime, default=get_pst_now)

class EventLog(Base):
    __tablename__ = "event_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    event_type = Column(String(64), nullable=False) 
    assignment = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    details = Column(Text, nullable=True) 
    created_at = Column(DateTime, default=get_pst_now)

class Question(Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    assignment = Column(String, nullable=False, index=True)
    domain = Column(String, nullable=False, index=True)
    question_text = Column(Text, nullable=False)
    rubric = Column(Text, nullable=True)
    created_at = Column(DateTime, default=get_pst_now)
    updated_at = Column(DateTime, default=get_pst_now, onupdate=get_pst_now)

# Pretest (admin-configured single-choice items + per-user result)
class PretestQuestion(Base):
    __tablename__ = "pretest_questions"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    options_json = Column(Text, nullable=False)  # JSON array of strings
    correct_index = Column(Integer, nullable=False)  # 0-based
    created_at = Column(DateTime, default=get_pst_now)

class PretestResult(Base):
    __tablename__ = "pretest_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    answers_json = Column(Text, nullable=False)  # JSON: [{id, choice_index}]
    num_correct = Column(Integer, nullable=False)
    completed_at = Column(DateTime, default=get_pst_now)

# Posttest (admin-configured single-choice items + per-user result)
class PosttestQuestion(Base):
    __tablename__ = "posttest_questions"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(Text, nullable=False)
    options_json = Column(Text, nullable=False)  # JSON array of strings
    correct_index = Column(Integer, nullable=False)  # 0-based
    created_at = Column(DateTime, default=get_pst_now)

class PosttestResult(Base):
    __tablename__ = "posttest_results"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    answers_json = Column(Text, nullable=False)  # JSON: [{id, choice_index}]
    num_correct = Column(Integer, nullable=False)
    completed_at = Column(DateTime, default=get_pst_now)

Base.metadata.create_all(bind=engine)

# App 
app = FastAPI(title="AI Meta-Feedback API", version="3.0")

# Add global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    print(f"Global exception handler caught: {type(exc).__name__}: {exc}")
    return {"detail": f"Server error: {str(exc)}", "error_type": type(exc).__name__}
# app.mount("/app", StaticFiles(directory="/Users/ruixilin/Desktop/Meta_testversion", html=True), name="app")
ui_dir = os.getenv("UI_DIR", os.path.join(os.path.dirname(__file__), "."))  
app.mount("/app", StaticFiles(directory=ui_dir, html=True), name="app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utils
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    password = password[:72]
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    plain_password = plain_password[:72]
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = get_utc_now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def log_event(db: Session, user_id: int, event_type: str, assignment: str = None, domain: str = None, details: str = None):
    event = EventLog(
        user_id=user_id,
        event_type=event_type,
        assignment=assignment,
        domain=domain,
        details=details
    )
    db.add(event)
    db.commit()

def create_draft_snapshot(db: Session, user_id: int, assignment: str, domain: str, plan: str = None, code: str = None, tests: str = None, feedback_md: str = None, version_tag: str = None):
    snapshot = DraftHistory(
        user_id=user_id,
        assignment=assignment,
        domain=domain,
        plan=plan,
        code=code,
        tests=tests,
        feedback_md=feedback_md,
        version_tag=version_tag
    )
    db.add(snapshot)
    db.commit()
    return snapshot

def should_create_snapshot(db: Session, user_id: int, assignment: str, domain: str, new_plan: str, new_code: str, new_tests: str) -> bool:
    # Get the latest snapshot for this assignment
    latest = db.query(DraftHistory).filter(
        DraftHistory.user_id == user_id,
        DraftHistory.assignment == assignment,
        DraftHistory.domain == domain
    ).order_by(DraftHistory.created_at.desc()).first()
    
    if not latest:
        return True 
    
    if (latest.plan != new_plan or 
        latest.code != new_code or 
        latest.tests != new_tests):
        return True
    
    return False

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        user = db.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication")

async def get_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Startup
@app.on_event("startup")
async def startup_event():
    print("DEBUG: Startup event triggered")
    db = SessionLocal()
    try:
        print("DEBUG: Database session created")
        # Check if database needs migration
        try:
            db.query(Draft.assignment).first()
            print("Database schema is up to date")

            # --- lightweight migration: add confidence_level column if missing ---
            from sqlalchemy import text
            try:
                with engine.connect() as conn:
                    cols = conn.execute(text("PRAGMA table_info(submissions);")).fetchall()
                    col_names = {c[1] for c in cols}  # the second element is the column name
                    if "confidence_level" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN confidence_level INTEGER;"))
                        print("Added 'confidence_level' column to submissions")
                    # Add per-part evaluator columns if missing
                    if "a_correct" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN a_correct BOOLEAN;"))
                        print("Added 'a_correct' column to submissions")
                    if "b_correct" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN b_correct BOOLEAN;"))
                        print("Added 'b_correct' column to submissions")
                    if "c_correct" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN c_correct BOOLEAN;"))
                        print("Added 'c_correct' column to submissions")
                    if "a_reason" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN a_reason TEXT;"))
                        print("Added 'a_reason' column to submissions")
                    if "b_reason" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN b_reason TEXT;"))
                        print("Added 'b_reason' column to submissions")
                    if "c_reason" not in col_names:
                        conn.execute(text("ALTER TABLE submissions ADD COLUMN c_reason TEXT;"))
                        print("Added 'c_reason' column to submissions")
                    # Add per-part open/pass timestamps
                    for col in ["a_opened_at","a_passed_at","b_opened_at","b_passed_at","c_opened_at","c_passed_at"]:
                        if col not in col_names:
                            conn.execute(text(f"ALTER TABLE submissions ADD COLUMN {col} TEXT;"))
                            print(f"Added '{col}' column to submissions")
            except Exception as e:
                print("Could not alter table for confidence_level:", e)

            # Add feedback_assignment column to users if missing
            try:
                with engine.connect() as conn:
                    ucols = conn.execute(text("PRAGMA table_info(users);")).fetchall()
                    user_col_names = {c[1] for c in ucols}
                    if "feedback_assignment" not in user_col_names:
                        conn.execute(text("ALTER TABLE users ADD COLUMN feedback_assignment TEXT;"))
                        print("Added 'feedback_assignment' column to users")
            except Exception as e:
                print("Could not alter users table for feedback_assignment:", e)

            # Ensure pretest tables exist
            try:
                Base.metadata.create_all(bind=engine)
            except Exception as e:
                print("Could not create pretest tables:", e)

        except Exception as e:
            print("Database schema needs migration. Please delete meta_feedback.db and restart the server.")
            print(f"Error: {e}")
            return
        
        print("DEBUG: Checking for admin user")
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("DEBUG: Admin user not found, creating...")
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                is_admin=True,
                feedback_enabled=False,
            )
            db.add(admin_user)
            db.commit()
            print("✅ Admin user created: username='admin', password='admin123'")
        else:
            print("DEBUG: Admin user already exists")
    except Exception as e:
        print(f"DEBUG: Startup error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()
        print("DEBUG: Startup completed")
        
@app.middleware("http")
async def no_cache_html(request, call_next):
    resp = await call_next(request)
    if request.url.path.endswith(".html"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


# Schemas
class UserCreate(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str
    is_admin: bool

class ConfidenceIn(BaseModel):
    assignment: str
    domain: str
    confidence_level: int

class SubmissionCreate(BaseModel):
    assignment: str
    domain: str
    plan: str
    code: str
    tests: str
    confidence_level: Optional[int] = None

class SubmissionResponse(BaseModel):
    id: int
    username: str
    assignment: str
    domain: str
    plan: str
    code: str
    tests: str
    confidence_level: Optional[int] = None
    timestamp: str
    has_feedback: bool = False

class PretestQuestionIn(BaseModel):
    question_text: str
    options: list[str]
    correct_index: int

class PretestQuestionOut(BaseModel):
    id: int
    question_text: str
    options: list[str]

class PretestQuestionOutAdmin(PretestQuestionOut):
    correct_index: int

class PretestSubmitIn(BaseModel):
    answers: list[dict]  # [{ "id": int, "choice_index": int }]

class PretestStatusOut(BaseModel):
    completed: bool
    num_correct: Optional[int] = None
    class Config:
        from_attributes = True

# Posttest schemas
class PosttestQuestionIn(BaseModel):
    question_text: str
    options: list[str]
    correct_index: int

class PosttestQuestionOut(BaseModel):
    id: int
    question_text: str
    options: list[str]

class PosttestQuestionOutAdmin(PosttestQuestionOut):
    correct_index: int

class PosttestSubmitIn(BaseModel):
    answers: list[dict]  # [{ "id": int, "choice_index": int }]

class PosttestStatusOut(BaseModel):
    completed: bool
    num_correct: Optional[int] = None
    class Config:
        from_attributes = True

class FeedbackToggle(BaseModel):
    username: str
    enabled: bool

class FeedbackGlobalToggle(BaseModel):
    enabled: bool

# REPLACE the existing AnalyzeResponse with this:
class AnalyzeResponse(BaseModel):
    plan_suggestions: List[str]
    code_suggestions: List[str]
    test_suggestions: List[str]
    summary: Optional[str] = None
class EvaluateRequest(BaseModel):
    assignment: str
    domain: str
    part: str  # 'A' (Plan), 'B' (Code), 'C' (Tests)
    plan: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None

class EvaluateResponse(BaseModel):
    is_correct: bool
    reason: str
class ClarifyIn(BaseModel):
    assignment: str
    domain: str
    part: str  # 'A' (Plan), 'B' (Code), 'C' (Tests)
    question: str

class ClarifyOut(BaseModel):
    answer: str




class DraftIn(BaseModel):
    assignment: Optional[str] = None
    domain: Optional[str] = None
    plan: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None

class DraftOut(DraftIn):
    feedback_md: Optional[str] = None
    updated_at: Optional[str] = None

class DraftHistoryResponse(BaseModel):
    id: int
    assignment: str
    domain: str
    plan: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None
    feedback_md: Optional[str] = None
    version_tag: Optional[str] = None
    created_at: str
    class Config:
        from_attributes = True

class EventLogCreate(BaseModel):
    event_type: str
    assignment: Optional[str] = None
    domain: Optional[str] = None
    details: Optional[str] = None

class EventLogResponse(BaseModel):
    id: int
    event_type: str
    assignment: Optional[str] = None
    domain: Optional[str] = None
    details: Optional[str] = None
    created_at: str
    class Config:
        from_attributes = True

class QuestionCreate(BaseModel):
    assignment: str
    domain: str
    question_text: str
    rubric: Optional[str] = None

class QuestionUpdate(BaseModel):
    assignment: Optional[str] = None
    domain: Optional[str] = None
    question_text: Optional[str] = None
    rubric: Optional[str] = None

class QuestionResponse(BaseModel):
    id: int
    assignment: str
    domain: str
    question_text: str
    rubric: Optional[str] = None
    created_at: str
    updated_at: str
    class Config:
        from_attributes = True

class PartSubmissionRequest(BaseModel):
    assignment: str
    domain: str
    part: str  # 'A', 'B', or 'C'
    plan: Optional[str] = None
    code: Optional[str] = None
    tests: Optional[str] = None

class PartSubmissionResponse(BaseModel):
    success: bool
    message: str
    part: str

class PartOpenIn(BaseModel):
    assignment: str
    domain: str
    part: str  # 'A','B','C'

# LLM prompts
SYSTEM_PROMPT = """
You are an educational assistant that provides Meta‑Feedback.
Goal: help students improve their Plan, Code, and Tests with clear, actionable guidance.
Principles:
- Be specific, supportive, and concise (what to change and why it helps).
- Do NOT provide full solutions or pasteable code; prefer schematic hints.
- For Tests, use natural language (what to cover, edge cases, expected outcomes).
"""

USER_TEMPLATE = """Domain: {domain}

[ASSIGNMENT QUESTION] (context only; do not reveal or restate)
{question}

[DESIGN_PLAN.md]
{plan}

[Code]
{code}

[Tests]
{tests}

Provide Meta‑Feedback only (process‑level). Focus on how to improve approach, self‑monitoring, and feedback‑seeking — not on giving the answer.
Be actionable and concise. Do not give full solutions or pasteable code.
For each section (Plan, Code, Tests), produce 4–7 suggestions. Each suggestion can be 2–4 sentences and may include:
- a brief checklist or workflow to self‑check,
- a schematic hint (no pasteable code) for direction,
- what feedback to request next and how to verify it,
- how to transfer the habit to future tasks.
Return ONLY valid JSON:
{{
  "plan_suggestions": ["string"],
  "code_suggestions": ["string"],
  "test_suggestions": ["string"]
}}

"""

def call_openai(domain: str, plan: str, code: str, tests: str, question_text: str = None) -> Dict[str, Any]:
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"DEBUG: API key found: {api_key[:10] if api_key else 'None'}...")
        if not api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured. Please set OPENAI_API_KEY environment variable.")
        try:
            client = OpenAI(api_key=api_key)
            print("DEBUG: OpenAI client initialized successfully")
        except Exception as e:
            print(f"DEBUG: OpenAI client initialization failed: {e}")
            raise HTTPException(status_code=500, detail=f"OpenAI client initialization failed: {e}")
    
    user_msg = USER_TEMPLATE.format(
        domain=(domain or "").strip(),
        plan=(plan or "").strip(),
        code=(code or "").strip(),
        tests=(tests or "").strip(),
        question=(question_text or "").strip(),
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        import json
        content = resp.choices[0].message.content
        data = json.loads(content)
        # INSIDE call_openai, after json.loads(content)
        for k in ("plan_suggestions","code_suggestions","test_suggestions"):
            vals = data.get(k, [])
            if not isinstance(vals, list):
                vals = []
            vals = [s for s in vals if isinstance(s, str) and len(s.strip()) >= 20]
            data[k] = vals[:4]
        return data

    except Exception as e:
        # Handle specific OpenAI authentication errors
        if "invalid_api_key" in str(e) or "Incorrect API key" in str(e):
            raise HTTPException(status_code=500, detail="OpenAI API key is invalid. Please check your OPENAI_API_KEY environment variable.")
        elif "authentication" in str(e).lower():
            raise HTTPException(status_code=500, detail="OpenAI authentication failed. Please check your API key.")
        else:
            raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

# Routes
@app.get("/")
def root():
    # redirect to the UI
    return RedirectResponse(url="/app/index.html")

# Auth
@app.post("/signup", response_model=Token)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    exist = db.query(User).filter(User.username == user.username).first()
    if exist:
        raise HTTPException(status_code=400, detail="Username already exists")
    new_user = User(
        username=user.username,
        hashed_password=get_password_hash(user.password),
        is_admin=False,
        feedback_enabled=False,
    )
    db.add(new_user)
    db.commit(); db.refresh(new_user)
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username, "is_admin": False}

@app.post("/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    try:
        print(f"DEBUG: Attempting login for user: {user.username}")
        u = db.query(User).filter(User.username == user.username).first()
        print(f"DEBUG: User found: {u is not None}")
        if not u:
            print("DEBUG: User not found")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        print(f"DEBUG: Checking password for user: {u.username}")
        if not verify_password(user.password, u.hashed_password):
            print("DEBUG: Password verification failed")
            raise HTTPException(status_code=401, detail="Invalid username or password")
        
        print(f"DEBUG: Password verified, logging event for user ID: {u.id}")
        # Log login event
        log_event(db, u.id, "login")
        
        print(f"DEBUG: Creating access token")
        access_token = create_access_token({"sub": user.username})
        print(f"DEBUG: Login successful for {user.username}")
        return {"access_token": access_token, "token_type": "bearer", "username": user.username, "is_admin": u.is_admin}
    except Exception as e:
        print(f"Login error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Login failed: {e}")

# Draft persistence
@app.get("/draft/me", response_model=DraftOut)
def get_my_draft(
    assignment: Optional[str] = None, 
    domain: Optional[str] = None,
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    d = db.query(Draft).filter(
        Draft.user_id == current_user.id,
        Draft.assignment == assignment,
        Draft.domain == domain
    ).first()
    return DraftOut(
        assignment=assignment,
        domain=domain,
        plan=d.plan if d else None,
        code=d.code if d else None,
        tests=d.tests if d else None,
        feedback_md=d.feedback_md if d else None,
        updated_at=d.updated_at.isoformat() if d and d.updated_at else None,
    )

@app.put("/draft/me", response_model=DraftOut)
def upsert_my_draft(payload: DraftIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    d = db.query(Draft).filter(
        Draft.user_id == current_user.id,
        Draft.assignment == payload.assignment,
        Draft.domain == payload.domain
    ).first()
    if not d:
        d = Draft(
            user_id=current_user.id,
            assignment=payload.assignment,
            domain=payload.domain
        )
        db.add(d)
    
    # Check if we should create a snapshot
    should_snapshot = should_create_snapshot(
        db, current_user.id, payload.assignment, payload.domain,
        payload.plan or "", payload.code or "", payload.tests or ""
    )
    
    # update fields only if provided
    if payload.plan is not None:  d.plan = payload.plan
    if payload.code is not None:  d.code = payload.code
    if payload.tests is not None: d.tests = payload.tests
    db.commit(); db.refresh(d)
    
    # Create snapshot if content changed
    if should_snapshot:
        create_draft_snapshot(
            db, current_user.id, payload.assignment, payload.domain,
            d.plan, d.code, d.tests, d.feedback_md, "auto-save"
        )
        log_event(db, current_user.id, "save_draft", payload.assignment, payload.domain)
    
    return DraftOut(
        assignment=d.assignment, domain=d.domain,
        plan=d.plan, code=d.code, tests=d.tests,
        feedback_md=d.feedback_md, updated_at=d.updated_at.isoformat() if d.updated_at else None
    )

@app.post("/draft/me", response_model=DraftOut)
def create_my_draft(payload: DraftIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return upsert_my_draft(payload, current_user, db)

# Submissions
@app.post("/submit", response_model=SubmissionResponse)
def submit_assignment(
    submission: SubmissionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_s = Submission(
        username=current_user.username,
        assignment=submission.assignment,
        domain=submission.domain,
        plan=submission.plan,
        code=submission.code,
        tests=submission.tests,
        confidence_level=submission.confidence_level,
        timestamp=get_pst_now()
    )
    db.add(new_s); db.commit(); db.refresh(new_s)
    return SubmissionResponse(
        id=new_s.id, username=new_s.username, assignment=new_s.assignment,
        domain=new_s.domain, plan=new_s.plan, code=new_s.code, tests=new_s.tests,
        confidence_level=new_s.confidence_level, timestamp=to_pst_string(new_s.timestamp), has_feedback=False
    )

@app.post("/submit-part", response_model=PartSubmissionResponse)
def submit_part(
    submission: PartSubmissionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Submit individual parts (A, B, C) of an assignment.
    Always creates a new Submission row to preserve full version history.
    If a previous submission exists, copy its other parts so the latest row
    contains the most recent Plan/Code/Tests together.
    """

    if submission.part not in ['A', 'B', 'C']:
        raise HTTPException(status_code=400, detail="Part must be 'A', 'B', or 'C'")

    # Get latest submission for this user+assignment+domain (if any)
    latest = (db.query(Submission)
                .filter(Submission.username == current_user.username,
                        Submission.assignment == submission.assignment,
                        Submission.domain == submission.domain)
                .order_by(Submission.timestamp.desc())
                .first())

    # Start with previous values if they exist
    new_plan = latest.plan if latest else None
    new_code = latest.code if latest else None
    new_tests = latest.tests if latest else None

    # Apply incoming part change
    if submission.part == 'A':
        if not submission.plan:
            raise HTTPException(status_code=400, detail="Missing 'plan' for part A")
        new_plan = submission.plan
    elif submission.part == 'B':
        if not submission.code:
            raise HTTPException(status_code=400, detail="Missing 'code' for part B")
        new_code = submission.code
    elif submission.part == 'C':
        if not submission.tests:
            raise HTTPException(status_code=400, detail="Missing 'tests' for part C")
        new_tests = submission.tests

    # Create a new versioned submission row
    new_s = Submission(
        username=current_user.username,
        assignment=submission.assignment,
        domain=submission.domain,
        plan=new_plan,
        code=new_code,
        tests=new_tests,
        confidence_level=None,
        timestamp=get_pst_now()
    )
    db.add(new_s)
    db.commit()
    
    return PartSubmissionResponse(
        success=True,
        message=f"Part {submission.part} submitted successfully",
        part=submission.part
    )

# Mark part opened time (first time only)
@app.post("/submission/part-open")
def mark_part_open(
    payload: PartOpenIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    part = payload.part
    if part not in ['A','B','C']:
        raise HTTPException(status_code=400, detail="part must be 'A','B', or 'C'")
    # Find latest submission row for this user+assignment+domain or create one
    latest = (db.query(Submission)
                .filter(Submission.username == current_user.username,
                        Submission.assignment == payload.assignment,
                        Submission.domain == payload.domain)
                .order_by(Submission.timestamp.desc())
                .first())
    if not latest:
        latest = Submission(
            username=current_user.username,
            assignment=payload.assignment,
            domain=payload.domain,
            plan=None, code=None, tests=None,
            confidence_level=None,
            timestamp=get_pst_now()
        )
        db.add(latest); db.commit(); db.refresh(latest)
    now_iso = get_pst_now().isoformat()
    try:
        if part == 'A' and getattr(latest, "a_opened_at", None) in (None, ''):
            latest.a_opened_at = now_iso
        elif part == 'B' and getattr(latest, "b_opened_at", None) in (None, ''):
            latest.b_opened_at = now_iso
        elif part == 'C' and getattr(latest, "c_opened_at", None) in (None, ''):
            latest.c_opened_at = now_iso
        db.commit()
    except Exception:
        db.rollback()
    return {"success": True}


@app.get("/submissions", response_model=List[SubmissionResponse])
def get_all_submissions(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    rows = db.query(Submission).order_by(Submission.timestamp.desc()).all()
    return [
        SubmissionResponse(
            id=s.id, username=s.username, assignment=s.assignment, domain=s.domain,
            plan=s.plan or "", code=s.code or "", tests=s.tests or "", confidence_level=s.confidence_level,
            timestamp=to_pst_string(s.timestamp), has_feedback=False
        )
        for s in rows
    ]

@app.get("/my-submissions", response_model=List[SubmissionResponse])
def get_my_submissions(
    assignment: Optional[str] = None,
    domain: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(Submission).filter(Submission.username == current_user.username)
    if assignment:
        query = query.filter(Submission.assignment == assignment)
    if domain:
        query = query.filter(Submission.domain == domain)
    rows = query.order_by(Submission.timestamp.desc()).all()
    return [
        SubmissionResponse(
            id=s.id, username=s.username, assignment=s.assignment, domain=s.domain,
            plan=s.plan or "", code=s.code or "", tests=s.tests or "", confidence_level=s.confidence_level,
            timestamp=to_pst_string(s.timestamp), has_feedback=False
        )
        for s in rows
    ]

# Admin controls
class FeedbackToggle(BaseModel):
    username: str
    enabled: bool

@app.post("/admin/toggle-feedback")
def toggle_feedback(toggle: FeedbackToggle, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == toggle.username).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    u.feedback_enabled = toggle.enabled
    db.commit()
    return {"success": True, "username": toggle.username, "enabled": toggle.enabled}

@app.post("/admin/toggle-feedback-all")
def toggle_feedback_all(toggle: FeedbackGlobalToggle, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Enable/disable AI feedback for ALL non-admin users"""
    users = db.query(User).filter(User.is_admin == False).all()
    for u in users:
        u.feedback_enabled = toggle.enabled
    db.commit()
    return {"success": True, "enabled": toggle.enabled, "affected": len(users)}

@app.get("/feedback-status")
def check_feedback_status(current_user: User = Depends(get_current_user)):
    return {
        "enabled": current_user.feedback_enabled,
        "feedback_count": current_user.feedback_count,
        "max_feedback": 999999,
        "allowed_assignment": current_user.feedback_assignment
    }

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(submission: SubmissionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.feedback_enabled:
        raise HTTPException(status_code=403, detail="AI feedback not enabled for your account")
    # If admin restricted feedback to a single assignment, enforce it
    if current_user.feedback_assignment and current_user.feedback_assignment != submission.assignment:
        raise HTTPException(status_code=403, detail="AI feedback is disabled for this assignment")

    # Log analyze start
    log_event(db, current_user.id, "analyze_start", submission.assignment, submission.domain)

    # Fetch the question text for this assignment and domain
    question = db.query(Question).filter(
        Question.assignment == submission.assignment,
        Question.domain == submission.domain
    ).first()
    
    question_text = question.question_text if question else "No specific question provided for this assignment and domain."
    
    print(f"DEBUG: About to call OpenAI with domain={submission.domain}")
    try:
        result = call_openai(submission.domain, submission.plan, submission.code, submission.tests, question_text)
        print(f"DEBUG: OpenAI call successful")
    except Exception as e:
        print(f"DEBUG: OpenAI call failed: {e}")
        raise

    # Update latest draft with content + feedback
    d = db.query(Draft).filter(
    Draft.user_id == current_user.id,
    Draft.assignment == submission.assignment,
    Draft.domain == submission.domain
    ).first()
    if not d:
        d = Draft(user_id=current_user.id,
                assignment=submission.assignment,
                domain=submission.domain)
        db.add(d)

    d.plan = submission.plan
    d.code = submission.code
    d.tests = submission.tests

    md = "\n\n".join([
        "**Plan suggestions**",
        *[f"- {s}" for s in result.get("plan_suggestions", [])],
        "",
        "**Code suggestions**",
        *[f"- {s}" for s in result.get("code_suggestions", [])],
        "",
        "**Test suggestions**",
        *[f"- {s}" for s in result.get("test_suggestions", [])],
    ])
    d.feedback_md = md

    create_draft_snapshot(
        db, current_user.id, submission.assignment, submission.domain,
        d.plan, d.code, d.tests, d.feedback_md, "analyze"
    )

    current_user.feedback_count += 1
    log_event(db, current_user.id, "analyze_done", submission.assignment, submission.domain)
    db.add(Feedback(user_id=current_user.id, content_md=d.feedback_md, source="instant"))
    db.commit()

    return AnalyzeResponse(
        plan_suggestions=result.get("plan_suggestions", []),
        code_suggestions=result.get("code_suggestions", []),
        test_suggestions=result.get("test_suggestions", []),
        summary=result.get("summary", None)
    )

@app.post("/evaluate-correctness", response_model=EvaluateResponse)
def evaluate_correctness(payload: EvaluateRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Use GPT to check whether the user's submission is correct/complete enough to proceed."""
    if payload.part not in ['A','B','C']:
        raise HTTPException(status_code=400, detail="part must be 'A', 'B', or 'C'")
    # Fetch assignment question for context
    question = db.query(Question).filter(
        Question.assignment == payload.assignment,
        Question.domain == payload.domain
    ).first()
    question_text = question.question_text if question else ""
    rubric_text = (question.rubric or "") if question else ""

    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Return graceful 200 so frontend can keep the gating experience without a 500
            return EvaluateResponse(is_correct=False, reason="Evaluation service is not configured. Contact admin.")
        client = OpenAI(api_key=api_key)

    # Build evaluation prompt for JSON Boolean verdict
    # Section-specific framing (rubric-first, align strictly to admin question)
    if payload.part == 'A':
        section_name = 'DESIGN_PLAN'
        section_text = (payload.plan or '').strip()
        if rubric_text:
            guidance = (
                "Evaluate ONLY the Design Plan against the following rubric (do not add unrelated criteria). "
                "If the rubric enumerates points, the plan should address these ideas, but it does NOT need to list them verbatim or as headings. "
                "Accept concise narrative that clearly conveys them; allow equivalent phrasing and combined sentences. "
                "For typical DP prompts, look for: (1) chosen approach and why, (2) state definition (what each DP entry means), "
                "(3) how a state depends on smaller ones / what choices are compared, (4) base conditions, and (5) which DP entry or call yields the final result. "
                "If one minor item is implicit yet obvious from context, still mark is_correct=true. "
                "Mark is_correct=false only when core elements are missing or incorrect (e.g., no state definition and no dependency logic), not for formatting. "
                "Additionally, if the answer voluntarily states time/space complexity and it is materially incorrect or contradicts the described design, set is_correct=false even if complexity is not explicitly required by the rubric."
                f"\n{rubric_text}\n"
            )
        else:
            guidance = (
                "Evaluate ONLY against explicit requirements stated in the assignment text; do not require items that are not explicitly requested. "
                "If the prompt lists points, the plan should address these ideas; explicit headings are NOT required. "
                "Accept clear narrative coverage and equivalent phrasing; be lenient on format and wording. "
                "Mark false only when core elements are missing or wrong; do not fail for minor omissions if intent is clear overall. "
                "If the answer voluntarily states time/space complexity and it is materially incorrect or contradicts the design, set is_correct=false."
            )
    elif payload.part == 'B':
        section_name = 'CODE'
        section_text = (payload.code or '').strip()
        if rubric_text:
            guidance = (
                "Evaluate ONLY the Code for correctness relative to the rubric and required behaviors; do not add extra constraints: "
                f"\n{rubric_text}\n"
                "Focus on whether required behaviors are implemented; if behavior is present with minor/pseudocode issues, set is_correct=true and explain briefly."
            )
        else:
            guidance = (
                "Evaluate ONLY the Code against behaviors explicitly required in the assignment text. Ignore unstated extras. "
                "Be LENIENT about minor pseudocode/syntax issues or trivial off-by-one mistakes when the algorithmic intent is correct; in such cases set is_correct=true and explain briefly."
            )
    else:
        section_name = 'TESTS'
        section_text = (payload.tests or '').strip()
        if rubric_text:
            guidance = (
                "Evaluate ONLY the Tests for coverage intent against the rubric; accept natural-language test descriptions. "
                "Do not require executable code, exact names, or specific formatting. "
                f"\n{rubric_text}\n"
                "If the rubric lists concrete cases, accept any clear mention/description of those cases (even if summarized). "
                "If the rubric is high-level, judge reasonable coverage: at least typical + boundary (e.g., empty/smallest, normal, and an error/edge) is sufficient. "
                "A concise bullet list that only names scenarios (e.g., 'ascending', 'descending', 'duplicates', 'random', 'empty', 'single') counts as coverage; "
                "do NOT require expected outputs to be written explicitly. "
                "For stability, any explicit mention such as 'stability', 'stable', 'ties keep original order', or an example like 'A before C for equal keys' is sufficient to satisfy the stability check; "
                "do NOT require an executable assertion. "
                "Mark is_correct=true when the submission signals these core scenarios, even if the wording is brief."
            )
        else:
            guidance = (
                "Evaluate ONLY the Tests versus explicitly requested/implicit coverage in the assignment text; do not require unstated extras. "
                "Accept natural-language test descriptions. Be VERY LENIENT about naming/formatting; do not require code. "
                "A concise list of scenario names (e.g., 'ascending', 'descending', 'duplicates', 'random', 'empty', 'single') is sufficient to count as covered; "
                "do NOT require explicit expected outputs. "
                "For stability, any explicit mention such as 'stability', 'stable', 'ties keep original order', or an example like 'A before C for equal keys' is sufficient; "
                "do NOT require an executable assertion. "
                "Mark true if the answer, in spirit, covers typical, boundary, and at least one edge/error case, even briefly."
            )

    eval_user_msg = (
        f"[ASSIGNMENT QUESTION]\n{question_text}\n\n"
        f"[{section_name}]\n{section_text}\n\n"
        f"{guidance} Return STRICT JSON with keys: is_correct (true/false) and reason (short sentence)."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict evaluator. Output JSON only with keys is_correct (boolean) and reason (string)."},
                {"role": "user", "content": eval_user_msg},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        import json
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
        except Exception:
            # Strict mode: if JSON is malformed, treat as not correct to avoid random flips
            return EvaluateResponse(is_correct=False, reason="Evaluator returned invalid JSON response.")
        is_correct = bool(data.get("is_correct", False))
        reason = str(data.get("reason", "Evaluation complete."))

        # Persist per-part result on the latest submission for this user/assignment/domain
        try:
            latest = (db.query(Submission)
                      .filter(Submission.username == current_user.username,
                              Submission.assignment == payload.assignment,
                              Submission.domain == payload.domain)
                      .order_by(Submission.timestamp.desc())
                      .first())
            if latest:
                if payload.part == 'A':
                    latest.a_correct = is_correct
                    latest.a_reason = reason
                    if is_correct and getattr(latest, "a_passed_at", None) in (None, ''):
                        latest.a_passed_at = get_pst_now().isoformat()
                elif payload.part == 'B':
                    latest.b_correct = is_correct
                    latest.b_reason = reason
                    if is_correct and getattr(latest, "b_passed_at", None) in (None, ''):
                        latest.b_passed_at = get_pst_now().isoformat()
                else:
                    latest.c_correct = is_correct
                    latest.c_reason = reason
                    if is_correct and getattr(latest, "c_passed_at", None) in (None, ''):
                        latest.c_passed_at = get_pst_now().isoformat()
                db.commit()
        except Exception as _:
            # Non-fatal: continue returning the response
            db.rollback()

        return EvaluateResponse(is_correct=is_correct, reason=reason)
    except Exception as e:
        # Graceful failure: do not 500; return not-correct with reason
        return EvaluateResponse(is_correct=False, reason=f"Evaluation error: {e}")

def _sanitize_meta_text(text: str) -> str:
    """Remove any code blocks or obvious code-like content to enforce meta-feedback only."""
    if not text:
        return ""
    # Strip triple backtick code blocks
    out_lines = []
    in_block = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_block = not in_block
            continue
        if in_block:
            continue
        out_lines.append(line)
    text = "\n".join(out_lines)
    # Remove lines that look like code definitions
    blocked_prefixes = ("def ", "class ", "#include", "public ", "private ", "function ", "for (", "while (", "if (", "else {", "try {")
    filtered = []
    for line in text.splitlines():
        if any(line.strip().startswith(p) for p in blocked_prefixes):
            continue
        filtered.append(line)
    text = "\n".join(filtered)
    # Hard-limit length and normalize whitespace
    text = " ".join(text.split())
    return text[:1200]

@app.post("/feedback/clarify", response_model=ClarifyOut)
def clarify_feedback(
    payload: ClarifyIn,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Answer a student's clarification about prior meta‑feedback without giving solutions."""
    # Enforce same gating as feedback generation (per-user, per-assignment)
    if not current_user.feedback_enabled:
        raise HTTPException(status_code=403, detail="AI feedback not enabled for your account")
    if current_user.feedback_assignment and current_user.feedback_assignment != payload.assignment:
        raise HTTPException(status_code=403, detail="AI feedback is disabled for this assignment")
    if payload.part not in ("A","B","C"):
        raise HTTPException(status_code=400, detail="part must be 'A', 'B', or 'C'")

    # Gather context: question/rubric + latest student draft + last feedback shown
    q = db.query(Question).filter(
        Question.assignment == payload.assignment,
        Question.domain == payload.domain
    ).first()
    question_text = q.question_text if q else ""
    rubric_text = (q.rubric or "") if q else ""

    d = db.query(Draft).filter(
        Draft.user_id == current_user.id,
        Draft.assignment == payload.assignment,
        Draft.domain == payload.domain
    ).first()
    plan = d.plan if d and d.plan else ""
    code = d.code if d and d.code else ""
    tests = d.tests if d and d.tests else ""
    prior_feedback = d.feedback_md if d and d.feedback_md else ""

    # Initialize OpenAI client
    global client
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback graceful response
            return ClarifyOut(answer="The clarification service is not configured. Please contact your instructor.")
        client = OpenAI(api_key=api_key)

    # Build strict meta‑feedback clarifier prompt
    system_prompt = (
        "You are a Meta‑Feedback Clarifier. Your job is to clarify previous meta‑feedback so the student can self‑correct.\n"
        "Absolute constraints (never violate):\n"
        "- No final pseudocode or pasteable code.\n"
        "- No step‑by‑step full solution.\n"
        "- No telling which algorithm to use or naming the algorithm as the answer.\n"
        "- Do NOT explain the solution; instead, guide thinking with questions, checks, and process hints.\n"
        "Style: 2–5 sentences. Be specific, supportive, and point to what to check next. Encourage self‑verification."
    )
    user_prompt = (
        f"Domain: {payload.domain}\n\n"
        f"[ASSIGNMENT QUESTION] (context; do not reveal or restate)\n{question_text}\n\n"
        f"[RUBRIC] (for evaluator alignment)\n{rubric_text}\n\n"
        f"[STUDENT_WORK]\n"
        f"Plan:\n{plan}\n\nCode:\n{code}\n\nTests:\n{tests}\n\n"
        f"[META_FEEDBACK_SHOWN]\n{prior_feedback}\n\n"
        f"[STUDENT_CLARIFY_QUESTION]\n{payload.question}\n\n"
        "Respond with meta‑feedback clarification ONLY (no solutions, no algorithms, no code)."
    )
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return ClarifyOut(answer=f"Clarification temporarily unavailable: {e}")

    safe = _sanitize_meta_text(content)
    # Log event (without storing full content)
    try:
        log_event(db, current_user.id, "feedback_clarify", payload.assignment, payload.domain, details=f"part={payload.part}, qlen={len(payload.question)}")
    except Exception:
        pass
    return ClarifyOut(answer=safe if safe else "Here is a meta‑level way to proceed: identify what the feedback asks you to verify, pick one check to run, and describe how your current work satisfies or fails it. Then adjust just that part and re‑evaluate before moving on.")
@app.get("/admin/health-evaluator")
def get_users_healthcheck(admin: User = Depends(get_admin_user)):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"ok": False, "message": "OPENAI_API_KEY is not configured."}
        global client
        if client is None:
            client = OpenAI(api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": "Reply with ok"}, {"role": "user", "content": "ok"}],
                temperature=0.0,
                max_tokens=2,
            )
            content = (resp.choices[0].message.content or "").strip().lower()
            if "ok" in content:
                return {"ok": True, "message": "Evaluator connected."}
            return {"ok": False, "message": f"Unexpected reply: {content[:50]}"}
        except Exception as e:
            return {"ok": False, "message": f"OpenAI call failed: {e}"}
    except Exception as e:
        return {"ok": False, "message": f"Health check error: {e}"}
@app.get("/users")
def get_users(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"username": u.username, "is_admin": u.is_admin, "feedback_enabled": u.feedback_enabled, "feedback_assignment": u.feedback_assignment} for u in users]

class FeedbackAssignmentToggle(BaseModel):
    username: str
    assignment: Optional[str] = None  # None disables feedback

@app.post("/admin/set-feedback-assignment")
def set_feedback_assignment(payload: FeedbackAssignmentToggle, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    u = db.query(User).filter(User.username == payload.username).first()
    if not u:
        raise HTTPException(status_code=404, detail="User not found")
    # If assignment provided, enable feedback and bind to that assignment; otherwise disable entirely
    if payload.assignment and payload.assignment.strip():
        u.feedback_enabled = True
        u.feedback_assignment = payload.assignment.strip()
    else:
        u.feedback_enabled = False
        u.feedback_assignment = None
    db.commit()
    return {"success": True, "username": u.username, "feedback_enabled": u.feedback_enabled, "feedback_assignment": u.feedback_assignment}

# ---------- Pretest Endpoints ----------
from json import dumps as _json_dumps, loads as _json_loads

@app.get("/pretest/questions", response_model=List[PretestQuestionOut])
def get_pretest_questions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(PretestQuestion).order_by(PretestQuestion.id.asc()).all()
    out: List[PretestQuestionOut] = []
    for q in rows:
        try:
            opts = _json_loads(q.options_json)
        except Exception:
            opts = []
        out.append(PretestQuestionOut(id=q.id, question_text=q.question_text, options=opts))
    return out

@app.get("/pretest/status", response_model=PretestStatusOut)
def get_pretest_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    res = (db.query(PretestResult)
             .filter(PretestResult.user_id == current_user.id)
             .order_by(PretestResult.completed_at.desc())
             .first())
    if not res:
        return PretestStatusOut(completed=False)
    return PretestStatusOut(completed=True, num_correct=res.num_correct)

@app.post("/pretest/submit", response_model=PretestStatusOut)
def submit_pretest(payload: PretestSubmitIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Load questions
    questions = {q.id: q for q in db.query(PretestQuestion).all()}
    num_correct = 0
    total = len(questions)
    for a in payload.answers:
        qid = int(a.get("id"))
        choice = int(a.get("choice_index"))
        q = questions.get(qid)
        if q and choice == q.correct_index:
            num_correct += 1
    res = PretestResult(
        user_id=current_user.id,
        answers_json=_json_dumps(payload.answers),
        num_correct=num_correct,
        completed_at=get_pst_now()
    )
    db.add(res); db.commit()
    # Also record a synthetic row in submissions for unified export
    try:
        summary = f"Pretest: {num_correct}/{total} correct"
        sub_row = Submission(
            username=current_user.username,
            assignment="Pretest",
            domain="Assessment",
            plan=summary,
            code=None,
            tests=_json_dumps({"answers": payload.answers, "num_correct": num_correct, "total": total}),
            confidence_level=None,
            timestamp=get_pst_now()
        )
        db.add(sub_row); db.commit()
        try:
            log_event(db, current_user.id, "pretest_submit_recorded", assignment="Pretest", domain="Assessment", details=summary)
        except Exception:
            pass
    except Exception:
        db.rollback()
    return PretestStatusOut(completed=True, num_correct=num_correct)

# Admin CRUD for pretest questions
@app.get("/admin/pretest/questions", response_model=List[PretestQuestionOutAdmin])
def admin_get_pretest(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    rows = db.query(PretestQuestion).order_by(PretestQuestion.id.asc()).all()
    out: List[PretestQuestionOutAdmin] = []
    for q in rows:
        try:
            opts = _json_loads(q.options_json)
        except Exception:
            opts = []
        out.append(PretestQuestionOutAdmin(id=q.id, question_text=q.question_text, options=opts, correct_index=q.correct_index))
    return out

@app.post("/admin/pretest/questions", response_model=PretestQuestionOutAdmin)
def admin_create_pretest(q: PretestQuestionIn, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    if not q.options or q.correct_index < 0 or q.correct_index >= len(q.options):
        raise HTTPException(status_code=400, detail="Invalid options or correct_index")
    row = PretestQuestion(question_text=q.question_text, options_json=_json_dumps(q.options), correct_index=q.correct_index)
    db.add(row); db.commit(); db.refresh(row)
    return PretestQuestionOutAdmin(id=row.id, question_text=row.question_text, options=q.options, correct_index=row.correct_index)

@app.put("/admin/pretest/questions/{qid}", response_model=PretestQuestionOutAdmin)
def admin_update_pretest(qid: int, q: PretestQuestionIn, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    row = db.query(PretestQuestion).filter(PretestQuestion.id == qid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if not q.options or q.correct_index < 0 or q.correct_index >= len(q.options):
        raise HTTPException(status_code=400, detail="Invalid options or correct_index")
    row.question_text = q.question_text
    row.options_json = _json_dumps(q.options)
    row.correct_index = q.correct_index
    db.commit(); db.refresh(row)
    return PretestQuestionOutAdmin(id=row.id, question_text=row.question_text, options=_json_loads(row.options_json), correct_index=row.correct_index)

@app.delete("/admin/pretest/questions/{qid}")
def admin_delete_pretest(qid: int, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    row = db.query(PretestQuestion).filter(PretestQuestion.id == qid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(row); db.commit()
    return {"success": True}

# ---------- Posttest Endpoints ----------
@app.get("/posttest/questions", response_model=List[PosttestQuestionOut])
def get_posttest_questions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(PosttestQuestion).order_by(PosttestQuestion.id.asc()).all()
    out: List[PosttestQuestionOut] = []
    for q in rows:
        try:
            opts = _json_loads(q.options_json)
        except Exception:
            opts = []
        out.append(PosttestQuestionOut(id=q.id, question_text=q.question_text, options=opts))
    return out

@app.get("/posttest/status", response_model=PosttestStatusOut)
def get_posttest_status(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    res = (db.query(PosttestResult)
             .filter(PosttestResult.user_id == current_user.id)
             .order_by(PosttestResult.completed_at.desc())
             .first())
    if not res:
        return PosttestStatusOut(completed=False)
    return PosttestStatusOut(completed=True, num_correct=res.num_correct)

@app.post("/posttest/submit", response_model=PosttestStatusOut)
def submit_posttest(payload: PosttestSubmitIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    # Load questions
    questions = {q.id: q for q in db.query(PosttestQuestion).all()}
    num_correct = 0
    total = len(questions)
    for a in payload.answers:
        qid = int(a.get("id"))
        choice = int(a.get("choice_index"))
        q = questions.get(qid)
        if q and choice == q.correct_index:
            num_correct += 1
    res = PosttestResult(
        user_id=current_user.id,
        answers_json=_json_dumps(payload.answers),
        num_correct=num_correct,
        completed_at=get_pst_now()
    )
    db.add(res); db.commit()
    # Also record a synthetic row in submissions for unified export
    try:
        summary = f"Posttest: {num_correct}/{total} correct"
        sub_row = Submission(
            username=current_user.username,
            assignment="Posttest",
            domain="Assessment",
            plan=summary,
            code=None,
            tests=_json_dumps({"answers": payload.answers, "num_correct": num_correct, "total": total}),
            confidence_level=None,
            timestamp=get_pst_now()
        )
        db.add(sub_row); db.commit()
        try:
            log_event(db, current_user.id, "posttest_submit_recorded", assignment="Posttest", domain="Assessment", details=summary)
        except Exception:
            pass
    except Exception:
        db.rollback()
    return PosttestStatusOut(completed=True, num_correct=num_correct)

# Admin CRUD for posttest questions
@app.get("/admin/posttest/questions", response_model=List[PosttestQuestionOutAdmin])
def admin_get_posttest(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    rows = db.query(PosttestQuestion).order_by(PosttestQuestion.id.asc()).all()
    out: List[PosttestQuestionOutAdmin] = []
    for q in rows:
        try:
            opts = _json_loads(q.options_json)
        except Exception:
            opts = []
        out.append(PosttestQuestionOutAdmin(id=q.id, question_text=q.question_text, options=opts, correct_index=q.correct_index))
    return out

@app.post("/admin/posttest/questions", response_model=PosttestQuestionOutAdmin)
def admin_create_posttest(q: PosttestQuestionIn, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    if not q.options or q.correct_index < 0 or q.correct_index >= len(q.options):
        raise HTTPException(status_code=400, detail="Invalid options or correct_index")
    row = PosttestQuestion(question_text=q.question_text, options_json=_json_dumps(q.options), correct_index=q.correct_index)
    db.add(row); db.commit(); db.refresh(row)
    return PosttestQuestionOutAdmin(id=row.id, question_text=row.question_text, options=q.options, correct_index=row.correct_index)

@app.put("/admin/posttest/questions/{qid}", response_model=PosttestQuestionOutAdmin)
def admin_update_posttest(qid: int, q: PosttestQuestionIn, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    row = db.query(PosttestQuestion).filter(PosttestQuestion.id == qid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    if not q.options or q.correct_index < 0 or q.correct_index >= len(q.options):
        raise HTTPException(status_code=400, detail="Invalid options or correct_index")
    row.question_text = q.question_text
    row.options_json = _json_dumps(q.options)
    row.correct_index = q.correct_index
    db.commit(); db.refresh(row)
    return PosttestQuestionOutAdmin(id=row.id, question_text=row.question_text, options=_json_loads(row.options_json), correct_index=row.correct_index)

@app.delete("/admin/posttest/questions/{qid}")
def admin_delete_posttest(qid: int, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    row = db.query(PosttestQuestion).filter(PosttestQuestion.id == qid).first()
    if not row:
        raise HTTPException(status_code=404, detail="Not found")
    db.delete(row); db.commit()
    return {"success": True}

# Draft History & Event Logging 
@app.get("/draft-history", response_model=List[DraftHistoryResponse])
def get_draft_history(
    assignment: Optional[str] = None,
    domain: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get draft history for a user, optionally filtered by assignment/domain"""
    query = db.query(DraftHistory).filter(DraftHistory.user_id == current_user.id)
    
    if assignment:
        query = query.filter(DraftHistory.assignment == assignment)
    if domain:
        query = query.filter(DraftHistory.domain == domain)
    
    history = query.order_by(DraftHistory.created_at.desc()).all()
    return [
        DraftHistoryResponse(
            id=h.id, assignment=h.assignment, domain=h.domain,
            plan=h.plan, code=h.code, tests=h.tests, feedback_md=h.feedback_md,
            version_tag=h.version_tag, created_at=h.created_at.isoformat()
        )
        for h in history
    ]

@app.get("/event-log", response_model=List[EventLogResponse])
def get_event_log(
    event_type: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get event log for a user, optionally filtered by event type"""
    query = db.query(EventLog).filter(EventLog.user_id == current_user.id)
    
    if event_type:
        query = query.filter(EventLog.event_type == event_type)
    
    events = query.order_by(EventLog.created_at.desc()).limit(100).all()  # Limit to last 100 events
    return [
        EventLogResponse(
            id=e.id, event_type=e.event_type, assignment=e.assignment,
            domain=e.domain, details=e.details, created_at=e.created_at.isoformat()
        )
        for e in events
    ]

@app.post("/log-event")
def log_user_event(
    event: EventLogCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Log a user event (for frontend to call)"""
    log_event(
        db, current_user.id, event.event_type,
        event.assignment, event.domain, event.details
    )
    return {"success": True}

# Question Management Routes (Admin Only)
@app.get("/admin/questions", response_model=List[QuestionResponse])
def get_questions(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Get all questions (admin only)"""
    questions = db.query(Question).order_by(Question.assignment, Question.domain).all()
    return [
        QuestionResponse(
            id=q.id, assignment=q.assignment, domain=q.domain,
            question_text=q.question_text, rubric=q.rubric,
            created_at=q.created_at.isoformat(), updated_at=q.updated_at.isoformat()
        )
        for q in questions
    ]

@app.post("/admin/questions", response_model=QuestionResponse)
def create_question(
    question: QuestionCreate, 
    admin: User = Depends(get_admin_user), 
    db: Session = Depends(get_db)
):
    """Create a new question (admin only)"""
    # Check if question already exists for this assignment+domain combination
    existing = db.query(Question).filter(
        Question.assignment == question.assignment,
        Question.domain == question.domain
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Question already exists for this assignment and domain")
    
    new_question = Question(
        assignment=question.assignment,
        domain=question.domain,
        question_text=question.question_text,
        rubric=question.rubric
    )
    db.add(new_question)
    db.commit()
    db.refresh(new_question)
    
    return QuestionResponse(
        id=new_question.id, assignment=new_question.assignment, domain=new_question.domain,
        question_text=new_question.question_text, rubric=new_question.rubric,
        created_at=new_question.created_at.isoformat(), updated_at=new_question.updated_at.isoformat()
    )

@app.put("/admin/questions/{question_id}", response_model=QuestionResponse)
def update_question(
    question_id: int,
    question_update: QuestionUpdate,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Update a question (admin only)"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    # Check if updating assignment/domain would create a duplicate
    if question_update.assignment or question_update.domain:
        new_assignment = question_update.assignment or question.assignment
        new_domain = question_update.domain or question.domain
        
        existing = db.query(Question).filter(
            Question.assignment == new_assignment,
            Question.domain == new_domain,
            Question.id != question_id
        ).first()
        if existing:
            raise HTTPException(status_code=400, detail="Question already exists for this assignment and domain")
    
    # Update fields
    if question_update.assignment is not None:
        question.assignment = question_update.assignment
    if question_update.domain is not None:
        question.domain = question_update.domain
    if question_update.question_text is not None:
        question.question_text = question_update.question_text
    if question_update.rubric is not None:
        question.rubric = question_update.rubric
    
    db.commit()
    db.refresh(question)
    
    return QuestionResponse(
        id=question.id, assignment=question.assignment, domain=question.domain,
        question_text=question.question_text, rubric=question.rubric,
        created_at=question.created_at.isoformat(), updated_at=question.updated_at.isoformat()
    )

@app.delete("/admin/questions/{question_id}")
def delete_question(
    question_id: int,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Delete a question (admin only)"""
    question = db.query(Question).filter(Question.id == question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")
    
    db.delete(question)
    db.commit()
    return {"success": True, "message": "Question deleted successfully"}

# Admin Delete Operations
@app.delete("/admin/users/{username}")
def delete_user(username: str, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Delete a user (admin only)"""
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot delete admin user")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete related data first
    db.query(Submission).filter(Submission.username == username).delete()
    db.query(Draft).filter(Draft.user_id == user.id).delete()
    db.query(Feedback).filter(Feedback.user_id == user.id).delete()
    db.query(DraftHistory).filter(DraftHistory.user_id == user.id).delete()
    db.query(EventLog).filter(EventLog.user_id == user.id).delete()
    # Pretest/Posttest results
    try:
        db.query(PretestResult).filter(PretestResult.user_id == user.id).delete()
    except Exception:
        pass
    try:
        db.query(PosttestResult).filter(PosttestResult.user_id == user.id).delete()
    except Exception:
        pass
    
    # Delete the user
    db.delete(user)
    db.commit()
    
    return {"success": True, "message": f"User '{username}' and all related data deleted successfully"}

@app.delete("/admin/users")
def delete_all_non_admin_users(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Delete ALL non-admin users and their related data (admin only)"""
    # Collect non-admin users
    users = db.query(User).filter(User.is_admin == False).all()
    user_ids = [u.id for u in users]
    usernames = [u.username for u in users]
    if not users:
        return {"success": True, "deleted": 0}
    # Delete related rows by user ids/usernames
    if usernames:
        db.query(Submission).filter(Submission.username.in_(usernames)).delete(synchronize_session=False)
    if user_ids:
        db.query(Draft).filter(Draft.user_id.in_(user_ids)).delete(synchronize_session=False)
        db.query(Feedback).filter(Feedback.user_id.in_(user_ids)).delete(synchronize_session=False)
        db.query(DraftHistory).filter(DraftHistory.user_id.in_(user_ids)).delete(synchronize_session=False)
        db.query(EventLog).filter(EventLog.user_id.in_(user_ids)).delete(synchronize_session=False)
        try:
            db.query(PretestResult).filter(PretestResult.user_id.in_(user_ids)).delete(synchronize_session=False)
        except Exception:
            pass
        try:
            db.query(PosttestResult).filter(PosttestResult.user_id.in_(user_ids)).delete(synchronize_session=False)
        except Exception:
            pass
        db.query(User).filter(User.id.in_(user_ids)).delete(synchronize_session=False)
    db.commit()
    return {"success": True, "deleted": len(user_ids)}


@app.post("/admin/reset-password")
def reset_user_password(username: str, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Reset a user's password to a temporary one (admin only)"""
    if username == "admin":
        raise HTTPException(status_code=400, detail="Cannot reset admin password")
    
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate a temporary password
    import secrets
    import string
    temp_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(8))
    
    # Hash the temporary password
    hashed_temp = get_password_hash(temp_password)
    user.hashed_password = hashed_temp
    
    db.commit()
    
    return {
        "success": True, 
        "message": f"Password reset for {username}",
        "temporary_password": temp_password,
        "note": "Share this temporary password with the user"
    }


@app.delete("/admin/submissions/{submission_id}")
def delete_submission(submission_id: int, admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Delete a submission (admin only)"""
    submission = db.query(Submission).filter(Submission.id == submission_id).first()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    
    db.delete(submission)
    db.commit()
    
    return {"success": True, "message": f"Submission {submission_id} deleted successfully"}


@app.delete("/admin/submissions/confidence")
def delete_all_confidence_submissions(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Delete all submissions with confidence levels (admin only)"""
    # Delete all submissions that have confidence levels
    deleted_count = db.query(Submission).filter(Submission.confidence_level.isnot(None)).delete()
    db.commit()
    
    return {"success": True, "message": f"Deleted {deleted_count} submissions with confidence levels"}


from fastapi import Body

@app.put("/submission/confidence")
def set_confidence(
    payload: ConfidenceIn = Body(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Clamp confidence to 0..100
    lvl = max(0, min(100, int(payload.confidence_level)))

    # Find the most recent submission for current user + assignment + domain
    sub = (db.query(Submission)
           .filter(Submission.username == current_user.username,
                   Submission.assignment == payload.assignment,
                   Submission.domain == payload.domain)
           .order_by(Submission.timestamp.desc())
           .first())

    if not sub:
        raise HTTPException(status_code=404, detail="No submission found to attach confidence.")

    sub.confidence_level = lvl
    db.commit()

    # Optional: log an event for analytics
    log_event(db, current_user.id, "submit_confidence",
              payload.assignment, payload.domain, details=str(lvl))

    # Silent evaluation: even if frontend skipped or service is unavailable,
    # attempt a backend evaluation and persist to the database
    try:
        # Fetch the latest two submissions (to infer which section changed)
        recent_subs = (db.query(Submission)
                       .filter(Submission.username == current_user.username,
                               Submission.assignment == payload.assignment,
                               Submission.domain == payload.domain)
                       .order_by(Submission.timestamp.desc())
                       .limit(2)
                       .all())
        sub = recent_subs[0] if recent_subs else None
        prev = recent_subs[1] if len(recent_subs) > 1 else None
        if sub is not None:
            # Infer which part to evaluate this time
            def _val(x): return x or ""
            if prev is not None:
                if _val(sub.plan) != _val(prev.plan):
                    part = 'A'
                    section_text = _val(sub.plan).strip()
                elif _val(sub.code) != _val(prev.code):
                    part = 'B'
                    section_text = _val(sub.code).strip()
                elif _val(sub.tests) != _val(prev.tests):
                    part = 'C'
                    section_text = _val(sub.tests).strip()
                else:
                    # No diff → default to C by priority
                    part = 'C'
                    section_text = _val(sub.tests).strip()
            else:
                # Heuristic when there is no previous submission
                if sub.plan and not sub.code and not sub.tests:
                    part = 'A'; section_text = _val(sub.plan).strip()
                elif sub.code and not sub.tests:
                    part = 'B'; section_text = _val(sub.code).strip()
                else:
                    part = 'C'; section_text = _val(sub.tests).strip()

            # Fetch question and rubric
            question = db.query(Question).filter(
                Question.assignment == payload.assignment,
                Question.domain == payload.domain
            ).first()
            question_text = question.question_text if question else ""
            rubric_text = (question.rubric or "") if question else ""

            # Initialize OpenAI client
            global client
            if client is None:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)

            def _write_result_to_submission(val: bool, reason_text: str):
                try:
                    if part == 'A':
                        sub.a_correct = bool(val); sub.a_reason = reason_text
                    elif part == 'B':
                        sub.b_correct = bool(val); sub.b_reason = reason_text
                    else:
                        sub.c_correct = bool(val); sub.c_reason = reason_text
                    db.commit()
                except Exception:
                    db.rollback()

            # Only attempt evaluation if API key is configured; otherwise record as 0
            if client is not None:
                # Assemble a prompt similar to /evaluate-correctness (simplified)
                if part == 'A':
                    section_name = 'DESIGN_PLAN'
                elif part == 'B':
                    section_name = 'CODE'
                else:
                    section_name = 'TESTS'
                # Keep rubric alignment with evaluate_correctness
                if part == 'A':
                    guidance = ("Evaluate ONLY the Design Plan against the assignment text/rubric. "
                                "If the rubric lists items, the plan should address those ideas, but does NOT need verbatim headings. "
                                "Accept clear narrative coverage, equivalent phrasing, and combined sentences; be lenient on format. "
                                "Look for: approach+why, state definition, dependency/choices, base conditions, final result entry. "
                                "If one minor item is implicit yet obvious from context, still mark true; mark false only when core elements are missing or incorrect. "
                                "If the answer voluntarily states time/space complexity and it is materially incorrect or contradicts the design, set is_correct=false even if complexity is not explicitly required. "
                                "Return JSON with is_correct and reason.")
                elif part == 'B':
                    guidance = ("Evaluate ONLY the Code for required behaviors; "
                                "be lenient about minor pseudocode/syntax; return JSON with is_correct and reason.")
                else:
                    guidance = ("Evaluate ONLY the Tests for coverage intent; accept natural-language descriptions and concise bullet lists of scenario names. "
                                "Be VERY LENIENT about naming/formatting and do not require code or explicit expected outputs. "
                                "If scenarios like 'ascending', 'descending', 'duplicates', 'random', 'empty', 'single' are named, treat them as covered. "
                                "For stability, any explicit mention such as 'stability', 'stable', 'ties keep original order', or an example 'A before C for equal keys' is sufficient; do NOT require assertions. "
                                "Mark true if typical + boundary + one edge/error case are indicated, even briefly; return JSON with is_correct and reason.")

                eval_user_msg = (
                    f"[ASSIGNMENT QUESTION]\n{question_text}\n\n"
                    f"[{section_name}]\n{section_text}\n\n"
                    f"{guidance}"
                )
                try:
                    resp = client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": "You are a strict evaluator. Output JSON only with keys is_correct (boolean) and reason (string)."},
                            {"role": "user", "content": eval_user_msg},
                        ],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    import json
                    content = resp.choices[0].message.content
                    data = json.loads(content)
                    is_correct = bool(data.get("is_correct", False))
                    reason = str(data.get("reason", "Evaluation complete."))
                    _write_result_to_submission(is_correct, reason)
                    # add pass timestamps
                    try:
                        if is_correct:
                            now_iso = get_pst_now().isoformat()
                            if part == 'A' and getattr(sub, "a_passed_at", None) in (None, ''):
                                sub.a_passed_at = now_iso
                            elif part == 'B' and getattr(sub, "b_passed_at", None) in (None, ''):
                                sub.b_passed_at = now_iso
                            elif part == 'C' and getattr(sub, "c_passed_at", None) in (None, ''):
                                sub.c_passed_at = now_iso
                            db.commit()
                    except Exception:
                        db.rollback()
                except Exception:
                    # Evaluation failed: record as 0
                    _write_result_to_submission(False, "auto/silent evaluation failed")
            else:
                # No evaluation client available: record as 0
                _write_result_to_submission(False, "auto/silent evaluation skipped")
    except Exception:
        pass

    return {"success": True, "confidence_level": lvl}

# Confidence Level Management
@app.get("/admin/submissions-by-confidence", response_model=List[SubmissionResponse])
def get_submissions_by_confidence(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    """Get the most recent confidence submission per user, sorted by confidence (low→high)."""
    # Subquery to get latest timestamp per user where confidence is present
    latest_per_user = (
        db.query(
            Submission.username.label("username"),
            func.max(Submission.timestamp).label("max_ts")
        )
        .filter(Submission.confidence_level.isnot(None))
        .group_by(Submission.username)
        .subquery()
    )

    # Join to get the full submission rows corresponding to each user's latest
    rows = (
        db.query(Submission)
        .join(
            latest_per_user,
            (Submission.username == latest_per_user.c.username) & (Submission.timestamp == latest_per_user.c.max_ts)
        )
        .order_by(Submission.confidence_level.asc())
        .all()
    )
    return [
        SubmissionResponse(
            id=s.id, username=s.username, assignment=s.assignment, domain=s.domain,
            plan=s.plan or "", code=s.code or "", tests=s.tests or "", confidence_level=s.confidence_level,
            timestamp=to_pst_string(s.timestamp), has_feedback=False
        )
        for s in rows
    ]

@app.post("/questions", response_model=QuestionResponse)
def create_question(
    q: QuestionCreate,
    admin: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    new_q = Question(
        assignment=q.assignment,
        domain=q.domain,
        question_text=q.question_text,
        rubric=q.rubric,
    )
    db.add(new_q)
    db.commit()
    db.refresh(new_q)
    return QuestionResponse.model_validate(new_q)

