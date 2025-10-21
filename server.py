# server.py
# FastAPI backend with SQLite + persistent drafts & feedback history
# Run: pip install fastapi uvicorn "openai>=1.40.0" pydantic python-jose bcrypt python-multipart sqlalchemy
# Then: uvicorn server:app --reload --port 8000
# Env: export OPENAI_API_KEY=sk-... ; export SECRET_KEY= ;

import os
from datetime import datetime, timedelta
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

client = OpenAI()
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
    created_at = Column(DateTime, default=datetime.utcnow)

class Submission(Base):
    __tablename__ = "submissions"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, nullable=False)
    assignment = Column(String, nullable=False)
    domain = Column(String, nullable=False)
    plan = Column(Text, nullable=True)
    code = Column(Text, nullable=True)
    tests = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

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
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    __table_args__ = ({"sqlite_autoincrement": True},)

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    content_md = Column(Text)
    source = Column(String(32), default="instant")  
    assignment_id = Column(String(128), nullable=True)
    created_at = Column(DateTime, server_default=func.now())

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
    created_at = Column(DateTime, server_default=func.now())

class EventLog(Base):
    __tablename__ = "event_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    event_type = Column(String(64), nullable=False) 
    assignment = Column(String, nullable=True)
    domain = Column(String, nullable=True)
    details = Column(Text, nullable=True) 
    created_at = Column(DateTime, server_default=func.now())

Base.metadata.create_all(bind=engine)

# App 
app = FastAPI(title="AI Meta-Feedback API", version="3.0")
app.mount("/app", StaticFiles(directory="/Users/ruixilin/Desktop/Meta_testversion", html=True), name="app")

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
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
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
    db = SessionLocal()
    try:
        # Check if database needs migration
        try:
            db.query(Draft.assignment).first()
            print("✅ Database schema is up to date")
        except Exception as e:
            print("❌ Database schema needs migration. Please delete meta_feedback.db and restart the server.")
            print(f"Error: {e}")
            return
            
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            admin_user = User(
                username="admin",
                hashed_password=get_password_hash("admin123"),
                is_admin=True,
                feedback_enabled=False,
            )
            db.add(admin_user)
            db.commit()
            print("✅ Admin user created: username='admin', password='admin123'")
    finally:
        db.close()

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

class SubmissionCreate(BaseModel):
    assignment: str
    domain: str
    plan: str
    code: str
    tests: str

class SubmissionResponse(BaseModel):
    id: int
    username: str
    assignment: str
    domain: str
    plan: str
    code: str
    tests: str
    timestamp: str
    has_feedback: bool = False
    class Config:
        from_attributes = True

class FeedbackToggle(BaseModel):
    username: str
    enabled: bool

class AnalyzeResponse(BaseModel):
    score_explanation: str
    findings: List[str]
    gaps: List[str]
    suggestions: List[str]
    code_feedback: List[str]
    test_feedback: List[str]
    summary: str

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

# LLM prompts
SYSTEM_PROMPT = """You are an educational meta-feedback assistant for university-level algorithm tasks.
Given a design plan, Python code, and tests, produce concise feedback:
(1) missing or unclear parts;
(2) 2 short, actionable suggestions;
(3) code feedback on clarity, logic, and efficiency;
(4) test feedback on coverage;
(5) 1-2 sentence score rationale.
IF NO code is present (empty or whitespace), put EXACTLY 3 concise starter HINTS
INSIDE `code_feedback` (not in suggestions). Each hint must be a single bullet,
start with 'HINT:' and stay brief (≤ 15 words).
Use brief bullet points and a supportive tone.
"""

USER_TEMPLATE = """Domain: {domain}

[DESIGN_PLAN.md]
{plan}

[Code]
{code}

[Tests]
{tests}

Return ONLY valid JSON with this structure:
{{
  "score_explanation": "<1-2 sentence heuristic explanation without numbers>",
  "gaps": ["...", "..."],
  "suggestions": ["...", "..."],
  "code_feedback": ["...", "..."],
  "test_feedback": ["...", "..."],
  "summary": "<brief summary>"
}}
"""

def call_openai(domain: str, plan: str, code: str, tests: str) -> Dict[str, Any]:
    user_msg = USER_TEMPLATE.format(
        domain=(domain or "").strip(),
        plan=(plan or "").strip(),
        code=(code or "").strip(),
        tests=(tests or "").strip(),
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

        # fill missing fields safely
        for k in ["score_explanation","findings","gaps","suggestions","code_feedback","test_feedback","summary"]:
            if k not in data:
                data[k] = [] if k in ["findings","gaps","suggestions","code_feedback","test_feedback"] else ""

        # enforce hints when no code provided
        if not (code or "").strip():
            cf = [s.strip() for s in (data.get("code_feedback") or []) if s and s.strip()]
            cf = [s if s.lower().startswith("hint:") else f"HINT: {s}" for s in cf]
            cf = [s[:120] for s in cf]
            fallback = [
                "HINT: Start a function like `def solve(data):`.",
                "HINT: Write 2-3 tiny tests (empty, single, reversed).",
                "HINT: Make it work first, then optimize & check complexity.",
            ]
            i = 0
            while len(cf) < 3 and i < len(fallback):
                cf.append(fallback[i]); i += 1
            data["code_feedback"] = cf[:3]

        return data
    except Exception as e:
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
    u = db.query(User).filter(User.username == user.username).first()
    if not u or not verify_password(user.password, u.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    # Log login event
    log_event(db, u.id, "login")
    
    access_token = create_access_token({"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "username": user.username, "is_admin": u.is_admin}

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
        timestamp=datetime.utcnow()
    )
    db.add(new_s); db.commit(); db.refresh(new_s)
    return SubmissionResponse(
        id=new_s.id, username=new_s.username, assignment=new_s.assignment,
        domain=new_s.domain, plan=new_s.plan, code=new_s.code, tests=new_s.tests,
        timestamp=new_s.timestamp.isoformat(), has_feedback=False
    )

@app.get("/submissions", response_model=List[SubmissionResponse])
def get_all_submissions(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    rows = db.query(Submission).order_by(Submission.timestamp.desc()).all()
    return [
        SubmissionResponse(
            id=s.id, username=s.username, assignment=s.assignment, domain=s.domain,
            plan=s.plan, code=s.code, tests=s.tests, timestamp=s.timestamp.isoformat(), has_feedback=False
        )
        for s in rows
    ]

@app.get("/my-submissions", response_model=List[SubmissionResponse])
def get_my_submissions(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(Submission).filter(Submission.username == current_user.username)\
                               .order_by(Submission.timestamp.desc()).all()
    return [
        SubmissionResponse(
            id=s.id, username=s.username, assignment=s.assignment, domain=s.domain,
            plan=s.plan, code=s.code, tests=s.tests, timestamp=s.timestamp.isoformat(), has_feedback=False
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

@app.get("/feedback-status")
def check_feedback_status(current_user: User = Depends(get_current_user)):
    return {
        "enabled": current_user.feedback_enabled,
        "feedback_count": current_user.feedback_count,
        "max_feedback": 2
    }

# Feedback generation + persistence
class AnalyzeResponse(BaseModel):
    score_explanation: str
    findings: List[str]
    gaps: List[str]
    suggestions: List[str]
    code_feedback: List[str]
    test_feedback: List[str]
    summary: str

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(submission: SubmissionCreate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not current_user.feedback_enabled:
        raise HTTPException(status_code=403, detail="AI feedback not enabled for your account")
    
    if current_user.feedback_count >= 2:
        raise HTTPException(status_code=403, detail="You have reached the maximum number of feedback sessions (2)")

    # Log analyze start
    log_event(db, current_user.id, "analyze_start", submission.assignment, submission.domain)

    result = call_openai(submission.domain, submission.plan, submission.code, submission.tests)

    # Update latest draft with content + feedback
    d = db.query(Draft).filter(
        Draft.user_id == current_user.id,
        Draft.assignment == submission.assignment,
        Draft.domain == submission.domain
    ).first()
    if not d:
        d = Draft(
            user_id=current_user.id,
            assignment=submission.assignment,
            domain=submission.domain
        )
        db.add(d)
    d.plan = submission.plan
    d.code = submission.code
    d.tests = submission.tests
    # Store a compact, readable feedback text blob for restore
    text_md = [
        f"**Score explanation**\n{result.get('score_explanation','')}",
        "**Findings**\n" + "\n".join(f"- {x}" for x in (result.get('findings') or [])),
        "**Gaps**\n" + "\n".join(f"- {x}" for x in (result.get('gaps') or [])),
        "**Suggestions**\n" + "\n".join(f"- {x}" for x in (result.get('suggestions') or [])),
        "**Code feedback**\n" + "\n".join(f"- {x}" for x in (result.get('code_feedback') or [])),
        "**Test feedback**\n" + "\n".join(f"- {x}" for x in (result.get('test_feedback') or [])),
        f"**Summary**\n{result.get('summary','')}",
    ]
    d.feedback_md = "\n\n".join(text_md)

    # Create snapshot with feedback
    create_draft_snapshot(
        db, current_user.id, submission.assignment, submission.domain,
        submission.plan, submission.code, submission.tests, d.feedback_md, "analyze"
    )
    
    # Increment feedback counter
    current_user.feedback_count += 1
    
    # Log analyze completion
    log_event(db, current_user.id, "analyze_done", submission.assignment, submission.domain)

    # Append to history
    db.add(Feedback(user_id=current_user.id, content_md=d.feedback_md, source="instant"))
    db.commit()

    return AnalyzeResponse(**result)

@app.get("/users")
def get_users(admin: User = Depends(get_admin_user), db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"username": u.username, "is_admin": u.is_admin, "feedback_enabled": u.feedback_enabled} for u in users]

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
