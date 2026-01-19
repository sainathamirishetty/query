from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional,Dict
from contextlib import asynccontextmanager
import torch
import uvicorn
import uuid
import shutil
from datetime import datetime
from pathlib import Path
import logging
import os
import sqlite3
from threading import Lock
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import docx

from transformers.models.voxtral.modeling_voxtral import VoxtralForConditionalGeneration
from transformers import AutoProcessor

import sys
import numpy as np
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
RECORDINGS_DIR = Path("./server_recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)
DB_FILE = Path("./sessions.db")

model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
jobs = {}
all_results = []

embedding_model = None

# Database lock
db_lock = Lock()


def initialize_sessions_database():
    """Initialize SQLite database for session management"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                session_name TEXT,
                meeting_date TEXT,
                meeting_time TEXT,
                venue TEXT,
                agenda TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            )
        """)

        # Results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_results (
                result_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                summary_length TEXT,
                result_text TEXT NOT NULL,
                result_order INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            )
        """)

        # Query sessions table (for managing Q&A sessions)
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS query_sessions (
                        query_session_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        session_name TEXT,
                        selected_sessions TEXT,
                        uploaded_documents TEXT,
                        created_at DATETIME NOT NULL,
                        updated_at DATETIME NOT NULL
                    )
                """)

        # Q&A history table
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS qa_history (
                        qa_id TEXT PRIMARY KEY,
                        query_session_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        sources TEXT,
                        created_at DATETIME NOT NULL,
                        FOREIGN KEY (query_session_id) REFERENCES query_sessions(query_session_id)
                    )
                """)

        # Uploaded documents table
        cursor.execute("""
                    CREATE TABLE IF NOT EXISTS uploaded_documents (
                        document_id TEXT PRIMARY KEY,
                        query_session_id TEXT NOT NULL,
                        filename TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at DATETIME NOT NULL,
                        FOREIGN KEY (query_session_id) REFERENCES query_sessions(query_session_id)
                    )
                """)

        conn.commit()
        conn.close()


def create_session(username: str, session_name: str = None) -> str:
    """Create a new session"""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    if not session_name:
        session_name = f"Discussion - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_id, username, session_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, username, session_name, now, now))
        conn.commit()
        conn.close()

    return session_id


def update_session_metadata(session_id: str, meeting_date: str = None, meeting_time: str = None,
                            venue: str = None, agenda: str = None):
    """Update session metadata"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        updates = []
        params = []

        if meeting_date is not None:
            updates.append("meeting_date = ?")
            params.append(meeting_date)
        if meeting_time is not None:
            updates.append("meeting_time = ?")
            params.append(meeting_time)
        if venue is not None:
            updates.append("venue = ?")
            params.append(venue)
        if agenda is not None:
            updates.append("agenda = ?")
            params.append(agenda)

        updates.append("updated_at = ?")
        params.append(datetime.now().isoformat())
        params.append(session_id)

        query = f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?"
        cursor.execute(query, params)
        conn.commit()
        conn.close()


def save_result_to_session(session_id: str, mode: str, summary_length: str, result_text: str, result_order: int):
    """Save a result to a session"""
    result_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO session_results (result_id, session_id, mode, summary_length, result_text, result_order, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (result_id, session_id, mode, summary_length, result_text, result_order, now))

        # Update session timestamp
        cursor.execute("""
            UPDATE sessions SET updated_at = ? WHERE session_id = ?
        """, (now, session_id))

        conn.commit()
        conn.close()

    return result_id


def get_session_results(session_id: str):
    """Get all results for a session"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT result_id, mode, summary_length, result_text, result_order, created_at
            FROM session_results
            WHERE session_id = ?
            ORDER BY result_order ASC
        """, (session_id,))

        results = []
        for row in cursor.fetchall():
            results.append({
                "result_id": row[0],
                "mode": row[1],
                "summary_length": row[2],
                "result_text": row[3],
                "result_order": row[4],
                "created_at": row[5]
            })

        conn.close()
        return results


def get_user_sessions(username: str):
    """Get all sessions for a user"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.session_id, s.session_name, s.meeting_date, s.meeting_time, 
                   s.venue, s.agenda, s.created_at, s.updated_at,
                   COUNT(r.result_id) as result_count
            FROM sessions s
            LEFT JOIN session_results r ON s.session_id = r.session_id
            WHERE s.username = ?
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
        """, (username,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "session_id": row[0],
                "session_name": row[1],
                "meeting_date": row[2],
                "meeting_time": row[3],
                "venue": row[4],
                "agenda": row[5],
                "created_at": row[6],
                "updated_at": row[7],
                "result_count": row[8]
            })

        conn.close()
        return sessions


def get_session_details(session_id: str):
    """Get full session details including metadata and results"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get session metadata
        cursor.execute("""
            SELECT session_id, username, session_name, meeting_date, meeting_time, 
                   venue, agenda, created_at, updated_at
            FROM sessions
            WHERE session_id = ?
        """, (session_id,))

        session_row = cursor.fetchone()
        if not session_row:
            conn.close()
            return None

        session_data = {
            "session_id": session_row[0],
            "username": session_row[1],
            "session_name": session_row[2],
            "meeting_date": session_row[3],
            "meeting_time": session_row[4],
            "venue": session_row[5],
            "agenda": session_row[6],
            "created_at": session_row[7],
            "updated_at": session_row[8]
        }

        conn.close()

    # Get results
    session_data["results"] = get_session_results(session_id)
    return session_data

def rename_session(session_id: str, new_name: str):
    """Rename a session"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE sessions 
            SET session_name = ?, updated_at = ?
            WHERE session_id = ?
        """, (new_name, datetime.now().isoformat(), session_id))
        conn.commit()
        conn.close()


def delete_session(session_id: str):
    """Delete a session and all its results"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Delete results first
        cursor.execute("DELETE FROM session_results WHERE session_id = ?", (session_id,))
        deleted_results = cursor.rowcount

        # Delete session
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        deleted_sessions = cursor.rowcount

        conn.commit()
        conn.close()

        logger.info(f"Deleted {deleted_results} results and {deleted_sessions} session(s)")
        return deleted_sessions > 0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global model, processor,embedding_model
    try:
        logger.info("=" * 70)
        logger.info("Initializing sessions database...")
        initialize_sessions_database()
        logger.info("=" * 70)
        logger.info("Loading AI model...")
        logger.info("=" * 70)
        repo_id = "/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"
        processor = AutoProcessor.from_pretrained(repo_id)
        model = VoxtralForConditionalGeneration.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16, device_map=device
        )
        logger.info(f"✓ Model loaded successfully on {device}")
        logger.info("=" * 70)

        # Load embedding model for RAG
        logger.info("Loading embedding model for RAG...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Embedding model loaded")
        logger.info("=" * 70)
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down GPU server...")


app = FastAPI(title="DEAIS GPU Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GlobalSummaryRequest(BaseModel):
    texts: List[str]
    session_id: Optional[str] = None


class SessionMetadataRequest(BaseModel):
    session_id: str
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    venue: Optional[str] = None
    agenda: Optional[str] = None

class CreateQuerySessionRequest(BaseModel):
    username: str
    session_name: Optional[str] = None

class UpdateQuerySessionRequest(BaseModel):
    query_session_id: str
    selected_sessions: List[str]

class AskQuestionRequest(BaseModel):
    query_session_id: str
    question: str
    top_k: int = 5

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/create_session")
async def create_new_session(username: str = Form(...), session_name: str = Form(None)):
    """Create a new session"""
    try:
        session_id = create_session(username, session_name)
        logger.info(f"✓ Created new session: {session_id} for user: {username}")
        return {
            "status": "success",
            "session_id": session_id,
            "message": "Session created successfully"
        }
    except Exception as e:
        logger.error(f"✗ Error creating session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_session_metadata")
async def update_metadata(request: SessionMetadataRequest):
    """Update session metadata"""
    try:
        update_session_metadata(
            request.session_id,
            request.meeting_date,
            request.meeting_time,
            request.venue,
            request.agenda
        )
        return {"status": "success", "message": "Metadata updated"}
    except Exception as e:
        logger.error(f"✗ Error updating metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_sessions/{username}")
async def get_sessions(username: str):
    """Get all sessions for a user"""
    try:
        sessions = get_user_sessions(username)
        return {
            "status": "success",
            "sessions": sessions
        }
    except Exception as e:
        logger.error(f"✗ Error fetching sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full session details"""
    try:
        session_data = get_session_details(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "status": "success",
            "session": session_data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error fetching session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rename_session")
async def rename_session_endpoint(request: Request):
    """Rename a session"""
    try:
        body = await request.json()
        rename_session(body['session_id'], body['new_name'])
        return {"status": "success", "message": "Session renamed"}
    except Exception as e:
        logger.error(f"✗ Error renaming session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_session")
async def delete_session_endpoint(request: Request):
    """Delete a session"""
    try:
        body = await request.json()
        session_id = body.get('session_id')

        if not session_id:
            logger.error("No session_id provided")
            raise HTTPException(status_code=400, detail="session_id is required")

        logger.info(f"Attempting to delete session: {session_id}")
        result = delete_session(session_id)

        if result:
            logger.info(f"✓ Successfully deleted session: {session_id}")
            return {"status": "success", "message": "Session deleted"}
        else:
            logger.warning(f"Session not found: {session_id}")
            return {"status": "error", "message": "Session not found"}

    except Exception as e:
        logger.error(f"✗ Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def delete_session(session_id: str):
    """Delete a session and all its results"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        # ... existing code ...
        conn.close()
        return deleted_sessions > 0


def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        import io
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"

        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        return ""


def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        import io
        doc_file = io.BytesIO(file_content)
        doc = docx.Document(doc_file)

        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"

        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        return ""
def extract_text_from_txt(file_content: bytes) -> str:
    """Extract text from TXT file"""
    try:
        return file_content.decode('utf-8')
    except Exception as e:
        logger.error(f"Error extracting TXT text: {str(e)}")
        return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if len(chunk.strip()) > 50:  # Skip very short chunks
            chunks.append(chunk.strip())

    return chunks
def get_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for texts"""
    if not embedding_model:
        raise Exception("Embedding model not loaded")
    return embedding_model.encode(texts)


def find_relevant_chunks(query: str, chunks_data: List[Dict], top_k: int = 5) -> List[Dict]:
    """Find most relevant chunks using semantic search"""
    if not chunks_data:
        return []

    # Generate embeddings
    query_embedding = get_embeddings([query])[0]
    chunk_texts = [c['text'] for c in chunks_data]
    chunk_embeddings = get_embeddings(chunk_texts)

    # Calculate cosine similarity
    similarities = np.dot(chunk_embeddings, query_embedding) / (
            np.linalg.norm(chunk_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    # Get top-k most similar
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    relevant_chunks = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Threshold for relevance
            relevant_chunks.append({
                **chunks_data[idx],
                'similarity': float(similarities[idx])
            })

    return relevant_chunks


def create_query_session(username: str, session_name: str = None) -> str:
    """Create a new query session"""
    query_session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    if not session_name:
        session_name = f"Query Session - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_sessions (query_session_id, username, session_name, selected_sessions, uploaded_documents, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (query_session_id, username, session_name, "[]", "[]", now, now))
        conn.commit()
        conn.close()

    return query_session_id
def update_query_session_selections(query_session_id: str, selected_sessions: List[str]):
    """Update selected sessions for a query session"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE query_sessions 
            SET selected_sessions = ?, updated_at = ?
            WHERE query_session_id = ?
        """, (json.dumps(selected_sessions), datetime.now().isoformat(), query_session_id))
        conn.commit()
        conn.close()


def save_uploaded_document(query_session_id: str, filename: str, content: str) -> str:
    """Save uploaded document content"""
    document_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO uploaded_documents (document_id, query_session_id, filename, content, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (document_id, query_session_id, filename, content, now))

        # Update query session's uploaded_documents list
        cursor.execute("""
            SELECT uploaded_documents FROM query_sessions WHERE query_session_id = ?
        """, (query_session_id,))
        result = cursor.fetchone()
        if result:
            docs = json.loads(result[0])
            docs.append({"document_id": document_id, "filename": filename})
            cursor.execute("""
                UPDATE query_sessions 
                SET uploaded_documents = ?, updated_at = ?
                WHERE query_session_id = ?
            """, (json.dumps(docs), datetime.now().isoformat(), query_session_id))

        conn.commit()
        conn.close()

    return document_id


def get_query_session_data(query_session_id: str) -> Dict:
    """Get all data for a query session"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get query session info
        cursor.execute("""
            SELECT query_session_id, username, session_name, selected_sessions, uploaded_documents, created_at
            FROM query_sessions
            WHERE query_session_id = ?
        """, (query_session_id,))

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        query_session = {
            "query_session_id": row[0],
            "username": row[1],
            "session_name": row[2],
            "selected_sessions": json.loads(row[3]),
            "uploaded_documents": json.loads(row[4]),
            "created_at": row[5]
        }

        # Get Q&A history
        cursor.execute("""
            SELECT qa_id, question, answer, sources, created_at
            FROM qa_history
            WHERE query_session_id = ?
            ORDER BY created_at ASC
        """, (query_session_id,))

        qa_history = []
        for qa_row in cursor.fetchall():
            qa_history.append({
                "qa_id": qa_row[0],
                "question": qa_row[1],
                "answer": qa_row[2],
                "sources": json.loads(qa_row[3]) if qa_row[3] else [],
                "created_at": qa_row[4]
            })

        query_session["qa_history"] = qa_history

        conn.close()
        return query_session


def get_user_query_sessions(username: str) -> List[Dict]:
    """Get all query sessions for a user"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT q.query_session_id, q.session_name, q.selected_sessions, q.uploaded_documents, q.created_at, q.updated_at,
                   COUNT(qa.qa_id) as qa_count
            FROM query_sessions q
            LEFT JOIN qa_history qa ON q.query_session_id = qa.query_session_id
            WHERE q.username = ?
            GROUP BY q.query_session_id
            ORDER BY q.updated_at DESC
        """, (username,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                "query_session_id": row[0],
                "session_name": row[1],
                "selected_sessions": json.loads(row[2]),
                "uploaded_documents": json.loads(row[3]),
                "created_at": row[4],
                "updated_at": row[5],
                "qa_count": row[6]
            })

        conn.close()
        return sessions


def save_qa_to_history(query_session_id: str, question: str, answer: str, sources: List[Dict]) -> str:
    """Save Q&A to history"""
    qa_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO qa_history (qa_id, query_session_id, question, answer, sources, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (qa_id, query_session_id, question, answer, json.dumps(sources), now))

        # Update query session timestamp
        cursor.execute("""
            UPDATE query_sessions SET updated_at = ? WHERE query_session_id = ?
        """, (now, query_session_id))

        conn.commit()
        conn.close()

    return qa_id


def delete_query_session(query_session_id: str):
    """Delete a query session and all its data"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Delete Q&A history
        cursor.execute("DELETE FROM qa_history WHERE query_session_id = ?", (query_session_id,))

        # Delete uploaded documents
        cursor.execute("DELETE FROM uploaded_documents WHERE query_session_id = ?", (query_session_id,))

        # Delete query session
        cursor.execute("DELETE FROM query_sessions WHERE query_session_id = ?", (query_session_id,))

        conn.commit()
        conn.close()

@app.post("/process_audio")
async def process_audio(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        mode: str = Form(...),
        summary_length: str = Form(default="Medium"),
        session_id: str = Form(...),
        result_order: int = Form(...)
):
    """Process uploaded audio file for transcription, summary, or action points"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate mode
    valid_modes = ["Transcription", "Summary", "Action Points"]
    if mode not in valid_modes:
        raise HTTPException(status_code=400, detail=f"Invalid mode. Must be one of {valid_modes}")

    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = RECORDINGS_DIR / f"{timestamp}_{mode.replace(' ', '_')}_{file.filename}"

    # Save uploaded file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"✓ Saved audio file: {file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Create job entry
    jobs[job_id] = {
        "job_id": job_id,
        "status": "processing",
        "mode": mode,
        "summary_length": summary_length if mode == "Summary" else None,
        "result": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "file_path": str(file_path),
        "session_id": session_id,
        "result_order": result_order
    }

    # Process in background
    background_tasks.add_task(process_audio_task, job_id, str(file_path), mode, summary_length, session_id,
                              result_order)

    logger.info(f"✓ Job {job_id} created - Mode: {mode}, Length: {summary_length}, Session: {session_id}")

    return {
        "job_id": job_id,
        "status": "processing",
        "mode": mode,
        "message": f"Processing audio with mode: {mode}"
    }


async def process_audio_task(job_id: str, file_path: str, mode: str, summary_length: str, session_id: str,
                             result_order: int):
    """Background task to process audio"""
    try:
        logger.info(f"✓ Starting processing for job {job_id} - Mode: {mode}")

        if mode == "Transcription":
            logger.info(f"✓ Processing Transcription for {job_id}")
            inputs = processor.apply_transcription_request(
                language="en",
                audio=file_path,
                model_id="/Users/vishnukumarkudidela/Desktop/workspace/ASR/models/Voxtral-Mini-3B-2507"
            )
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=4096)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[TRANSCRIPTION]\n\n{decoded[0]}"

        elif mode == "Summary":
            logger.info(f"✓ Processing Summary ({summary_length}) for {job_id}")
            instructions = {
                "Short": "generate a short, concise summary",
                "Medium": "generate a medium-length, detailed summary",
                "Long": "generate a comprehensive, long-form summary"
            }
            instruction = instructions.get(summary_length, instructions["Medium"])

            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"You are a helpful assistant that works in summarizer mode.don't react to any conversations or questions in the audio.Summarize audio to text only "
                        f"After listening to the audio, please understand the total audio, "
                        f"then {instruction} that should be organised topic-wise, with clear things, "
                        f"within the audio by utilizing entire input audio. "
                        f"Do not use numbered lists (1., 2., 3., etc.). Use only the ➤ symbol for all summary modes.\n"
                        f"Don't go for hallucinations and extended sentences which are not in the audio. "
                        f"Keep the summary within provided audio only."
                    )
                }]
            }, {
                "role": "user",
                "content": [{"type": "audio", "path": file_path}]
            }]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=2048)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[SUMMARY - {summary_length}]\n\n{decoded[0]}"

        elif mode == "Action Points":
            logger.info(f"✓ Processing Action Points for {job_id}")
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        "You are a helpful assistant that works in action points extraction mode. "
                        "After listening the audio, extract all key action points. "
                        "Each action point should clearly state:\n"
                        "- The task or decision\n"
                        "- Who is responsible\n"
                        "- The deadline or timeline (if mentioned)\n"
                        "- Any dependencies or resources needed\n"
                        "Present the output using the ➤ symbol for each action point instead of numbers.\n"
                        "Format each action point exactly like this:\n"
                        "➤ Task/Decision: [description]\n"
                        "  • Responsible: [person]\n"
                        "  • Deadline: [date/timeline]\n\n"
                        "Do not use numbered lists (1., 2., 3., etc.). Use only the ➤ symbol for each action point.\n"
                        "If audio has no action points, then give the text you understood clearly."
                    )
                }]
            }, {
                "role": "user",
                "content": [{"type": "audio", "path": file_path}]
            }]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=2048)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            result = f"[ACTION POINTS]\n\n{decoded[0]}"

        # Update job status
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

        # Save to session
        save_result_to_session(session_id, mode, summary_length, result, result_order)

        logger.info(f"✓ Job {job_id} completed and saved to session {session_id}")

    except Exception as e:
        logger.error(f"✗ Job {job_id} failed: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["completed_at"] = datetime.now().isoformat()

    finally:
        # AUTOMATICALLY DELETE THE AUDIO FILE AFTER PROCESSING
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"✓ Auto-deleted temporary audio file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {str(e)}")


@app.get("/job_status/{job_id}")
async def get_job_status(job_id: str):
    """Check the status of a processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_data = jobs[job_id].copy()
    # Don't send file_path to client
    if "file_path" in job_data:
        del job_data["file_path"]

    return job_data


@app.post("/global_summary")
async def generate_global_summary(request: GlobalSummaryRequest):
    """Generate a global summary from multiple text results"""
    if not model or not processor:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    try:
        # Filter out transcriptions, keep only summaries and action points
        filtered = [t for t in request.texts if not t.strip().startswith("[TRANSCRIPTION]")]

        if not filtered:
            raise HTTPException(status_code=400, detail="No valid summaries or action points to process")

        combined = "\n\n".join(filtered)

        conversation = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": (
                    "You are an expert meeting summarizer. Your task is to create a comprehensive global summary "
                    "from multiple meeting segments.\n\n"

                    "INSTRUCTIONS:\n"
                    "1. Read and analyze all the provided text segments carefully\n"
                    "2. Identify and group related topics across all segments\n"
                    "3. Create a cohesive, well-organized summary that covers ALL topics discussed\n"
                    "4. Maintain chronological flow when relevant\n"
                    "5. Preserve important details, decisions, and discussions\n"
                    "6. Remove redundancies while keeping unique information from each segment\n"
                    "7. don't repeat the sentences keep as unique\n\n"

                    "OUTPUT STRUCTURE:\n"
                    "- Start with an 'OVERVIEW' section (2-3 sentences about the overall meeting)\n"
                    "- Organize content by TOPICS with clear headings\n"
                    "- Under each topic, provide key points and discussions\n"
                    "- Include a 'KEY DECISIONS' section if any decisions were made\n"
                    "- End with 'ACTION ITEMS' section if action points exist\n\n"

                    "IMPORTANT GUIDELINES:\n"
                    "- Do NOT add information not present in the original text\n"
                    "- Do NOT miss any important topics, even if briefly mentioned\n"
                    "- Use clear, professional language\n"
                    "- Be concise but comprehensive\n"
                    "- don't repeat the sentences keep as unique\n"
                    "- don't add action points twice in global summary\n"
                    "- If a topic appears in multiple segments, consolidate it intelligently\n\n"

                    f"TEXT TO SUMMARIZE:\n\n{combined}"
                )
            }]
        }]

        inputs = processor.apply_chat_template(conversation)
        inputs = inputs.to(device, dtype=torch.bfloat16)
        outputs = model.generate(**inputs, max_new_tokens=4096)
        decoded = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        global_summary = decoded[0]

        # Save to session if provided
        if request.session_id:
            # Get current result count for ordering
            results = get_session_results(request.session_id)
            result_order = len(results) + 1
            save_result_to_session(request.session_id, "GLOBAL SUMMARY", None, global_summary, result_order)
            logger.info(f"✓ Global summary saved to session {request.session_id}")

        logger.info("✓ Global summary generated successfully")

        return {
            "status": "success",
            "global_summary": global_summary
        }

    except Exception as e:
        logger.error(f"✗ Global summary generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_query_session")
async def create_query_session_endpoint(request: CreateQuerySessionRequest):
    """Create a new query session"""
    try:
        query_session_id = create_query_session(request.username, request.session_name)
        logger.info(f"✓ Created query session: {query_session_id}")
        return {
            "status": "success",
            "query_session_id": query_session_id
        }
    except Exception as e:
        logger.error(f"✗ Error creating query session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_query_session_selections")
async def update_query_session_selections_endpoint(request: UpdateQuerySessionRequest):
    """Update selected sessions for query session"""
    try:
        update_query_session_selections(request.query_session_id, request.selected_sessions)
        return {"status": "success"}
    except Exception as e:
        logger.error(f"✗ Error updating selections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document")
async def upload_document_endpoint(
        query_session_id: str = Form(...),
        file: UploadFile = File(...)
):
    """Upload and process document for query session"""
    try:
        # Read file content
        content = await file.read()

        # Extract text based on file type
        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            text_content = extract_text_from_pdf(content)
        elif filename.endswith('.docx'):
            text_content = extract_text_from_docx(content)
        elif filename.endswith('.txt'):
            text_content = extract_text_from_txt(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, DOCX, or TXT")

        if not text_content:
            raise HTTPException(status_code=400, detail="Could not extract text from document")

        # Save to database
        document_id = save_uploaded_document(query_session_id, file.filename, text_content)

        logger.info(f"✓ Uploaded document: {file.filename} for query session {query_session_id}")

        return {
            "status": "success",
            "document_id": document_id,
            "filename": file.filename,
            "content_length": len(text_content)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query_session/{query_session_id}")
async def get_query_session_endpoint(query_session_id: str):
    """Get query session data with history"""
    try:
        data = get_query_session_data(query_session_id)
        if not data:
            raise HTTPException(status_code=404, detail="Query session not found")

        return {
            "status": "success",
            "query_session": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error fetching query session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_query_sessions/{username}")
async def get_user_query_sessions_endpoint(username: str):
    """Get all query sessions for a user"""
    try:
        sessions = get_user_query_sessions(username)
        return {
            "status": "success",
            "query_sessions": sessions
        }
    except Exception as e:
        logger.error(f"✗ Error fetching query sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question")
async def ask_question_endpoint(request: AskQuestionRequest):
    """Answer question using RAG from selected sessions and documents"""
    if not model or not processor or not embedding_model:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Get query session data
        query_session = get_query_session_data(request.query_session_id)
        if not query_session:
            raise HTTPException(status_code=404, detail="Query session not found")

        # Collect all chunks from selected sessions
        chunks_data = []

        # Add chunks from selected discussion sessions
        for session_id in query_session['selected_sessions']:
            session_results = get_session_results(session_id)
            session_details = get_session_details(session_id)
            session_name = session_details['session_name'] if session_details else "Unknown Session"

            for result in session_results:
                text = result['result_text']
                # Split into paragraphs
                paragraphs = text.split('\n\n')
                for para in paragraphs:
                    if len(para.strip()) > 50:
                        chunks_data.append({
                            'text': para.strip(),
                            'source_type': 'session',
                            'source_name': session_name,
                            'session_id': session_id,
                            'mode': result['mode']
                        })

        # Add chunks from uploaded documents
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT document_id, filename, content
                FROM uploaded_documents
                WHERE query_session_id = ?
            """, (request.query_session_id,))

            for row in cursor.fetchall():
                document_id, filename, content = row
                document_chunks = chunk_text(content)
                for chunk in document_chunks:
                    chunks_data.append({
                        'text': chunk,
                        'source_type': 'document',
                        'source_name': filename,
                        'document_id': document_id
                    })

            conn.close()

        if not chunks_data:
            return {
                "status": "success",
                "answer": "No data available. Please select discussion sessions or upload documents first.",
                "sources": [],
                "qa_id": None
            }

        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(request.question, chunks_data, top_k=request.top_k)

        if not relevant_chunks:
            answer = "I couldn't find relevant information to answer your question based on the selected sessions and documents."
            sources = []
        else:
            # Build context
            context_parts = []
            for i, chunk in enumerate(relevant_chunks, 1):
                source_label = f"[Source {i}: {chunk['source_name']}]"
                context_parts.append(f"{source_label}\n{chunk['text']}")

            context = "\n\n".join(context_parts)

            # Generate answer using Voxtral
            conversation = [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": (
                        f"You are a helpful assistant answering questions based on meeting sessions and documents.\n\n"
                        f"CONTEXT:\n{context}\n\n"
                        f"QUESTION: {request.question}\n\n"
                        f"INSTRUCTIONS:\n"
                        f"- Answer the question based ONLY on the provided context\n"
                        f"- If the context doesn't contain enough information, say so clearly\n"
                        f"- Be specific and cite which source(s) support your answer\n"
                        f"- Use bullet points for multiple points\n"
                        f"- Be concise but complete\n"
                        f"- Don't make up information not in the context"
                    )
                }]
            }]

            inputs = processor.apply_chat_template(conversation)
            inputs = inputs.to(device, dtype=torch.bfloat16)
            outputs = model.generate(**inputs, max_new_tokens=1024)
            decoded = processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            answer = decoded[0]

            # Prepare sources
            sources = [{
                'source_type': chunk['source_type'],
                'source_name': chunk['source_name'],
                'text_snippet': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'],
                'similarity': chunk['similarity']
            } for chunk in relevant_chunks]

        # Save to history
        qa_id = save_qa_to_history(request.query_session_id, request.question, answer, sources)

        logger.info(f"✓ Answered question for query session {request.query_session_id}")

        return {
            "status": "success",
            "answer": answer,
            "sources": sources,
            "qa_id": qa_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error answering question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_query_session")
async def delete_query_session_endpoint(request: Request):
    """Delete a query session"""
    try:
        body = await request.json()
        query_session_id = body.get('query_session_id')

        if not query_session_id:
            raise HTTPException(status_code=400, detail="query_session_id is required")

        delete_query_session(query_session_id)

        return {"status": "success", "message": "Query session deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"✗ Error deleting query session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = 8001
    server_ip = "127.0.0.1"

    print(f"\n{'=' * 70}")
    print(f"GPU Processing Server")
    print(f"{'=' * 70}")
    print(f"✓ Server Address:      http://{server_ip}:{port}")
    print(f"✓ Health Check:        http://{server_ip}:{port}/health")
    print(f"✓ Device:              {device}")
    print(f"✓ Recordings Dir:      {RECORDINGS_DIR.absolute()}")
    print(f"{'=' * 70}")
    print(f"✓ Loading AI model... (this may take 1-2 minutes)")
    print(f"{'=' * 70}\n")

    uvicorn.run(app, host=server_ip, port=port)