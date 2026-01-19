from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Response, Depends, Cookie
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import httpx
import logging
from pathlib import Path
from datetime import datetime, timedelta
import os
import secrets
import sqlite3
from threading import Lock
import re
from urllib.parse import quote

# LDAP imports


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GPU_SERVER_URL = "http://127.0.0.1:8001"  # GPU processing server
LOGS_DIR = Path("./logs")
LOGS_DIR.mkdir(exist_ok=True)
DB_FILE = Path("./auth_sessions.db")
SESSION_TIMEOUT = 7200  # 2 hours in seconds

# Database lock
db_lock = Lock()

app = FastAPI(title="DEAIS MOM Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/mom_icons", StaticFiles(directory="mom_icons"), name="mom_icons")
#app.mount("/public", StaticFiles(directory="public"), name="public")
#app.mount("/static", StaticFiles(directory="static"), name="static")


class GlobalSummaryRequest(BaseModel):
    texts: List[str]
    session_id: Optional[str] = None


class SessionMetadataRequest(BaseModel):
    session_id: str
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    venue: Optional[str] = None
    agenda: Optional[str] = None


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

def initialize_auth_database():
    """Initialize SQLite database for session management"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at DATETIME NOT NULL,
                last_activity DATETIME NOT NULL,
                ip_address TEXT
            )
        """)

        # Local users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS local_users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        conn.commit()

        # Add default users if table is empty
        cursor.execute("SELECT COUNT(*) FROM local_users")
        if cursor.fetchone()[0] == 0:
            default_users = [
                ("admin", "admin123", "Administrator"),
                ("john", "pass123", "John Doe"),
                ("jane", "pass456", "Jane Smith"),
                ("alice", "pass789", "Alice Johnson"),
            ]

            for username, password, display_name in default_users:
                cursor.execute("""
                    INSERT INTO local_users (username, password, display_name, created_at)
                    VALUES (?, ?, ?, ?)
                """, (username, password, display_name, datetime.now().isoformat()))

            conn.commit()
            logger.info(f"✓ Created {len(default_users)} default users")

        conn.close()


def cleanup_expired_sessions():
    """Remove expired sessions from database"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        expiry_time = (datetime.now() - timedelta(seconds=SESSION_TIMEOUT)).isoformat()
        cursor.execute("DELETE FROM sessions WHERE last_activity < ?", (expiry_time,))
        conn.commit()
        conn.close()


def create_session(username: str, display_name: str, ip_address: str) -> str:
    """Create a new session and return session token"""
    session_token = secrets.token_urlsafe(32)
    now = datetime.now().isoformat()

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (session_token, username, display_name, created_at, last_activity, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_token, username, display_name, now, now, ip_address))
        conn.commit()
        conn.close()

    return session_token


def validate_session(session_token: str) -> Optional[dict]:
    """Validate session token and return user data if valid"""
    if not session_token:
        return None

    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT username, display_name, last_activity 
            FROM sessions 
            WHERE session_token = ?
        """, (session_token,))
        result = cursor.fetchone()

        if result:
            username, display_name, last_activity = result
            last_activity_dt = datetime.fromisoformat(last_activity)

            # Check if session expired
            if datetime.now() - last_activity_dt > timedelta(seconds=SESSION_TIMEOUT):
                cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
                conn.commit()
                conn.close()
                return None

            # Update last activity
            cursor.execute("""
                UPDATE sessions 
                SET last_activity = ? 
                WHERE session_token = ?
            """, (datetime.now().isoformat(), session_token))
            conn.commit()
            conn.close()

            return {
                "username": username,
                "display_name": display_name
            }

        conn.close()
        return None


def delete_session(session_token: str):
    """Delete a session (logout)"""
    with db_lock:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sessions WHERE session_token = ?", (session_token,))
        conn.commit()
        conn.close()


# ============================================================================
# LDAP AUTHENTICATION
# ============================================================================

def authenticate_local_user(username: str, password: str, client_ip: str):
    """
    Simple authentication against local database
    """
    try:
        with db_lock:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT password, display_name 
                FROM local_users 
                WHERE username = ?
            """, (username,))

            result = cursor.fetchone()
            conn.close()

            if result:
                stored_password, display_name = result
                if password == stored_password:
                    logger.info(f"✓ Login Success: {username} ({display_name}) from {client_ip}")
                    return True, display_name
                else:
                    logger.warning(f"✗ Login Failed: {username} - Wrong password from {client_ip}")
                    return False, "Invalid password"
            else:
                logger.warning(f"✗ Login Failed: {username} - User not found from {client_ip}")
                return False, "User not found"

    except Exception as e:
        logger.error(f"✗ Auth Error: {username} - {str(e)}")
        return False, f"Authentication error: {str(e)}"


# ============================================================================
# SESSION DEPENDENCY
# ============================================================================

async def get_current_user(session_token: Optional[str] = Cookie(None)):
    """Dependency to get current logged-in user"""
    logger.info(f"session token received:{session_token}")
    if not session_token:
        return None

    user = validate_session(session_token)
    logger.info(f"Validated user: {user}")
    return user


async def require_auth(user: Optional[dict] = Depends(get_current_user)):
    """Dependency to require authentication"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ============================================================================
# LOGGING FUNCTIONS
# ============================================================================

def get_log_filename():
    """Get today's log filename"""
    today = datetime.now().strftime("%Y-%m-%d")
    return LOGS_DIR / f"{today}.txt"


def write_log(log_message: str):
    """Write log message to today's log file"""
    try:
        log_file = get_log_filename()
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")
    except Exception as e:
        logger.error(f"Error writing to log file: {str(e)}")


def get_client_ip(request: Request) -> str:
    """Extract client IP address from request"""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    if request.client:
        return request.client.host

    return "Unknown"


def log_page_access(ip: str, page: str):
    """Log page access"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"IP: {ip} | Timestamp: {timestamp} | Page: {page}"
    write_log(log_message)


def log_dictation(ip: str, duration_seconds: float, result_text: str):
    """Log dictation session"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    minutes = int(duration_seconds // 60)
    seconds = int(duration_seconds % 60)
    duration_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

    log_message = f"IP: {ip} | Timestamp: {timestamp} | Page: Dictation | Duration: {duration_str}"
    write_log(log_message)


def log_discussion_summary_record(ip: str, mode: str, summary_length: str, filename: str, status: str,
                                  result_text: str = ""):
    """Log Discussion Summary with RECORD"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if mode == "Summary":
        mode_text = f"Summary ({summary_length})"
    else:
        mode_text = mode

    log_message = f"IP: {ip} | Timestamp: {timestamp} | Page: Discussion Summary [RECORD] | Mode: {mode_text} | File: {filename} | Status: {status}"
    write_log(log_message)


def log_discussion_summary_upload(ip: str, mode: str, summary_length: str, filename: str, status: str,
                                  result_text: str = ""):
    """Log Discussion Summary with UPLOAD"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if mode == "Summary":
        mode_text = f"Summary ({summary_length})"
    else:
        mode_text = mode

    log_message = f"IP: {ip} | Timestamp: {timestamp} | Page: Discussion Summary [UPLOAD] | Mode: {mode_text} | File: {filename} | Status: {status}"
    write_log(log_message)


# Store job metadata temporarily to link with results
job_metadata = {}


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/logo.png", media_type="image/png")


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    client_ip = get_client_ip(request)
    log_page_access(client_ip, "Login Page")

    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title></title>
    <style>
        body {
            margin: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            color: #fff;
        }
        .login-container {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-radius: 25px;
            padding: 50px 60px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            max-width: 400px;
            width: 90%;
        }
        h1 {
            color: #fff;
            margin-bottom: 10px;
            font-size: 2em;
        }
        p.subtitle {
            color: #e0e0e0;
            font-size: 1.1em;
            margin-bottom: 35px;
        }
        .form-group {
            margin-bottom: 25px;
            text-align: left;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #fff;
        }
        input[type="text"],
        input[type="password"] {
            width: 100%;
            padding: 12px 15px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            background: rgba(255, 255, 255, 0.9);
            box-sizing: border-box;
        }
        input:focus {
            outline: 2px solid #ffeb3b;
        }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            background: linear-gradient(135deg, #ffeb3b 0%, #ffc107 100%);
            color: #333;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 10px;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }
        .error {
            background: rgba(255, 82, 82, 0.9);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        .success {
            background: rgba(76, 175, 80, 0.9);
            color: white;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            display: none;
        }
        footer {
            margin-top: 30px;
            color: #eaeaea;
            font-size: 0.85em;
        }
    </style>
    </head>
    <body>
        <div class="login-container">
            <h1></h1>
            <p class="subtitle">You Can Login With RCNET Username & Password here</p>

            <div id="error-message" class="error"></div>
            <div id="success-message" class="success"></div>

            <form id="login-form">
                <div class="form-group">
                    <label for="username">PIS Number / Username</label>
                    <input type="text" id="username" name="username" required autofocus>
                </div>

                <div class="form-group">
                    <label for="password">Password</label>
                    <input type="password" id="password" name="password" required>
                </div>

                <button type="submit" id="login-btn">Login</button>
            </form>

            <footer>
                <p></p>
                <p></p>
            </footer>
        </div>

        <script>
            const form = document.getElementById('login-form');
            const errorDiv = document.getElementById('error-message');
            const successDiv = document.getElementById('success-message');
            const loginBtn = document.getElementById('login-btn');

            form.addEventListener('submit', async (e) => {
                e.preventDefault();

                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;

                // Hide messages
                errorDiv.style.display = 'none';
                successDiv.style.display = 'none';

                // Disable button
                loginBtn.disabled = true;
                loginBtn.textContent = 'Authenticating...';

                try {
                    const formData = new FormData();
                    formData.append('username', username);
                    formData.append('password', password);

                    const response = await fetch('/authenticate', {
                        method: 'POST',
                        body: formData,
                        credentials: "include"

                    });
                    console.log(response)
                    const data = await response.json();

                    if (response.ok) {
                        successDiv.textContent = 'Login successful! ...';
                        successDiv.style.display = 'block';
                        setTimeout(() => {
                            window.location.href = '/';
                        }, 1000);
                    } else {

                        errorDiv.textContent = data.error || 'Authentication failed';
                        errorDiv.style.display = 'block';
                        loginBtn.disabled = false;
                        loginBtn.textContent = 'Login';
                    }
                } catch (error) {
                    errorDiv.textContent = 'username or password incorrect';
                    errorDiv.style.display = 'block';
                    loginBtn.disabled = false;
                    loginBtn.textContent = 'Login';
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/authenticate")
async def authenticate(
        request: Request,
        username: str = Form(...),
        password: str = Form(...)
):
    """Authenticate user via LDAP and create session"""
    client_ip = get_client_ip(request)

    # Clean expired sessions
    cleanup_expired_sessions()

    # Authenticate via LDAP
    success, display_name = authenticate_local_user(username, password, client_ip)

    if success:
        # Create session
        session_token = create_session(username, display_name, client_ip)

        # Log successful login
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"IP: {client_ip} | Timestamp: {timestamp} | Page: AD Login Success | User: {username} ({display_name})"
        write_log(log_message)

        # Create redirect response and set cookie
        response = JSONResponse(content={
            "success": True,
            "message": "login Successful",
            "redirect": "/"
        })
        response.set_cookie(
            key="session_token",
            value=session_token,
            httponly=True,
            secure=False,  # Set to True in production
            samesite="lax",

            path="/"
        )

        return response
    else:
        # Log failed login
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"IP: {client_ip} | Timestamp: {timestamp} | Page: AD Login Failed | User: {username} | Reason: {display_name}"
        write_log(log_message)

        # Redirect back to login with error message
        from urllib.parse import quote
        error_msg = quote(display_name)
        return RedirectResponse(
            url=f"/login?error={error_msg}",
            status_code=303
        )


@app.get("/logout")
async def logout(
        response: Response,
        request: Request,
        session_token: Optional[str] = Cookie(None)
):
    """Logout user and clear session"""
    client_ip = get_client_ip(request)

    if session_token:
        user = validate_session(session_token)
        if user:
            # Log logout
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"IP: {client_ip} | Timestamp: {timestamp} | Page: Logout | User: {user['username']}"
            write_log(log_message)

        # Delete session
        delete_session(session_token)

    # Clear cookie
    response.delete_cookie(key="session_token")

    return RedirectResponse(url="/login", status_code=303)


# ============================================================================
# PROTECTED HTML ROUTES
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request, user: dict = Depends(require_auth)):
    """Serve the index.html - PROTECTED"""
    try:
        client_ip = get_client_ip(request)
        log_page_access(client_ip, f"Index (Home) - User: {user['username']}")

        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Inject user info and logout button
        user_info_html = f"""
        <div style="position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.9); 
                    padding: 10px 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    z-index: 1000;">
            <span style="color: #333; font-weight: 600;">Welcome, {user['display_name']}</span>
            <a href="/logout" style="margin-left: 15px; color: #667eea; text-decoration: none; 
                                     font-weight: 600;">Logout</a>
        </div>
        """
        html_content = html_content.replace("</body>", f"{user_info_html}</body>")

        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: index.html not found</h1>",
            status_code=404
        )


@app.get("/Dictation.html", response_class=HTMLResponse)
async def serve_audio_assistant(request: Request, user: dict = Depends(require_auth)):
    """Serve the Dictation page - PROTECTED"""
    try:
        client_ip = get_client_ip(request)
        log_page_access(client_ip, f"Dictation - User: {user['username']}")

        with open("Dictation.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Inject user info
        user_info_html = f"""
        <div style="position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.9); 
                    padding: 10px 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    z-index: 1000;">
            <span style="color: #333; font-weight: 600;">{user['display_name']}</span>
            <a href="/logout" style="margin-left: 15px; color: #667eea; text-decoration: none; 
                                     font-weight: 600;">Logout</a>
        </div>
        """
        html_content = html_content.replace("</body>", f"{user_info_html}</body>")

        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Dictation.html not found</h1>",
            status_code=404
        )


@app.get("/U_S_dis.html", response_class=HTMLResponse)
async def serve_meeting_minutes(request: Request, user: dict = Depends(require_auth)):
    """Serve the Discussion Summary page - PROTECTED"""
    try:
        client_ip = get_client_ip(request)
        log_page_access(client_ip, f"Discussion Summary - User: {user['username']}")

        with open("U_S_dis.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Inject user info
        user_info_html = f"""
        <div style="position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.9); 
                    padding: 10px 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    z-index: 1000;">
            <span style="color: #333; font-weight: 600;">{user['display_name']}</span>
            <a href="/logout" style="margin-left: 15px; color: #667eea; text-decoration: none; 
                                     font-weight: 600;">Logout</a>
        </div>
        """
        html_content = html_content.replace("</body>", f"{user_info_html}</body>")

        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Discussion_Summary.html not found</h1>",
            status_code=404
        )


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint - PUBLIC"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{GPU_SERVER_URL}/health")
            gpu_status = response.json()

            return {
                "web_server": "healthy",
                "gpu_server": gpu_status,
                "auth_system": "active"
            }
    except Exception as e:
        return {
            "web_server": "healthy",
            "gpu_server": f"unreachable: {str(e)}",
            "auth_system": "active"
        }


# ============================================================================
# API ROUTES (PROTECTED) - WITH SESSION SUPPORT
# ============================================================================

@app.post("/create_discussion_session")
async def create_discussion_session(
        request: Request,
        user: dict = Depends(require_auth),
        session_name: str = Form(None)
):
    """Create a new discussion session"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/create_session",
                data={
                    "username": user['username'],
                    "session_name": session_name
                }
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error creating session: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")


@app.post("/update_session_metadata")
async def update_session_metadata_endpoint(
        request: Request,
        user: dict = Depends(require_auth),
        metadata: SessionMetadataRequest = None
):
    """Update session metadata"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/update_session_metadata",
                json=metadata.dict()
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error updating metadata: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")


@app.get("/user_discussion_sessions")
async def get_user_discussion_sessions(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Get all discussion sessions for current user"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GPU_SERVER_URL}/user_sessions/{user['username']}"
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error fetching sessions: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")


@app.get("/discussion_session/{session_id}")
async def get_discussion_session(
        request: Request,
        session_id: str,
        user: dict = Depends(require_auth)
):
    """Get full session details"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GPU_SERVER_URL}/session/{session_id}"
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()
    except httpx.RequestError as e:
        logger.error(f"Error fetching session: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")

@app.post("/rename_discussion_session")
async def rename_discussion_session(
    request: Request,
    user: dict = Depends(require_auth)
):
    """Forward rename request to GPU server"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/rename_session",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete_discussion_session")
async def delete_discussion_session(
    request: Request,
    user: dict = Depends(require_auth)
):
    """Forward delete request to GPU server"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/delete_session",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_audio")
async def process_audio(
        request: Request,
        user: dict = Depends(require_auth),
        file: UploadFile = File(...),
        mode: str = Form(...),
        summary_length: str = Form(default="Medium"),
        is_upload: bool = Form(default=False),
        session_id: str = Form(...),
        result_order: int = Form(...)
):
    """Forward audio processing to GPU server - PROTECTED"""
    client_ip = get_client_ip(request)

    try:
        # Read file content
        file_content = await file.read()

        # Prepare multipart form data
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {
            "mode": mode,
            "summary_length": summary_length,
            "session_id": session_id,
            "result_order": result_order
        }

        # Forward to GPU server
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/process_audio",
                files=files,
                data=data
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            result = response.json()
            job_id = result.get("job_id", "unknown")

            # Store metadata for this job
            job_metadata[job_id] = {
                "ip": client_ip,
                "mode": mode,
                "summary_length": summary_length,
                "filename": file.filename,
                "is_upload": is_upload,
                "username": user['username']
            }

            return result

    except httpx.RequestError as e:
        logger.error(f"Error forwarding to GPU server: {str(e)}")

        # Log failure
        if is_upload:
            log_discussion_summary_upload(client_ip, mode, summary_length, file.filename, "FAILED", "")
        else:
            log_discussion_summary_record(client_ip, mode, summary_length, file.filename, "FAILED", "")

        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job_status/{job_id}")
async def get_job_status(request: Request, job_id: str, user: dict = Depends(require_auth)):
    """Check job status and log results - PROTECTED"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(f"{GPU_SERVER_URL}/job_status/{job_id}")

            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Job not found")

            result = response.json()

            # Log completed jobs
            if result.get("status") == "completed" and job_id in job_metadata:
                metadata = job_metadata[job_id]
                result_text = result.get("result", "")

                if metadata["is_upload"]:
                    log_discussion_summary_upload(
                        metadata["ip"],
                        metadata["mode"],
                        metadata["summary_length"],
                        metadata["filename"],
                        "SUCCESS",
                        result_text
                    )
                else:
                    log_discussion_summary_record(
                        metadata["ip"],
                        metadata["mode"],
                        metadata["summary_length"],
                        metadata["filename"],
                        "SUCCESS",
                        result_text
                    )

                # Clean up metadata
                del job_metadata[job_id]

            # Log failed jobs
            elif result.get("status") == "error" and job_id in job_metadata:
                metadata = job_metadata[job_id]

                if metadata["is_upload"]:
                    log_discussion_summary_upload(
                        metadata["ip"],
                        metadata["mode"],
                        metadata["summary_length"],
                        metadata["filename"],
                        "FAILED",
                        ""
                    )
                else:
                    log_discussion_summary_record(
                        metadata["ip"],
                        metadata["mode"],
                        metadata["summary_length"],
                        metadata["filename"],
                        "FAILED",
                        ""
                    )

                # Clean up metadata
                del job_metadata[job_id]

            return result

    except httpx.RequestError as e:
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")


@app.post("/log_dictation")
async def log_dictation_endpoint(
        request: Request,
        user: dict = Depends(require_auth),
        duration: float = Form(...),
        result: str = Form(...)
):
    """Endpoint for frontend to log dictation sessions - PROTECTED"""
    client_ip = get_client_ip(request)
    log_dictation(client_ip, duration, result)
    return {"status": "logged"}


@app.post("/global_summary")
async def generate_global_summary(
        request: Request,
        summary_request: GlobalSummaryRequest,
        user: dict = Depends(require_auth)
):
    """Forward global summary request to GPU server - PROTECTED"""
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/global_summary",
                json={"texts": summary_request.texts, "session_id": summary_request.session_id}
            )

            if response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=response.text)

            return response.json()

    except httpx.RequestError as e:
        logger.error(f"Error generating global summary: {str(e)}")
        raise HTTPException(status_code=503, detail=f"GPU server unreachable: {str(e)}")


@app.post("/create_query_session")
async def create_query_session_proxy(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Forward create query session request"""
    try:
        body = await request.json()
        body['username'] = user['username']  # Add username from auth

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/create_query_session",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update_query_session_selections")
async def update_query_session_selections_proxy(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Forward update selections request"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/update_query_session_selections",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_document")
async def upload_document_proxy(
        request: Request,
        user: dict = Depends(require_auth),
        query_session_id: str = Form(...),
        file: UploadFile = File(...)
):
    """Forward document upload"""
    try:
        file_content = await file.read()
        files = {"file": (file.filename, file_content, file.content_type)}
        data = {"query_session_id": query_session_id}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/upload_document",
                files=files,
                data=data
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query_session/{query_session_id}")
async def get_query_session_proxy(
        request: Request,
        query_session_id: str,
        user: dict = Depends(require_auth)
):
    """Forward get query session request"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GPU_SERVER_URL}/query_session/{query_session_id}"
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_query_sessions")
async def get_user_query_sessions_proxy(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Get all query sessions for current user"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{GPU_SERVER_URL}/user_query_sessions/{user['username']}"
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask_question")
async def ask_question_proxy(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Forward question to GPU server"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/ask_question",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete_query_session")
async def delete_query_session_proxy(
        request: Request,
        user: dict = Depends(require_auth)
):
    """Forward delete query session request"""
    try:
        body = await request.json()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{GPU_SERVER_URL}/delete_query_session",
                json=body
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/Query_Answering.html", response_class=HTMLResponse)
async def serve_query_answering(request: Request, user: dict = Depends(require_auth)):
    """Serve the Query Answering page - PROTECTED"""
    try:
        client_ip = get_client_ip(request)
        log_page_access(client_ip, f"Query Answering - User: {user['username']}")

        with open("Query_Answering.html", "r", encoding="utf-8") as f:
            html_content = f.read()

        # Inject user info
        user_info_html = f"""
        <div style="position: fixed; top: 20px; right: 20px; background: rgba(255,255,255,0.9); 
                    padding: 10px 20px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.2);
                    z-index: 1000;">
            <span style="color: #333; font-weight: 600;">{user['display_name']}</span>
            <a href="/logout" style="margin-left: 15px; color: #667eea; text-decoration: none; 
                                     font-weight: 600;">Logout</a>
        </div>
        """
        html_content = html_content.replace("</body>", f"{user_info_html}</body>")

        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Error: Query_Answering.html not found</h1>",
            status_code=404
        )
# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(401)
async def unauthorized_handler(request: Request, exc: HTTPException):
    """Redirect unauthorized requests to login page"""
    if request.url.path.startswith("/authenticate") or request.url.path.startswith("/login"):
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized"}
        )
    print("user unauthorised access", flush=True)
    return RedirectResponse(url="/login", status_code=303)


# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    port = 8000

    # Initialize authentication database
    initialize_auth_database()
    cleanup_expired_sessions()

    # Log server startup
    startup_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    startup_log = f"========== SERVER STARTUP | Timestamp: {startup_time} | Port: {port} | AUTH ENABLED =========="
    write_log(startup_log)

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=port,

    )