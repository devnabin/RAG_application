
from contextlib import asynccontextmanager
from html import escape
from pathlib import Path
import asyncio
import hashlib
import re
import secrets
import shutil
import time

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel

from RAG_backend import chat, delete_from_vector_db, search_vector_db, vector_db

# folder where the uploaded PDFs are saved
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# biggest PDF allowed: 20 MB
MAX_FILE_SIZE = 20 * 1024 * 1024

# Sessions: every browser gets a random id in a cookie and its own folder: uploads/<session_id>/
# The pages send a heartbeat every 30 seconds. If a session sends nothing for
# SESSION_TIMEOUT seconds (tab or browser closed), its files and chunks are deleted.
# name of the cookie that stores the session id
SESSION_COOKIE = "rag_session"
# delete a session after 5 minutes (in seconds) without a heartbeat
SESSION_TIMEOUT = 5 * 60
# check for expired sessions every 60 seconds
CLEANUP_EVERY = 60
# session_id -> {"last_seen": time, "busy": uploads in progress, "history": {file_id: [chat messages]}}
sessions: dict[str, dict] = {}


# folder with upload.html and chat.html
TEMPLATE_DIR = Path(__file__).parent / "templates"


# read an HTML page from the templates folder
def load_template(name: str) -> str:
    # Read the HTML file every time, so changes to the HTML show up without restarting
    return (TEMPLATE_DIR / name).read_text()


# body of the /ask request sent by chat.html
class Question(BaseModel):
    file_id: str
    question: str


# folder for one session's PDFs: uploads/<session_id>
def session_dir(session_id: str) -> Path:
    return UPLOAD_DIR / session_id


# raised when a PDF does not exist; pdf_not_found() below sends the user to the homepage
class PDFNotFound(Exception):
    pass


# return the PDF's path, but only if it belongs to this session
def get_pdf_path(session_id: str, file_id: str) -> Path:
    # file_id must be a sha256 hash we generated, which also blocks path traversal.
    # Looking only inside this session's folder means users can't open each other's PDFs.
    if not re.fullmatch(r"[0-9a-f]{64}", file_id):
        raise PDFNotFound()
    path = session_dir(session_id) / f"{file_id}.pdf"
    if not path.exists():
        raise PDFNotFound()
    return path


# the original file name the user uploaded (saved in a .name file next to the PDF)
def get_filename(pdf_path: Path) -> str:
    name_file = pdf_path.with_suffix(".name")
    return name_file.read_text() if name_file.exists() else "your PDF"


# all PDFs of a session as (file_id, file name), oldest first
def list_files(session_id: str) -> list[tuple[str, str]]:
    folder = session_dir(session_id)
    if not folder.exists():
        return []
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    return [(p.stem, get_filename(p)) for p in pdfs]


# delete one PDF, its .name file and its chunks in the vector db
def delete_file(pdf_path: Path):
    try:
        delete_from_vector_db(pdf_path.stem)
    except Exception as e:
        print(f"Could not delete chunks of {pdf_path.stem[:8]}: {e}")
    pdf_path.unlink(missing_ok=True)
    pdf_path.with_suffix(".name").unlink(missing_ok=True)


def delete_folder(folder: Path):
    # Delete every PDF (and its chunks) in a session folder, then the folder itself
    for pdf_path in folder.glob("*.pdf"):
        delete_file(pdf_path)
    shutil.rmtree(folder, ignore_errors=True)
    print(f"Deleted session folder {folder.name[:8]}")


# background task: every minute, delete sessions that sent no heartbeat for 5 minutes
async def cleanup_expired_sessions():
    while True:
        await asyncio.sleep(CLEANUP_EVERY)
        now = time.time()
        for session_id, info in list(sessions.items()):
            if info["busy"] == 0 and now - info["last_seen"] > SESSION_TIMEOUT:
                sessions.pop(session_id, None)
                await run_in_threadpool(delete_folder, session_dir(session_id))


# runs when the server starts (code before yield) and stops (code after yield)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sessions live in memory, so after a restart every old session is lost: delete their files
    for item in UPLOAD_DIR.iterdir():
        if item.is_dir():
            await run_in_threadpool(delete_folder, item)
        elif item.suffix == ".pdf":
            await run_in_threadpool(delete_file, item)
        else:
            item.unlink()
    task = asyncio.create_task(cleanup_expired_sessions())
    yield
    task.cancel()


# the FastAPI app
app = FastAPI(title="PDF Q&A", lifespan=lifespan)


# PDFNotFound raised anywhere -> redirect to the homepage
@app.exception_handler(PDFNotFound)
async def pdf_not_found(request: Request, exc: PDFNotFound):
    # PDF deleted, session expired, or someone else's link: send the user back to the homepage
    return RedirectResponse(url="/", status_code=303)


# runs before every request: find the user's session from the cookie, or create a new one
@app.middleware("http")
async def session_middleware(request: Request, call_next):
    session_id = request.cookies.get(SESSION_COOKIE)
    # no cookie, or the session already expired
    is_new = session_id not in sessions
    if is_new:
        session_id = secrets.token_hex(16)
        sessions[session_id] = {"last_seen": time.time(), "busy": 0, "history": {}}
    sessions[session_id]["last_seen"] = time.time()
    request.state.session_id = session_id

    response = await call_next(request)
    if is_new:
        # no max_age: the browser forgets the cookie when it is closed
        response.set_cookie(SESSION_COOKIE, session_id, httponly=True, samesite="lax")
    return response


# build the upload page with an error message and the user's list of PDFs
def render_upload(session_id: str, error: str = "") -> str:
    error_html = f'<div class="error">{error}</div>' if error else ""
    files = list_files(session_id)
    if files:
        items = "".join(
            f'<li><a href="/chat/{file_id}">📄 {escape(name)}</a>'
            f'<form action="/delete/{file_id}" method="post">'
            f'<button class="delete" type="submit">Delete</button></form></li>'
            for file_id, name in files
        )
        files_html = f'<h2>Your PDFs</h2><ul class="files">{items}</ul>'
    else:
        files_html = ""
    return load_template("upload.html").replace("{error}", error_html).replace("{files}", files_html)


# GET / : homepage with the upload form and the user's PDFs
@app.get("/", response_class=HTMLResponse)
def upload_page(request: Request):
    return render_upload(request.state.session_id)


# POST /upload : check and save the PDF, chunk + embed it, then open the chat page
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    session_id = request.state.session_id
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return HTMLResponse(render_upload(session_id, "Please upload a .pdf file."), status_code=400)

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return HTMLResponse(render_upload(session_id, "File is too large (max 20 MB)."), status_code=400)
    if not content.startswith(b"%PDF"):
        return HTMLResponse(render_upload(session_id, "That file is not a valid PDF."), status_code=400)

    # Same PDF in the same session -> same file_id, so re-uploads don't create duplicates.
    # The session id is part of the hash, so two users' copies never share chunks.
    file_id = hashlib.sha256(session_id.encode() + content).hexdigest()
    folder = session_dir(session_id)
    folder.mkdir(exist_ok=True)
    pdf_path = folder / f"{file_id}.pdf"

    if not pdf_path.exists():
        pdf_path.write_bytes(content)
        # don't let cleanup delete the session mid-upload
        sessions[session_id]["busy"] += 1
        try:
            # run in a thread so slow chunking/embedding doesn't freeze the server
            await run_in_threadpool(vector_db, file_id, pdf_path)
        except Exception as e:
            # so the same PDF can be processed again
            pdf_path.unlink(missing_ok=True)
            return HTMLResponse(render_upload(session_id, f"Could not process PDF: {escape(str(e))}"),
                                status_code=400)
        finally:
            sessions[session_id]["busy"] -= 1
            sessions[session_id]["last_seen"] = time.time()
        pdf_path.with_suffix(".name").write_text(Path(file.filename).name)

    return RedirectResponse(url=f"/chat/{file_id}", status_code=303)


# POST /delete/{file_id} : delete a PDF, its chunks and its chat history
@app.post("/delete/{file_id}")
async def delete_pdf(request: Request, file_id: str):
    pdf_path = get_pdf_path(request.state.session_id, file_id)
    await run_in_threadpool(delete_file, pdf_path)
    # forget this PDF's conversation
    sessions[request.state.session_id]["history"].pop(file_id, None)
    return RedirectResponse(url="/", status_code=303)


# POST /heartbeat : the pages call this every 30 seconds to keep the session alive
@app.post("/heartbeat")
def heartbeat():
    # The middleware already updated last_seen; this route just has to exist
    return {"ok": True}


# GET /chat/{file_id} : chat page for one PDF
@app.get("/chat/{file_id}", response_class=HTMLResponse)
def chat_page(request: Request, file_id: str):
    pdf_path = get_pdf_path(request.state.session_id, file_id)
    filename = escape(get_filename(pdf_path))
    return load_template("chat.html").replace("{filename}", filename).replace("{file_id}", file_id)


# POST /ask : search the PDF, ask the LLM, and return only the answer
@app.post("/ask")
def ask(request: Request, q: Question):
    # back to homepage if this PDF isn't in the user's session
    get_pdf_path(request.state.session_id, q.file_id)

    results = search_vector_db(q.file_id, q.question)

    # This session's conversation about this PDF. It lives in memory and is deleted with the session.
    history = sessions[request.state.session_id]["history"].setdefault(q.file_id, [])
    answer = chat(q.question, results, history)

    # Only the answer goes to the browser, never the chunks
    return {"answer": answer}


# start the server with: python app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
