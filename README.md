# RAG_EDU: Chat with your PDF

A small learning project where you upload a PDF and ask questions about it. The answers come from an LLM, but the LLM only sees the parts of the PDF that matter for your question. That idea is called RAG (Retrieval-Augmented Generation), and this project is my attempt to build it from scratch with FastAPI, ChromaDB and Azure OpenAI.

## What is RAG?

An LLM doesn't know what's inside your PDF. You could paste the whole PDF into the prompt, but long documents get expensive, slow, and sometimes don't fit at all.

RAG solves this in two parts:

1. **Before any question:** split the PDF into small pieces (chunks), turn each piece into an embedding (a list of numbers that captures its meaning), and save them in a vector database.
2. **For every question:** find the few chunks that best match the question, and give only those to the LLM along with the question.

So the LLM answers from your document instead of guessing, and it can tell you which page the answer came from.

## How the app works, step by step

### 1. Upload

The user opens the homepage and uploads a PDF.

- `app.py` checks that the file really is a PDF and is under 20 MB.
- The file is saved in `uploads/<session_id>/`. Each browser gets its own session, so users never see each other's files.
- The file is named with a hash of its content. Uploading the same PDF twice doesn't create a second copy.

### 2. Read and chunk

`vector_db()` in `RAG_backend.py` takes over.

- `load_pdf_pages()` reads the text of every page with `pypdf` and cleans up extra spaces.
- `chunk_pdf()` makes one chunk per page. Each chunk keeps its page as metadata: `page1`, `page2`, and so on.

Page-by-page chunking is the simplest option. It keeps page numbers easy, which is nice for showing where an answer came from.

### 3. Embed and store

Still inside `vector_db()`:

- The chunks are added to a ChromaDB collection called `pdf_chunks`.
- ChromaDB calls the OpenAI embedding model (`text-embedding-3-small`) on its own, because the embedding function is attached to the collection in `get_collection()`.
- Every chunk is stored with its text, its embedding and its metadata (`file_id`, `page`, `chunk_index`).
- The database lives in the `pdf_chromadb/` folder.

After this, the user is redirected to the chat page.

### 4. Ask a question

The chat page sends the question to `POST /ask`. Then `search_vector_db()` finds the relevant pages using two kinds of search:

- **Semantic search:** ChromaDB embeds the question and returns the 3 pages whose meaning is closest.
- **Keyword search:** `keyword_search()` looks for pages that literally contain the words in the question. Rare words count more than common ones.

Why both? Semantic search is good with meaning but bad with labels. Asking "what is the conclusion?" used to return the wrong pages, because the word "conclusion" is a section title and not really a topic. Keyword search catches those cases. The results are merged, keyword matches first, without repeating a page.

### 5. Generate the answer

`chat()` sends the LLM:

1. A system prompt with the rules (answer only from the PDF, mention the page, say so when the answer isn't there).
2. A few example conversations (few-shot prompting) about a made-up PDF, so the model learns the answer style.
3. The last 10 messages of this conversation, so follow-up questions like "explain that more simply" make sense.
4. The matching chunks and the new question.

Only the final answer goes back to the browser, never the raw chunks.

### 6. Clean up

- The user can delete a PDF from the homepage. That removes the file, its chunks in ChromaDB and its chat history.
- Both pages send a small "heartbeat" request every 30 seconds. If a session goes quiet for 5 minutes (tab or browser closed), the server deletes its files and chunks automatically.
- Sessions are kept in memory, so when the server restarts, everything in `uploads/` is cleared.

## The whole flow in one picture

```
Upload PDF
   |
   v
POST /upload --> vector_db()
                   |-- load_pdf_pages()   read text of each page
                   |-- chunk_pdf()        one chunk per page (page1, page2, ...)
                   '-- collection.add()   embed with OpenAI, store in ChromaDB
   |
   v
Chat page
   |
   v
POST /ask --> search_vector_db()   keyword search + semantic search
          --> chat()               system prompt + examples + history + chunks + question
          --> answer sent back to the browser
```

## Project structure

```
RAG_EDU/
├── app.py              FastAPI app: pages, upload, delete, sessions, /ask
├── RAG_backend.py      RAG logic: read PDF, chunk, embed, search, chat with the LLM
├── templates/
│   ├── upload.html     homepage with the upload form and the list of your PDFs
│   └── chat.html       chat page for one PDF
├── uploads/            uploaded PDFs, one folder per session (not in git)
└── pdf_chromadb/       ChromaDB database (not in git)
```

## API routes

| Route | What it does |
|---|---|
| `GET /` | Homepage with the upload form and your PDFs |
| `POST /upload` | Saves the PDF, chunks and embeds it, then opens the chat |
| `GET /chat/{file_id}` | Chat page for one PDF |
| `POST /ask` | Takes `{file_id, question}` and returns `{answer}` |
| `POST /delete/{file_id}` | Deletes a PDF, its chunks and its history |
| `POST /heartbeat` | Keeps the session alive |

If a PDF can't be found (deleted, expired session, or someone else's link), the user is sent back to the homepage.

FastAPI also gives you a test page at `http://127.0.0.1:8000/docs` where you can try each route.

## Setup

You need Python 3.10 or newer and an Azure OpenAI (or OpenAI) account.

1. Install the packages (from the project root, with the virtual environment active):

   ```bash
   pip install fastapi uvicorn python-multipart pypdf chromadb openai python-dotenv
   ```

2. Add these to the `.env` file in the project root:

   ```
   OPENAI_API_KEY=your-key
   ENDPOINT=your-endpoint-url
   DEPLOYMENT_NAME=your-chat-model-deployment
   ```

   The embedding model name is set in `get_collection()` in `RAG_backend.py`.

3. Run the app from inside `RAG_EDU`:

   ```bash
   python app.py
   ```

4. Open `http://127.0.0.1:8000` and upload a PDF.

## Settings you can change

| Setting | File | Default | Meaning |
|---|---|---|---|
| `MAX_FILE_SIZE` | app.py | 20 MB | Biggest PDF allowed |
| `SESSION_TIMEOUT` | app.py | 5 minutes | How long a quiet session lives |
| `CLEANUP_EVERY` | app.py | 60 seconds | How often expired sessions are checked |
| `n_results` in `search_vector_db()` | RAG_backend.py | 3 | Pages from semantic search |
| `n_results` in `keyword_search()` | RAG_backend.py | 2 | Pages from keyword search |
| `MAX_HISTORY` | RAG_backend.py | 10 | Past messages sent to the LLM |
| `max_output_tokens` in `chat()` | RAG_backend.py | 300 | Longest answer the LLM can give |

## Things to know

- **Uploads disappear when you edit code.** The app runs with `reload=True`, so saving a `.py` file restarts the server, and a restart clears all uploads. Upload your PDF again after each change.
- **Scanned PDFs don't work.** If a PDF is just images of pages, `pypdf` finds no text and the upload fails with "No text found".
- **Figures make messy text.** Charts and diagrams often come out as one word per line. They still get embedded, and sometimes show up in search results.
- **Whole pages are big chunks.** One page can cover several topics, which blurs its embedding. Smaller chunks (paragraphs, with the page number still in the metadata) would make search more precise.
- **Vague follow-ups search badly.** History helps the LLM understand "explain that", but the search itself only looks at the latest question.

## Ideas for next steps

- Split pages into paragraph-sized chunks and compare the answers.
- Clean figure text before embedding (join one-word lines, drop tokens like `<pad>`).
- Rewrite vague follow-up questions into full questions before searching, using the chat history.
- Let one chat search across all of a user's PDFs.
- Show the source pages under each answer in the UI.
