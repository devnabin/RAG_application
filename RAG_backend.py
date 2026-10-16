
# Standard library
import math
import os
import re
from pathlib import Path

# Third-party packages
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

# load OPENAI_API_KEY, ENDPOINT and DEPLOYMENT_NAME from the .env file
load_dotenv()

# extract data return to chunk_pdf
def load_pdf_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        # collapse repeated spaces
        text = re.sub(r"[ \t]+", " ", text)
        # collapse big gaps
        text = re.sub(r"\n{3,}", "\n\n", text)
        pages.append(text.strip())
    return pages

# load make a chunk -> extract data
def chunk_pdf(pdf_path: Path) -> list[dict]:
    """Page by page chunking: every page of the PDF becomes one chunk.

    Returns a list like:
        [{"text": "text of page 1...", "page": "page1", "chunk_index": 0},
         {"text": "text of page 2...", "page": "page2", "chunk_index": 1}, ...]
    """
    # pages[0] is the text of page 1
    pages = load_pdf_pages(pdf_path)

    chunks = []
    for page_num, text in enumerate(pages, start=1):
        # skip empty pages (e.g. a blank or scanned page)
        if not text:
            continue
        chunks.append({
            "text": text,
            # saved in the metadata as page1, page2, ...
            "page": f"page{page_num}",
            "chunk_index": len(chunks),
        })
    return chunks


# connect to ChromaDB and return the collection that stores every PDF's chunks
def get_collection():
    # Setting up Azure OpenAI embedding model (keys come from the .env file)
    load_dotenv()
    openai_ef = OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_base=os.getenv("ENDPOINT"),
        model_name="text-embedding-3-small",
    )

    # Connect to the db (saved in a folder next to this file)
    client = chromadb.PersistentClient(path=str(Path(__file__).parent / "pdf_chromadb"))

    # One collection for all PDFs; each chunk remembers which PDF it came from in
    # its metadata, so /ask can search only this PDF with where={"file_id": file_id}
    return client.get_or_create_collection(name="pdf_chunks", embedding_function=openai_ef)


# remove all chunks of one PDF from the vector db
def delete_from_vector_db(file_id: str):
    # Delete (D) on CRUD: remove every chunk of this PDF
    get_collection().delete(where={"file_id": file_id})
    print(f"[{file_id[:8]}] deleted chunks from vector db")


# chunk the PDF, embed every chunk and store it in the vector db
def vector_db(file_id: str, pdf_path: Path) -> list[dict]:
    chunks = chunk_pdf(pdf_path)
    if not chunks:
        raise ValueError("No text found in this PDF (it may be a scanned image).")
    print(f"[{file_id[:8]}] {len(chunks)} chunks")

    collection = get_collection()

    # Remove old chunks of this PDF first, so re-processing never leaves extra chunks behind
    collection.delete(where={"file_id": file_id})

    # Create: add chunks in batches (Chroma embeds each document with OpenAI automatically)
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[f"{file_id}-{c['chunk_index']}" for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"file_id": file_id, "page": c["page"], "chunk_index": c["chunk_index"]}
                       for c in batch],
        )

    print(f"[{file_id[:8]}] stored {len(chunks)} chunks, total in collection: {collection.count()}")
    return chunks


# common words that keyword search ignores
STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "what", "which", "who", "whom", "how", "why", "when",
    "where", "does", "did", "this", "that", "these", "those", "with", "from", "about", "into", "its",
    "can", "you", "your", "tell", "give", "explain", "please", "there", "their", "have", "has", "had",
    "pdf", "paper", "document", "say", "says", "mean", "means", "any", "all", "some", "more", "most",
}


# find chunks that contain the question's words (exact match)
def keyword_search(file_id: str, question: str, n_results: int = 2) -> list[dict]:
    """Find chunks that literally contain the question's words.

    Semantic search matches meaning, so it can miss section names like "conclusion" or exact
    terms like "BLEU". Rare words count more: "conclusion" (on 1 page) beats "attention" (on all pages).
    """
    words = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if len(w) > 2 and w not in STOPWORDS}
    if not words:
        return []

    data = get_collection().get(where={"file_id": file_id})
    docs = [d.lower() for d in data["documents"]]

    scored = []
    for doc, original, meta in zip(docs, data["documents"], data["metadatas"]):
        score = 0.0
        for w in words:
            pages_with_word = sum(1 for d in docs if w in d)
            if w in doc:
                # rare word -> bigger score
                score += math.log(1 + len(docs) / pages_with_word)
        if score > 0:
            scored.append((score, original, meta))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"text": doc, "page": meta["page"], "distance": None} for _, doc, meta in scored[:n_results]]


# hybrid search: keyword matches + semantic matches for the question
def search_vector_db(file_id: str, question: str, n_results: int = 3) -> list[dict]:
    # Read (R) on CRUD: semantic search, only inside this PDF's chunks
    print(question)
    results = get_collection().query(
        query_texts=[question],
        n_results=n_results,
        where={"file_id": file_id},
    )
    semantic = [
        {"text": doc, "page": meta["page"], "distance": dist}
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
    ]

    # Hybrid search: keyword matches first, then semantic matches, without repeating a page
    combined, seen = [], set()
    for chunk in keyword_search(file_id, question) + semantic:
        if chunk["page"] not in seen:
            seen.add(chunk["page"])
            combined.append(chunk)
    print("pages sent to LLM:", [c["page"] for c in combined])
    return combined


# conversation with llm
# LLM client (Azure OpenAI); the keys come from the .env file
client = OpenAI(
    base_url=os.getenv("ENDPOINT"),
    api_key=os.getenv("OPENAI_API_KEY"),
)
model = os.getenv("DEPLOYMENT_NAME")

# only send the last 10 messages (5 questions + 5 answers) to keep requests small
MAX_HISTORY = 10

# instructions that tell the LLM how to answer
SYSTEM_PROMPT = """
Act like a helpful PDF assistant. With every question you get some chunks of the user's PDF,
each one marked with its page like [page3]. Answer in a soft, sweet and clear tone.

Rules:
1. Answer only from the PDF chunks and the earlier conversation. Never make up facts.
2. If the answer is not in the chunks, say: I could not find that in your PDF.
3. Mention the page you used at the end, like (page3).
4. For greetings like hello, hi or good morning, or questions not related to the PDF, reply:
   I am your PDF assistant. You can ask me anything about your PDF.
5. Do not use markdown, new lines or tabs in the response.
6. Answer in a concise manner.
"""

# Few-shot examples: they teach the LLM the answer style. They use a made-up PDF,
# so the LLM learns the format without mixing these facts into the user's real PDF.
FEW_SHOT = [
    {
        "role": "user",
        "content": "PDF chunks:\n[page2] The study was carried out in 2023 with 120 students from three schools.\n"
                   "[page5] Results show online learning improved test scores by 12 percent.\n\n"
                   "Question: How many students were in the study?",
    },
    {
        "role": "assistant",
        "content": "The study included 120 students from three schools. (page2)",
    },
    {
        "role": "user",
        "content": "PDF chunks:\n[page2] The study was carried out in 2023 with 120 students from three schools.\n"
                   "[page5] Results show online learning improved test scores by 12 percent.\n\n"
                   "Question: Who funded the study?",
    },
    {
        "role": "assistant",
        "content": "I could not find that in your PDF.",
    },
    {
        "role": "user",
        "content": "PDF chunks:\n[page1] Introduction to renewable energy and solar panels.\n\n"
                   "Question: Hello, good morning!",
    },
    {
        "role": "assistant",
        "content": "I am your PDF assistant. You can ask me anything about your PDF.",
    },
]


# send the question + PDF chunks + chat history to the LLM and return its answer
def chat(question: str, chunks: list[dict], history: list[dict]) -> str:
    """Ask the LLM about the PDF.

    question: what the user typed
    chunks:   results from search_vector_db(), like [{"text": ..., "page": "page3", ...}]
    history:  this session's conversation for this PDF; the new question and answer are added to it
    """
    context = "\n".join(f"[{c['page']}] {c['text']}" for c in chunks)

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT,
            # earlier questions and answers, so follow-up questions make sense
            *history[-MAX_HISTORY:],
            {"role": "user", "content": f"PDF chunks:\n{context}\n\nQuestion: {question}"},
        ],
        max_output_tokens=300,
    )
    answer = response.output_text

    # Save only the question (not the chunks) so the history stays small
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    print(history)
    return answer
