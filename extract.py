import os
import logging
import torch
import fitz  # PyMuPDF for header detection
import re  # Regular expressions for header matching
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PDF_PATHS = [
    "data/48_laws_of_powers.pdf",
    "data/101_Conversations.pdf",
    "data/Atomic_Attraction_The_Psychology_of_Attraction.pdf",
    "data/Captivate.pdf",
    "data/Cues.pdf",
    "data/Damn_good_advice.pdf",
    "data/How_to_Make_Anyone_Fall_in_Love_with_You.pdf",
    "data/How_to_Talk_to_Anyone_Anytime_Anywhere.pdf",
    "data/How_To_Win_Friends_And_Influence_People.pdf",
    "data/Like_Switch.pdf",
    "data/Madhushala.pdf",
    "data/Milk_and_Honey.pdf",
    "data/Models.pdf",
    "data/Never_Eat_Alone.pdf",
    "data/Radical_Honesty.pdf",
    "data/The_Art_of_Seductions.pdf",
    "data/The_Charisma_Myth.pdf",
    "data/The_Dictionary_of_Body_Language.pdf",
    "data/The_Game.pdf",
    "data/The_Poetry_of_Pablo_Neruda.pdf",
    "data/The_Sun_and_Her_Flowers.pdf",
    "data/Words_That_Change_Minds_The_14_Patterns.pdf",
    "data/Yes_Man.pdf",
    "data/dead_thoughts.pdf",
    "data/मरे_विचार.pdf"
]

DB_FAISS_PATH = "vectorstore/db_faiss"


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("No GPU found! Switching to CPU mode.")
logging.info(f"Using device: {device}")


def clean_text(text):
    """Removes unwanted sections based on keywords."""
    unwanted_sections = [
        "contents", "table of contents", "acknowledgment", "acknowledgements",
        "references", "bibliography", "index"
    ]
    
    lower_text = text.lower()
    for section in unwanted_sections:
        if section in lower_text[:2000]:
            return ""
    return text


def extract_headers_and_chunks(pdf_path):
    """Extracts text by headers (e.g., H1 and its content as one chunk)."""
    doc = fitz.open(pdf_path)
    chunks = []
    current_chunk = ""
    header_pattern = r"^(\d+\.\s*.*|[A-Z][A-Z\s]+)$"

    for page in doc:
        blocks = page.get_text("blocks")
        for block in blocks:
            text = block[4].strip()
            if not text:
                continue
            if re.match(header_pattern, text):
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk.strip()))
                current_chunk = text + "\n"
            else:
                current_chunk += text + " "

    if current_chunk:
        chunks.append(Document(page_content=current_chunk.strip()))
    
    return chunks


def load_pdf_data():
    documents = []
    
    for pdf_path in PDF_PATHS:
        if not os.path.exists(pdf_path):
            logging.warning(f"File not found: {pdf_path}")
            continue 

        chunks = extract_headers_and_chunks(pdf_path)
        documents.extend(chunks)
    
    if not documents:
        logging.error("No documents loaded from PDFs!")
        return []
    
    return documents


def create_vector_db():
    pdf_docs = load_pdf_data()
    
    if not pdf_docs:
        logging.error("No data available to create vector database!")
        return
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    db = FAISS.from_documents(pdf_docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    
    print(f"Vector DB saved to {DB_FAISS_PATH}")


if __name__ == "__main__":
    create_vector_db()
