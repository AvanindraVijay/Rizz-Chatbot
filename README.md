Rizzler Chatbot 🤖🔥
Rizzler is an advanced AI-powered chatbot built with FastAPI and LangChain, leveraging LLaMA-2-7B (Q4) for conversational AI. It utilizes FAISS for vector database storage and sentence-transformers for efficient embeddings. Designed to generate smooth and engaging conversations, Rizzler is optimized for speed, accuracy, and multilingual support.

🚀 Features
Uvicorn-powered API for fast and scalable server deployment.

LLaMA-2-7B (Q4) model for high-quality text generation.

LangChain framework for intelligent query handling.

FAISS vector database for efficient semantic search.

SentenceTransformers (all-MiniLM-L6-v2) for lightweight and effective embeddings.

Multilingual support with automatic language detection and translation.

Async support for handling multiple user requests efficiently.

🛠️ Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/AvanindraVijay/Rizz-Chatbot.git
cd Rizz-Chatbot
Create and activate a virtual environment (optional but recommended):

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🏗️ How to Run
1️⃣ Start the Chatbot API
Run the chatbot API using Uvicorn:

bash
Copy
Edit
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
This will start the API at: http://localhost:8000

2️⃣ Chat with the Bot
Interact with the bot via Chainlit UI:

bash
Copy
Edit
chainlit run app.py
🧠 Model Details
LLaMA-2-7B Q4: Runs locally for text generation.

FAISS Vector Store: Stores embeddings for fast retrieval.

Sentence Transformers: Uses all-MiniLM-L6-v2 for embeddings.

📁 Project Structure
bash
Copy
Edit
Rizzler_chatbot/
│── model1/                     # LLaMA model files
│── vectorstore/                 # FAISS vector database
│── app.py                       # Chainlit-based chatbot API
│── app1.py                      # Additional API logic
│── extract.py                   # Data extraction utilities
│── requirements.txt              # Dependencies
│── chainlit.md                   # Chainlit configuration
│── data/                         # Data storage folder
│── rizz/                         # Additional utilities
│── __pycache__/                   # Cached Python files
│── .chainlit/                     # Chainlit configurations
🔥 Technologies Used
Python (3.9+)

FastAPI (for serving API)

Chainlit (for chatbot UI)

Torch (for model execution)

LangChain (for query processing)

FAISS (for vector storage)

Sentence Transformers (for embeddings)

Deep Translator (for multilingual support)

