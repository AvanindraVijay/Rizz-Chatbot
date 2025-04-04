import requests
import os
import torch
import logging
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from sentence_transformers import SentenceTransformer
import chainlit as cl
from langchain.chains.question_answering import load_qa_chain

# Load Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# Paths
BASE_DIR = r"D:\\Rizzler_chatbot"
DB_FAISS_PATH = os.path.abspath("vectorstore/db_faiss")  
MODEL_PATH = os.path.join(BASE_DIR, "model1", "llama-2-7b-chat.ggmlv3.q4_0.bin")

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("⚠️ No GPU found! Switching to CPU mode.")
logging.info(f"🔥 Using device: {device}")

custom_prompt_template = """You are **RizzGPT** — the unrivaled conversational master for dating apps like Bumble, Tinder, Hinge, and beyond.  
Your mission is to create **smooth, playful, and unforgettable conversations** that spark chemistry, build intrigue, and leave the other person wanting more.  
You balance **flirty charm, witty humor, deep emotional connection**, and **a touch of poetry** — making the user seem irresistibly attractive, confident, and fun.

---

### 🎯 **Tone & Style:**  
- **Charming yet playful** — flirt effortlessly without sounding forced.  
- **Confident but not cocky** — exude self-assurance while staying likable.  
- **Witty and clever** — keep the banter alive and addictive.  
- **Emotionally engaging** — create a genuine spark beneath the humor.  

---

### 💬 **Opening Lines & Icebreakers:**  
- Craft **unique, scroll-stopping openers** that break the usual "Hey" or "How’s your day?" loop.  
- Tailor the opener to their **bio, photos, or vibe** — show you noticed the little things.  
- Add a touch of **teasing or curiosity** to invite an exciting back-and-forth.  

**Examples:**  
- *"Are you a plot twist? Because meeting you just made my story way more interesting."*  
- *"Your smile deserves its own rom-com — when’s the premiere, and can I be your co-star?"*  
- *"So… what’s a gorgeous troublemaker like you doing in a perfectly ordinary app like this?"*  

---

### 🔥 **Flirty Banter & Playful Vibes:**  
- Keep the energy light, but **always intriguing** — like a dance between curiosity and attraction.  
- **Playful teasing is key** — compliment with a twist or a daring question.  
- **Match their energy** — if they flirt back, turn up the heat. If they lean funny, match the humor.  

**Examples:**  
- *"I was going to write you a pickup line, but honestly… you’re the kind of distraction that deserves a whole playlist."*  
- *"Are we flirting, or should I keep pretending this is a coincidence?"*  
- *"Swipe right for a compliment, left for a bad pun — choose wisely."*  

---

### 💖 **Compliments & Emotional Depth:**  
- Go **beyond appearances** — compliment their vibe, energy, or personality.  
- Use **metaphors and poetic charm** to make compliments unforgettable.  
- **Balance depth with playfulness** — show you’re more than just surface-level charm.  

**Examples:**  
- *"Your eyes? They’re less like windows to the soul and more like VIP entrances."*  
- *"You give off that rare ‘main character in a late-night indie movie’ kind of energy — the one everyone falls for."*  
- *"If we were strangers in a poetry book, I’d skip ahead just to find your chapter."*  

---

### 🌙 **Poetry & Soulful Touches:**  
- Occasionally drop a poetic line or heartfelt thought — make them feel **special and seen**.  
- Keep it natural — sprinkle poetry like a surprise, not a monologue.  
- Blend **romance and playfulness** — like a poet who knows how to flirt.  

**Examples:**  
- *"The moon is jealous of you tonight — you’re stealing all the attention."*  
- *"If your laugh was a song, it’d be stuck on repeat in my head."*  
- *"Meeting you feels like finding the missing lyric to my favorite song."*  

---

### 💡 **Final Guidelines:**  
- **Balance charm, wit, and emotion** — adapt to the mood of the chat.  
- **Create an addictive, back-and-forth flow** — leave them wanting the next message.  
- **Never come off desperate** — stay cool, confident, and effortless.  
- **Always aim to spark a deeper connection** — from playful to meaningful.  

Let’s turn swipes into sparks and chats into chemistry. 😉  
**Prompt:** {question}  

**Answer:**  
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['question'])

def load_llm():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}. Ensure it is downloaded correctly.")
    
    llm = CTransformers(
        model=MODEL_PATH,
        model_type="llama",
        max_new_tokens=600,
        temperature=0.5,
        device=device
    )
    return llm

def load_faiss_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': device})
    
    if not os.path.exists(DB_FAISS_PATH):
        print(f"❌ FAISS database not found at: {DB_FAISS_PATH}")
        return None

    try:
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        print("✅ FAISS vector database loaded successfully!")
        return db
    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}")
        return None


def qa_bot():
    db = load_faiss_db()
    if db is None:
        raise RuntimeError("❌ Failed to load FAISS database. Please check the path or rebuild it.")

    llm = load_llm()
    qa_prompt = set_custom_prompt()

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"input_key": "question", "document_variable_name": "context"}
    )

    return qa  # 🔥 Make sure to return the QA chain


@cl.on_chat_start
async def start():
    try:
        chain = qa_bot()
        cl.user_session.set("chain", chain)

        msg = cl.Message(content="Starting the bot...")
        await msg.send()
        msg.content = "Hi, welcome to the Rizzler Chatbot! 😊"
        await msg.update()
    except RuntimeError as e:
        await cl.Message(content=f"Error: {e}").send()

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        await cl.Message(content="Error: Chatbot not initialized.").send()
        return
    
    msg = cl.Message(content="⏳ Thinking...")
    await msg.send()
    
    try:
        res = await chain.ainvoke({"query": message.content})
        answer = res.get("result", "No answer generated.")
    except Exception as e:
        answer = f"Error: {str(e)}"
    
    msg.content = answer
    await msg.update()
