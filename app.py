import os
import logging
import traceback
import torch
import asyncio
import chainlit as cl
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from ctransformers import AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.language_models import BaseLLM
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
import re

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.abspath("D:/Rizzler_chatbot")
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")
MODEL_PATH = os.path.join(BASE_DIR, "model1", "llama-2-7b-chat.ggmlv3.q4_0.bin")

# Load Sentence Transformer model
try:
    sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    logger.info("Sentence Transformer loaded successfully!")
except Exception as e:
    logger.error("Failed to load Sentence Transformer: %s", e)
    raise

# Determine device
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

# Load LLaMA model
try:
    llm_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        model_type="llama"
    )
    logger.info("LLaMA model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load LLaMA model: {e}")
    raise

from langchain_core.outputs import LLMResult, Generation

class CTransformersLLM(BaseLLM):
    model: object = Field(..., description="Model instance")

    # Allow arbitrary types for Pydantic compatibility
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Synchronous call to the model."""
        output = self.model(prompt)
        if stop:
            for stop_seq in stop:
                if stop_seq in output:
                    output = output[:output.index(stop_seq)]
                    break
        return output.strip()

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        """Asynchronous call to the model (required for LangChain async support)."""
        return self._call(prompt, stop, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "ctransformers"

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,  # 🔥 FIXED: Added run_manager for compatibility
        **kwargs: Any
    ) -> LLMResult:
        """Updated _generate method to handle extra arguments and callbacks."""
        generations = []

        for prompt in prompts:
            text = self._call(prompt, stop=stop, **kwargs)
            generations.append([Generation(text=text)])

        return LLMResult(generations=generations)


# Wrap the model
wrapped_model = CTransformersLLM(model=llm_model)

# Custom Prompt Template
def set_custom_prompt():
    return PromptTemplate(
        template="""
        You are RizzGPT — the unrivaled conversational master for dating apps.
        Your mission is to create smooth, playful, and unforgettable conversations.

        Context: {context}
        Question: {question}
        Answer:""",
        input_variables=['context', 'question']
    )

# Language detection & translation
def translate_query(query: str) -> tuple[str, str]:
    try:
        lang = detect(query)
        if lang == "hi":
            translated = GoogleTranslator(source='hi', target='en').translate(query)
            return translated, "hi"
        return query, "en"
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return query, "unknown"

# Intent classification
def classify_intent(query: str) -> str:
    patterns = {
        "greetings": re.compile(r"\b(hi|hello|hey)\b", re.IGNORECASE),
        "farewell": re.compile(r"\b(bye|goodbye|see you)\b", re.IGNORECASE),
        "pickup": re.compile(r"\b(give me a pickup line|flirt with me)\b", re.IGNORECASE)
    }
    for intent, pattern in patterns.items():
        if pattern.search(query):
            return intent
    return "general"

# FAISS & RetrievalQA setup
def qa_bot():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )

        # Check if FAISS database exists
        if not os.path.exists(DB_FAISS_PATH):
            raise FileNotFoundError(f"FAISS database not found at {DB_FAISS_PATH}")

        vectorstore = FAISS.load_local(
            DB_FAISS_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

        qa_chain = RetrievalQA.from_chain_type(
            llm=wrapped_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": set_custom_prompt()}
        )

        return vectorstore, qa_chain
    except Exception as e:
        logger.error("Error in qa_bot: %s", traceback.format_exc())
        raise

# Chat session initialization
@cl.on_chat_start
async def start():
    try:
        db, chain = qa_bot()
        cl.user_session.set("vectorstore", db)
        cl.user_session.set("chain", chain)
        await cl.Message(content="Welcome to Rizzler! Ready for some fun? 😏").send()
    except Exception as e:
        logger.error("Error in start: %s", traceback.format_exc())
        await cl.Message(content="Failed to initialize. Try again later.").send()

# Streaming response generator
async def generate_response(query: str):
    try:
        chain = cl.user_session.get("chain")
        if not chain:
            yield "Chat not properly initialized!"
            return

        # Translate query if needed
        translated_query, lang = translate_query(query)

        # Get response from QA chain
        result = await chain.acall({"query": translated_query})
        response_text = result["result"]

        if not response_text or "I don't know" in response_text.lower():
            yield "Looks like I need more rizz!"
            return

        # Stream the response word by word
        for word in response_text.split():
            yield word + " "
            await asyncio.sleep(0.05)

    except Exception as e:
        logger.error("Error generating response: %s", traceback.format_exc())
        yield "Oops, my rizz engine stalled! Try again."

# Handle incoming messages
@cl.on_message
async def main(message: cl.Message):
    try:
        # Show typing indicator
        typing_msg = await cl.Message(content="Typing...").send()

        # Generate and stream response
        response_text = ""
        async for chunk in generate_response(message.content):
            response_text += chunk

        # Update the message with final response
        # await typing_msg.edit(content=response_text.strip())
        await cl.Message(content=response_text).send()

    except Exception as e:
        logger.error("Error in main: %s", traceback.format_exc())
        await cl.Message(content="An error occurred. Please try again.").send()
