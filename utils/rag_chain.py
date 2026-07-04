"""
rag_chain.py
------------
Builds the full RAG pipeline using pure LangChain Core LCEL.
Compatible with LangChain 1.x.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma

from utils.config import GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_OUTPUT_TOKENS
from utils.prompts import contextualize_q_prompt, qa_prompt
from utils.vector_store import get_retriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Chroma, api_key: str) -> Any:
    """
    Construct and return the full RAG chain using pure LCEL.

    Args:
        vector_store: A loaded / built Chroma vector store.
        api_key:      Google Generative AI API key.

    Returns:
        A callable that accepts:
            {"input": <question>, "chat_history": <list of BaseMessages>}
        and returns:
            {"answer": <str>, "context": <list[Document]>, "input": <str>}
    """
    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
        google_api_key=api_key,
    )

    # ------------------------------------------------------------------
    # Retriever
    # ------------------------------------------------------------------
    retriever = get_retriever(vector_store)

    # ------------------------------------------------------------------
    # Contextualisation chain
    # Rewrites follow-up questions into standalone queries.
    # ------------------------------------------------------------------
    contextualize_chain = contextualize_q_prompt | llm | StrOutputParser()

    def get_standalone_question(inputs: dict) -> str:
        """Rewrite question using chat history if history exists."""
        if inputs.get("chat_history"):
            return contextualize_chain.invoke(inputs)
        return inputs["input"]

    # ------------------------------------------------------------------
    # Full chain — manual LCEL composition
    # ------------------------------------------------------------------
    def full_chain(inputs: dict) -> dict:
        """
        Execute the full RAG pipeline:
          1. Rewrite question if needed
          2. Retrieve relevant chunks
          3. Build prompt with context
          4. Call Gemini
          5. Return answer + source docs
        """
        # Step 1: get standalone query
        standalone_q = get_standalone_question(inputs)

        # Step 2: retrieve relevant chunks
        docs = retriever.invoke(standalone_q)

        # Step 3: format docs into a single context string
        context_str = _format_docs(docs)

        logger.info("RAG | docs=%d, context_len=%d", len(docs), len(context_str))

        # Step 4: build the prompt and call the LLM
        prompt_value = qa_prompt.invoke({
            "input": inputs["input"],
            "chat_history": inputs.get("chat_history", []),
            "context": context_str,
        })

        # Step 5: call LLM and parse output
        response = llm.invoke(prompt_value)
        answer = StrOutputParser().invoke(response)

        logger.info("Gemini answer preview: %s", answer[:200])

        return {
            "input": inputs["input"],
            "context": docs,       # raw docs for the UI to display
            "answer": answer,
        }

    logger.info("RAG chain built successfully (model: %s).", GEMINI_MODEL)
    return full_chain


def invoke_chain(
    chain: Any,
    question: str,
    chat_history: list,
) -> dict:
    """
    Invoke the RAG chain.

    Args:
        chain:        The callable from build_rag_chain().
        question:     The user's current question string.
        chat_history: List of HumanMessage / AIMessage objects.

    Returns:
        Dict with keys "answer" (str) and "context" (list[Document]).
    """
    logger.info("Invoking RAG chain | question length=%d chars", len(question))
    return chain({
        "input": question,
        "chat_history": chat_history,
    })
