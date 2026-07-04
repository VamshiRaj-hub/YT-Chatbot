"""
rag_chain_groq.py
-----------------
RAG pipeline using Groq (LLaMA 3.3 70B) as the LLM backend.
Identical logic to rag_chain.py — only the LLM provider changes.

Free tier limits (as of 2026):
  - 14,400 requests/day
  - 30 requests/minute
  - No credit card required
  - Get your key at: https://console.groq.com
"""

import logging
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_chroma import Chroma

from utils.config_groq import GROQ_MODEL, GROQ_TEMPERATURE, GROQ_MAX_OUTPUT_TOKENS
from utils.prompts import contextualize_q_prompt, qa_prompt
from utils.vector_store import get_retriever

logger = logging.getLogger(__name__)


def _format_docs(docs: list[Document]) -> str:
    """Join retrieved Document objects into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain_groq(vector_store: Chroma, api_key: str) -> Any:
    """
    Construct and return the full RAG chain using Groq as the LLM.

    Args:
        vector_store: A loaded / built Chroma vector store.
        api_key:      Groq API key.

    Returns:
        A callable that accepts:
            {"input": <question>, "chat_history": <list of BaseMessages>}
        and returns:
            {"answer": <str>, "context": <list[Document]>, "input": <str>}
    """
    # ------------------------------------------------------------------
    # LLM — Groq LLaMA 3.3 70B
    # ------------------------------------------------------------------
    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_OUTPUT_TOKENS,
        groq_api_key=api_key,
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
    # Full RAG chain
    # ------------------------------------------------------------------
    def full_chain(inputs: dict) -> dict:
        """
        Execute the full RAG pipeline:
          1. Rewrite question into standalone query if needed
          2. Always fetch the metadata doc (type=metadata) directly
          3. Retrieve relevant transcript chunks from ChromaDB
          4. Combine metadata + chunks as context
          5. Build QA prompt and call Groq LLM
          6. Return answer + source docs
        """
        # Step 1: standalone question
        standalone_q = get_standalone_question(inputs)

        # Step 2: always fetch the metadata doc so metadata questions work
        try:
            meta_results = vector_store._collection.get(
                where={"type": "metadata"},
                limit=1,
            )
            meta_docs = [
                Document(page_content=doc, metadata=meta)
                for doc, meta in zip(
                    meta_results.get("documents", []),
                    meta_results.get("metadatas", [{}]),
                )
            ]
        except Exception:
            meta_docs = []

        # Step 3: retrieve relevant transcript chunks
        retrieved_docs = retriever.invoke(standalone_q)

        # Step 4: combine — metadata first, then retrieved chunks (deduplicated)
        seen = set()
        all_docs = []
        for doc in meta_docs + retrieved_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)

        context_str = _format_docs(all_docs)

        logger.info(
            "Groq RAG | meta_docs=%d, retrieved=%d, total=%d, context_len=%d",
            len(meta_docs), len(retrieved_docs), len(all_docs), len(context_str),
        )

        # Step 5: build prompt and call LLM
        prompt_value = qa_prompt.invoke({
            "input": inputs["input"],
            "chat_history": inputs.get("chat_history", []),
            "context": context_str,
        })

        response = llm.invoke(prompt_value)
        answer = StrOutputParser().invoke(response)

        logger.info("Groq answer preview: %s", answer[:150])

        return {
            "input": inputs["input"],
            "context": all_docs,
            "answer": answer,
        }

    logger.info("Groq RAG chain built successfully (model: %s).", GROQ_MODEL)
    return full_chain


def invoke_chain_groq(
    chain: Any,
    question: str,
    chat_history: list,
) -> dict:
    """
    Invoke the Groq RAG chain.

    Args:
        chain:        The callable from build_rag_chain_groq().
        question:     The user's current question string.
        chat_history: List of HumanMessage / AIMessage objects.

    Returns:
        Dict with keys "answer" (str) and "context" (list[Document]).
    """
    logger.info("Invoking Groq RAG chain | question length=%d chars", len(question))
    return chain({
        "input": question,
        "chat_history": chat_history,
    })
