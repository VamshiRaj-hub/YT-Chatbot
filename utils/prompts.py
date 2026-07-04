"""
prompts.py
----------
Defines all prompt templates used in the RAG pipeline.

Two templates are required by LangChain's history-aware retrieval chain:
  1. contextualize_q_prompt  – reformulates the user question using chat history
                               into a standalone question for the retriever.
  2. qa_prompt               – instructs Gemini to answer strictly from context.
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ---------------------------------------------------------------------------
# 1. Contextualisation prompt
#    Given the chat history and a follow-up question, produce a standalone
#    question that the retriever can use without needing the chat history.
# ---------------------------------------------------------------------------

CONTEXTUALIZE_Q_SYSTEM_PROMPT = (
    "You are a helpful assistant that reformulates user questions. "
    "Given the conversation history and the latest user question, "
    "produce a standalone question that can be understood without any "
    "prior context. Do NOT answer the question — only rewrite it if "
    "necessary. If the question is already standalone, return it as-is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# ---------------------------------------------------------------------------
# 2. QA / answer prompt
#    Instructs Gemini to answer strictly from the retrieved transcript chunks.
# ---------------------------------------------------------------------------

QA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant who has watched and fully understood the YouTube video "
    "provided. Answer the user's questions naturally and conversationally, as if you personally "
    "watched the video and are sharing what you know about it.\n\n"
    "RULES:\n"
    "1. For questions about the video, answer ONLY using information from the transcript context below.\n"
    "2. Never use phrases like 'according to the transcript', 'the transcript says', "
    "'based on the transcript', or any similar attribution. Just answer directly and naturally.\n"
    "3. Do NOT hallucinate facts about the video that are not in the transcript.\n"
    "4. If a question is about the video but the answer is not in the transcript, respond ONLY with: "
    '"The information is not available in the provided YouTube transcript."\n'
    "5. For general knowledge questions unrelated to the video (math, science, facts, etc.), "
    "answer them naturally using your own knowledge — you don't need to restrict yourself to the transcript.\n"
    "6. Keep answers clear, concise, and conversational.\n"
    "\n"
    "Transcript context:\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
