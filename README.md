# 🎬 YouTube Video Q&A Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** web application that lets you ask unlimited questions about any YouTube video — answered grounded in its transcript.

Built with **Python 3.12+**, **LangChain**, **Streamlit**, **Groq (LLaMA 3.3 70B)** or **Gemini 2.5 Flash**, **HuggingFace Embeddings**, and **ChromaDB**.

---

## ✨ Features

- 🔗 **YouTube URL input** — supports standard, short, embed, and Shorts URLs
- 📝 **Transcript loading** via LangChain's `YoutubeLoader`
- 🗂️ **Video metadata fetching** — title, channel, publish date, views via `pytubefix`
- ✂️ **Smart chunking** with `RecursiveCharacterTextSplitter` (1500 chars, 150 overlap)
- 🧠 **HuggingFace embeddings** (`all-MiniLM-L6-v2`) — runs locally, no API key needed
- 🗃️ **Persistent ChromaDB** with **MMR retrieval** (top-5 from pool of 30) for diverse chunks
- 🤖 **Dual LLM support** — switch between **Groq LLaMA 3.3 70B** and **Gemini 2.5 Flash** from the UI
- 🔄 **History-aware retrieval** — multi-turn conversations work correctly
- 💬 **Natural conversational answers** — no robotic "according to the transcript" phrasing
- 🌐 **General question support** — answers both video-specific and general knowledge questions
- 📄 **Expandable source chunks** — see exactly what the model used
- 🔁 **Duplicate video detection** — avoids re-embedding the same video
- 🧹 **Clear DB / Clear Chat** buttons
- 🎨 **Modern dark sidebar UI** with progress bar during processing
- 🔒 **Input sanitisation** — HTML-escaped chunk display, API key hashing in session state

---

## 🗂️ Project Structure

```
youtube_chatbot/
│
├── app.py                      # Streamlit UI + orchestration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .env / .env.example         # Environment variable templates
├── .gitignore
├── .streamlit/
│   └── config.toml             # Disables file watcher (suppresses torchvision warnings)
│
├── utils/
│   ├── __init__.py
│   ├── config.py               # App constants: chunks, retriever, embeddings, Gemini settings
│   ├── config_groq.py          # Groq model name, temperature, token limit, API key
│   ├── loader.py               # YouTube transcript loading + pytubefix metadata
│   ├── embeddings.py           # HuggingFace embeddings singleton (cached per process)
│   ├── vector_store.py         # ChromaDB: build (auto-clears), clear, MMR retriever
│   ├── rag_chain.py            # Gemini RAG chain (pure LCEL, LangChain 1.x compatible)
│   ├── rag_chain_groq.py       # Groq RAG chain with metadata doc injection
│   ├── prompts.py              # Contextualisation + QA prompts (shared by both chains)
│   └── helpers.py              # URL validation, video ID extraction, text utilities
│
├── chroma_db/                  # Auto-created; persistent vector store
└── assets/                     # Screenshots and static assets
```

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌──────────────────────────────┐
│  Contextualisation Chain     │  Rewrites follow-up questions into
│  (Groq or Gemini)            │  standalone queries using chat history
└─────────────┬────────────────┘
              │ standalone query
              ▼
┌──────────────────────────────────────────┐
│  ChromaDB MMR Retriever                  │  Fetches top-5 diverse chunks
│  (HuggingFace all-MiniLM-L6-v2)          │  from a pool of 30 candidates
│  + Metadata doc always injected directly │  (title, channel, date, views)
└─────────────┬────────────────────────────┘
              │ context string (metadata + chunks)
              ▼
┌──────────────────────────────┐
│  QA Chain                    │  Builds prompt, calls LLM,
│  (Groq or Gemini)            │  returns natural answer
└─────────────┬────────────────┘
              │ answer + source docs
              ▼
         Streamlit UI
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-chatbot.git
cd youtube-chatbot/youtube_chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key(s)

**For Groq** (recommended — free, fast, 14,400 req/day):
Get a free key at [console.groq.com](https://console.groq.com).

**For Gemini** (alternative — 20 req/day on free tier):
Get a key at [aistudio.google.com](https://aistudio.google.com/apikey).

Add to `.env` (optional — you can also enter keys in the sidebar):
```
GROQ_API_KEY=gsk_...
GOOGLE_API_KEY=AIza...
```

---

## 🚀 Running the Application

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

> The `.streamlit/config.toml` automatically disables Streamlit's file watcher to suppress noisy `torchvision` warnings from the `transformers` library.

---

## 🖥️ Usage

1. Select **LLM Provider** (Groq or Gemini) from the dropdown in the sidebar.
2. Enter the corresponding **API key** in the password field.
3. Paste a **YouTube URL** — any public video with captions.
4. Click **⚙️ Process Video** and watch the progress bar (Loading → Embedding → Building chain → Ready).
5. Ask questions in the chat box — press Enter to send.
6. Expand **"source chunks"** under any answer to see exactly what the model used.
7. Ask follow-up questions freely — chat history is maintained across turns.

---

## 🔄 Switching Between Groq and Gemini

No code changes needed. Simply:
1. Select the provider from the **LLM Provider** dropdown in the sidebar
2. Enter the matching API key
3. Click **⚙️ Process Video** again — the chain rebuilds with the new provider instantly (no re-embedding)

---

## 📸 Screenshots

> *(Add screenshots to the `assets/` folder and link them here)*

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `No transcript found` | The video has disabled captions. Try a video with auto-generated subtitles. |
| `Invalid YouTube URL` | Make sure the URL contains a valid 11-character video ID. |
| Groq API key invalid | Get a fresh key at [console.groq.com](https://console.groq.com). |
| Gemini API key invalid | Get a fresh key at [aistudio.google.com](https://aistudio.google.com/apikey). |
| Groq rate limit (429) | Free tier: 30 req/min, 14,400 req/day. Wait and retry, or switch to Gemini. |
| Gemini rate limit (429) | Free tier: 20 req/day for Gemini 2.5 Flash. Switch to Groq or wait until midnight PT. |
| Slow first run | HuggingFace downloads `all-MiniLM-L6-v2` (~90 MB) on first use. Fast after that. |
| Duplicate chunk retrieval | Click **Clear DB**, then **Process Video** to rebuild with current chunk settings. |
| DB keeps growing | Fixed — `build_vector_store` always clears the collection before re-embedding. |
| `chromadb` import errors | Run `pip install --upgrade chromadb langchain-chroma`. |
| `langchain.chains` not found | LangChain 1.x removed `langchain.chains` — the project uses `langchain_core` directly. |
| Metadata questions not answered | Clear DB and re-process — the metadata doc must be embedded fresh. |
| Port already in use | Run with `streamlit run app.py --server.port 8502`. |

---

## 🔮 Future Improvements

- [ ] Multi-language transcript support (`paraphrase-multilingual-MiniLM-L12-v2`)
- [ ] GPU acceleration for embeddings
- [ ] Playlist support (multiple videos in one session)
- [ ] Export chat history to PDF / Markdown
- [ ] Docker containerisation
- [ ] Streaming token-by-token output
- [ ] Authentication for multi-user deployments

---

## 🛡️ License

MIT License. See `LICENSE` for details.

---

## 🙏 Acknowledgements

- [LangChain](https://python.langchain.com/)
- [Streamlit](https://streamlit.io/)
- [Groq](https://console.groq.com/)
- [Google Gemini](https://ai.google.dev/)
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers)
- [ChromaDB](https://www.trychroma.com/)
- [pytubefix](https://github.com/JuanBindez/pytubefix)
