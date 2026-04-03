# RAG-chat-with-pdf
simple chat app. very broken.

<img width="2940" height="1510" alt="image" src="https://github.com/user-attachments/assets/50eff395-7f97-4516-b715-e0d65f53379b" />


# Chat with PDFs (RAG)

Simple Streamlit app to upload PDFs and ask questions about them using an LLM.

---

## Status

**Broken / incomplete**

* Chat input state bug (Streamlit session issue)
* No persistence (everything resets)
* Responses can be inconsistent
* Needs cleanup and fixes

---

## What it does

* Upload multiple PDFs
* Extract text
* Chunk + embed
* Store in FAISS
* Ask questions → get answers from LLM


---

## Tech

* Streamlit
* LangChain
* FAISS
* HuggingFace embeddings
* Groq LLM


---

## Run

```bash
pip install -r req.txt
streamlit run app.py
```

Create `.env`:

```
GROQ_API_KEY=your_key
```

---
<img width="1470" height="755" alt="Screenshot 2026-04-03 at 10 02 47 PM" src="https://github.com/user-attachments/assets/acc93893-8eed-4b31-ad94-a4a66cc20b13" />



## Notes

* Works sometimes after processing PDFs
* Breaks if session state gets messed up
* Not production ready

---

## TODO (if fixing later)

* Fix Streamlit session state bug
* Add source citations
* Persist vector DB
* Clean UI / state handling
