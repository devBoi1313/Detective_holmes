# 🔍 Sherlock Holmes Intelligence Bureau
### *Solving the Silver Blaze Mystery using NLP and Knowledge Graphs*

---

## What This Is

A **cinematic NLP investigation dashboard** that:
- Tokenizes and parses the Silver Blaze detective corpus
- Extracts characters, clues, and actions via NLTK
- Builds an interactive Knowledge Graph with PyVis
- Reconstructs Holmes-style logical deductions
- Renders everything as a premium Victorian-themed Streamlit app

**All logic is self-contained** — no external files, no APIs, no database.

---

## Project Files

```
sherlock_holmes_nlp/
├── app.py               ← Streamlit web app (primary)
├── notebook_backup.py   ← Terminal / Google Colab script (backup)
├── requirements.txt     ← 4 packages
└── README.md            ← This file
```

---

## ⚡ Option A — Streamlit Community Cloud (FREE, ~3 minutes)

### Step 1 — Create GitHub repo
1. Go to [github.com](https://github.com) → **New repository**
2. Name it `sherlock-holmes-nlp`
3. Upload these 3 files: `app.py`, `requirements.txt`, `notebook_backup.py`

### Step 2 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Select your repo → Branch: `main` → Main file: `app.py`
5. Click **Deploy** → Get a live public URL in ~60 seconds

**Cost: Free. No credit card. No server.**

---

## 💻 Option B — Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch app
streamlit run app.py
```
Opens at http://localhost:8501

---

## 🧪 Option C — Google Colab (Backup Demo)

Open a new Colab notebook and run these cells:

```python
# Cell 1 — Install
!pip install nltk pyvis pandas -q
```

```python
# Cell 2 — Upload files (or paste notebook_backup.py content)
# Upload notebook_backup.py via Files panel → then:
!python notebook_backup.py
```

```python
# Cell 3 — View the knowledge graph
from notebook_backup import show_graph_in_colab
show_graph_in_colab()
```

This prints all NLP outputs and renders the graph inline.

---

## 🧠 NLP Capabilities Demonstrated

| Capability | NLTK Function | Output Location |
|---|---|---|
| Tokenization | `word_tokenize`, `sent_tokenize` | Summary cards, Debug panel |
| POS Tagging | `pos_tag` | Debug panel, verb/noun extraction |
| Named Entity Recognition | `ne_chunk` + predefined list | Character Gallery |
| Frequency Distribution | `FreqDist`, `Counter` | Mention counts, noun ranking |
| Keyword Extraction | POS filter (NN tags) | Evidence Locker |
| Verb Extraction | POS filter (VB tags) | Action Registry |
| Relation Extraction | Co-occurrence rules | Knowledge Graph edges |
| Knowledge Graph | PyVis Network | Interactive graph section |
| Logical Inference | Rule chain | Holmes Deduction Engine |

---

## App Sections

```
1.  Hero               Victorian cinematic header + Holmes silhouette
2.  Executive Summary  5 live metric cards from NLP pipeline
3.  Case Timeline      8 events from sentence segmentation
4.  Character Gallery  9 characters with NER + mention counts
5.  Evidence Locker    8 clues from keyword/noun extraction
6.  Action Registry    7 actions from verb extraction
7.  Knowledge Graph    PyVis interactive network (30 nodes, 27 edges)
8.  Deduction Engine   Holmes reasoning chain (7 steps)
9.  Final Verdict      CASE CLOSED stamp with conclusions
10. NLP Methods        6 technique explanation cards
11. Debug Panel        Raw tokens, POS tags, nouns, verbs, dataframe
```

---

## Tech Stack

- **Python** — core logic
- **Streamlit** — web UI
- **NLTK** — NLP engine
- **PyVis** — interactive graph
- **pandas** — data display

---

## Quick Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| NLTK data missing | App auto-downloads on first run |
| Graph not rendering | Check browser JavaScript is enabled |
| Streamlit port busy | Run `streamlit run app.py --server.port 8502` |

---

*"When you have eliminated the impossible, whatever remains must be the truth."*
*— Sherlock Holmes*
