# CSE 5234 Lab 11 — Objective #3

RAG application that investigates how the Ohio State University College of Engineering planned to measure the success of the Master of Global Engineering Leadership (MGEL) degree.

The app follows the LangChain RAG tutorial: [Retrieval Augmented Generation (RAG)](https://python.langchain.com/docs/tutorials/rag/). The tutorial indexes a documentation site. This lab uses the same steps on one PDF:

Load → Split → Embed → Store → Retrieve → Generate

## Requirements met

| Requirement | Where it is met |
| --- | --- |
| Build a RAG application from the LangChain tutorial | `rag_app.py` loads the plan, splits it into chunks, embeds the chunks, stores them in an in-memory vector store, retrieves the passages for a question, and generates an answer from those passages only |
| Use `Engineering_Strategic-Plan_2018.pdf` | `data/Engineering_Strategic-Plan_2018.pdf` |
| Construct a prompt about how OSU planned to measure MGEL success | `MGEL_QUESTION` and `SYSTEM_PROMPT` in `rag_app.py` |

`MGEL_QUESTION` asks for each MGEL strategy and the metric, target, or deadline paired with it. `SYSTEM_PROMPT` tells the model to answer only from retrieved excerpts, to treat a strategy as an MGEL measure only when the strategy names MGEL, and to cite the PDF page.

## How to run

From this directory, with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python rag_app.py
```

The script prints the answer in the terminal and writes the same text to `mgel_answer.txt`. A saved run is already in that file.

On the first run, the embedding model and the local chat model download automatically. After `Download complete`, the terminal stays quiet while the model loads. The answer appears when generation finishes. A later run of the same command is faster.

To ask a different question about the plan:

```bash
python rag_app.py "Your question about the strategic plan"
```

Generation uses a local model (`mlx-community/Qwen2.5-7B-Instruct-4bit`) and does not need an API key. That model runs through MLX on Apple Silicon. To use a hosted model instead, set one of these environment variables and install the matching package from the comments in `requirements.txt`:

- `OPENAI_API_KEY` — `pip install langchain-openai`
- `ANTHROPIC_API_KEY` — `pip install langchain-anthropic`
- `GOOGLE_API_KEY` — `pip install langchain-google-genai`

## Result

MGEL is a strategic initiative under the Professional and Distance Education Programs (PDEP) goal on PDF page 13 (printed page 12). The plan measures it with three targets:

- Grow enrollment of the MGEL degree — enrollment of 40 students by 2020
- Deliver the degree at the highest quality — US News and World Report online engineering degree programs ranking within the top 25
- Add graduate distance-learning courses so MGEL has more technical tracks — two new technical tracks by 2020

The same section sets a separate target for professional and certificate programs in general: operate at least ten programs by 2020.
