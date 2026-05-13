# RAG Evaluation Dashboard

A system that automatically evaluates RAG (Retrieval Augmented Generation) pipeline 
quality using RAGAS metrics — measuring faithfulness, answer relevancy, context 
precision, and context recall across configurable pipeline settings.

**Live App:** [Click here to view the app](your-streamlit-url-here)

---

## Why RAG Evaluation Matters

Most RAG systems are tested manually — a developer asks a few questions, the answers 
look reasonable, and the system is deployed. This approach misses systematic failures 
and makes it impossible to measure whether changes actually improve quality.

This evaluation dashboard treats RAG quality as a measurable engineering problem — 
running automated experiments, scoring results objectively, and identifying exactly 
which component of the pipeline is failing.

---

## The Four RAGAS Metrics

**Faithfulness** — Are the answers grounded in the retrieved context, or is the model 
hallucinating from its training data? Score of 1.0 means every claim in the answer 
is supported by the retrieved chunks.

**Answer Relevancy** — Does the answer actually address what was asked? A high 
faithfulness but low relevancy means the model is staying within the context but 
not focusing on the question.

**Context Precision** — When chunks are retrieved, are they relevant to the question? 
Low precision means retrieval is returning noisy, irrelevant chunks alongside useful ones.

**Context Recall** — Did retrieval find all the information needed to answer correctly? 
Low recall means relevant information exists in the document but wasn't retrieved.

---

## Experimental Findings

Four experiments were run against Tesla's 2023 Annual Report using 8 test questions 
with ground truth answers.

| Experiment | Settings | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Overall |
|---|---|---|---|---|---|---|
| Baseline | cs=1000, k=3 | 0.59 | 0.55 | 0.85 | 0.62 | 0.66 |
| More chunks | cs=1000, k=5 | 0.54 | 0.45 | 0.87 | 0.88 | 0.69 |
| Smaller chunks + stricter prompt | cs=500, k=3 | 0.94 | 0.56 | 0.89 | 0.50 | 0.72 |
| Fixed ground truth | cs=500, k=3 | 0.91 | 0.78 | 0.82 | 0.62 | 0.78 |

**Key findings:**

**Faithfulness 0.59 → 0.91** — Rewriting the system prompt with strict rules forcing 
the model to only use retrieved context dramatically reduced hallucination. This proved 
the faithfulness problem was prompt engineering, not retrieval quality.

**Answer Relevancy 0.55 → 0.78** — Fixing ground truth answers to match how the 
document actually expresses information (millions vs billions) revealed that the model 
was often correct but being penalised for format mismatch.

**Increasing k improves recall but hurts faithfulness** — k=5 improved context recall 
from 0.62 to 0.88 but reduced faithfulness from 0.59 to 0.54. More context confused 
the model into mixing retrieved information with training knowledge.

**Context recall remains at 0.62** — Two questions consistently fail retrieval. 
The fix is hybrid search combining semantic similarity with keyword matching — 
financial figures are better found by exact keyword match than semantic similarity.

---

## How It Works

Upload PDF document
↓
Configure pipeline settings (chunk size, overlap, k)
↓
System automatically asks 8 test questions
↓
RAG pipeline answers each question and records retrieved chunks
↓
RAGAS uses GPT as a judge to score each answer on 4 metrics
↓
Dashboard displays scores, radar chart, and per-question analysis
↓
Results saved to JSON for comparison across experiments

---

## How to Run Locally

```bash
git clone https://github.com/RiSHLEO/rag-evaluation
cd rag-evaluation
pip install -r requirements.txt
```

Create a `.env` file: OPENAI_API_KEY=your-key-here

Then run:
```bash
cd app
streamlit run app.py
```

Upload the Tesla 2023 Annual Report PDF and click Run Evaluation.

---

## What I Would Improve With More Time

- Add hybrid search combining semantic similarity with BM25 keyword search to 
  improve context recall from 0.62
- Support custom test question upload — users define their own questions and 
  ground truths for any document
- Add experiment comparison view — plot multiple evaluation runs side by side 
  to visualise the impact of each change
- Add automatic ground truth generation — use GPT to generate question/answer 
  pairs from the document so manual ground truth creation isn't needed
- Support batch evaluation across multiple documents simultaneously