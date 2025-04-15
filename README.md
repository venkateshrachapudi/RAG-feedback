# Wiki: RAG Evaluation Pipeline with Feedback Logging, Prompt Improvement, and Auto-Retraining

## Overview

This document provides a comprehensive technical guide for building an advanced Retrieval-Augmented Generation (RAG) pipeline. The pipeline integrates:

- **LangChain** for orchestrating LLM and retrieval workflows
- **Claude 3.5** for generating high-quality answers
- **OpenSearch** for vector-based document retrieval
- **RAGAS and TruLens** for multi-dimensional evaluation
- **PostgreSQL** for logging user feedback
- **Scikit-learn** for auto-retraining a feedback classifier to iteratively improve prompt quality and response accuracy

## Architecture Diagram (Conceptual)

```
+-------------+        +--------------+        +--------------+       +-------------+
| User Query  +------->+ RAG Pipeline +------->+ Claude 3.5   +-----> | Answer/Docs |
+-------------+        | (LangChain)  |        | LLM Response |       +-------------+
                           |                            |
                           |                            v
                           |                     +------------+
                           |                     | TruLens UI |
                           |                            ^
                           v                            |
                 +-------------------+         +-------------------+
                 | OpenSearch Vector |<--------+   Embedding Store |
                 +-------------------+         +-------------------+
                           |
                           v
                 +-------------------------+
                 | Feedback Logger (SQL)   |
                 +-------------------------+
                           |
                           v
           +-----------------------------+
           | Prompt Improvement (Claude) |
           +-----------------------------+
                           |
                           v
           +------------------------------+
           | Auto-Retrain Classifier (ML) |
           +------------------------------+
```

---

## Components

### 1. LLM and RAG Chain

- **LLM:** Claude 3.5 Sonnet accessed via `langchain-anthropic`
- **Retriever:** `OpenSearchVectorSearch` configured with `OpenAIEmbeddings`
  ```python
  embeddings = OpenAIEmbeddings()
  vectorstore = OpenSearchVectorSearch(index_name="docs", embedding=embeddings, ...)
  ```
- **RAG Chain:** `RetrievalQA.from_chain_type(...)` returns the answer and source documents

### 2. Dataset Format

```json
[
  {
    "question": "What is the role of metformin in diabetes management?",
    "ground_truth": "Metformin reduces hepatic glucose production and increases insulin sensitivity."
  },
  ...
]
```

### 3. Evaluation Metrics

#### **RAGAS**

- Evaluates response quality using four metrics:
  - **Faithfulness** (does the answer align with retrieved docs?)
  - **Context Precision** (is context relevant?)
  - **Answer Relevancy** (semantic similarity to query)
  - **Answer Correctness** (alignment with ground truth)

#### **TruLens**

- Captures hallucination scores
- Groundedness tracing (source vs generation)
- Explorable in the `trulens-eval` dashboard

### 4. Feedback Storage

- PostgreSQL table `rag_feedback`

```sql
CREATE TABLE rag_feedback (
  id SERIAL PRIMARY KEY,
  question TEXT,
  answer TEXT,
  ground_truth TEXT,
  sources TEXT,
  user_rating INTEGER,
  user_comment TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);
```

- SQLAlchemy is used to write pandas DataFrames directly:

```python
df.to_sql("rag_feedback", engine, if_exists="append", index=False)
```

### 5. Prompt Improvement using Claude

- Prompts with rating < 4 are processed by Claude 3.5:

```python
prompt = f"Suggest how to improve this prompt based on this user feedback: {feedback}"
response = anthropic.invoke(prompt)
```

- Resulting prompts are stored in a `suggested_prompts` table

### 6. Auto-Retraining Prompt Classifier

- Pipeline:
  - Load historical feedback from PostgreSQL:
    ```python
    df = pd.read_sql("SELECT question, user_comment, user_rating FROM rag_feedback", engine)
    df["label"] = df["user_rating"].apply(lambda x: 1 if x >= 4 else 0)
    df["input_text"] = df["question"] + " " + df["user_comment"]
    ```
  - Use `TfidfVectorizer` to transform `input_text`
  - Train a `LogisticRegression` classifier:
    ```python
    X = vectorizer.fit_transform(df["input_text"])
    y = df["label"]
    model = LogisticRegression().fit(X, y)
    ```
  - Save artifacts:
    ```python
    joblib.dump(model, "feedback_classifier.pkl")
    joblib.dump(vectorizer, "feedback_vectorizer.pkl")
    ```

### Step 7: Auto-Train Classifier with Verified Feedback

- **Labeling:** Assign binary labels to feedback:
  - `rating >= 4` → 1 (positive)
  - `rating < 4` → 0 (negative)
- **Input Format:** Concatenate `question + user_comment` as features
- **Model Training:**
  - Vectorize using TF-IDF
  - Train Logistic Regression or Gradient Boosting
- **Verification Logic:**
  - Compare user rating with `ground_truth` similarity using a semantic similarity scorer (e.g., `sentence-transformers`)
  - If semantic similarity between model answer and ground\_truth exceeds threshold (e.g., 0.85), trust user feedback
  - If similarity is low but user rated it high, discard feedback
- **Integration with RAG:**
  - Use model + verification to identify poor quality prompts
  - Automatically rewrite these prompts:
    ```python
    rewrite_prompt = f"Rewrite the following prompt due to poor rating and verified misalignment with ground truth: {question}"
    improved = anthropic.invoke(rewrite_prompt)
    ```
  - Store rewritten prompt, route through RAG again
  - Retain both original and rewritten versions for A/B testing

---

## Tech Stack

| Component        | Technology                     |
| ---------------- | ------------------------------ |
| LLM              | Claude 3.5 via Anthropic       |
| Vector DB        | OpenSearch + OpenAI Embeddings |
| RAG Framework    | LangChain                      |
| Evaluation       | RAGAS, TruLens                 |
| Feedback Logging | PostgreSQL + SQLAlchemy        |
| Prompt Fixing    | Claude 3.5                     |
| ML Classifier    | scikit-learn + TF-IDF          |
| Dashboard        | Trulens, Streamlit (planned)   |

---

## Workflow Steps (Expanded)

### Step 1: Environment Setup

```bash
pip install langchain trulens ragas openai psycopg2 scikit-learn joblib sqlalchemy pandas sentence-transformers
```

### Step 2: Collect Inputs

- Questions
- Claude 3.5 answers
- Source docs (from retriever)
- Ground truth for evaluation

### Step 3: Run RAG

- Chain returns answer and `source_documents`
- Save outputs to PostgreSQL with timestamp

### Step 4: Evaluate Quality

- Convert to `ragas.Dataset`
- Run `evaluate()` from RAGAS
- Save metric scores alongside feedback logs

### Step 5: Log Feedback

- Collect user comments (rating scale 1-5)
- Persist to `rag_feedback`
- Validate against semantic similarity with ground truth

### Step 6: Prompt Enhancement

- Use Claude for suggestions on user feedback
- Store modified prompts for review and A/B testing

### Step 7: Auto-Train Classifier with Verification

- Label `rating >= 4` as 1 and `< 4` as 0
- Train model to classify `question + feedback`
- Use TF-IDF + Logistic Regression
- Verify user feedback against ground truth similarity
- Reject contradictory feedback
- Rewrite prompt using Claude if rating is bad and ground truth mismatch confirmed
- Feed improved prompt into RAG pipeline

---

## Files Generated

| File                      | Description                               |
| ------------------------- | ----------------------------------------- |
| `feedback_classifier.pkl` | Binary classification model (TF-IDF + LR) |
| `feedback_vectorizer.pkl` | Fitted TF-IDF vectorizer                  |
| `feedback_analysis.csv`   | Exported feedback logs and metrics        |

---

## Sample Prompt Improvement

**User Feedback:** "Correct answer but incomplete mechanism." **Improvement by Claude:** "Revise prompt to include all physiological pathways like absorption, insulin release, and hepatic impact."

---

## Future Roadmap

- Integrate Streamlit frontend for feedback UI
- Automate batch re-evaluation every 24h
- Use embedding clustering for failure mode discovery
- Incorporate advanced LLM-based rewriters (e.g., ReAct agents)
- Add reinforcement learning loop for adaptive prompt selection

---

## Conclusion

This RAG pipeline represents an end-to-end system that:

- Supports evaluation, feedback, learning, and optimization
- Encourages user-in-the-loop improvement
- Verifies user feedback before altering prompts or model inputs
- Leverages automation for scalable prompt enhancement and retraining

For contributions, customization, or deploying this as a service, consider modularizing each component and adding monitoring via Prometheus or Grafana.

