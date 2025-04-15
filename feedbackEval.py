# RAG EVALUATION PIPELINE WITH FEEDBACK, PROMPT IMPROVEMENT & AUTO-RETRAINING

# Install needed packages
# !pip install langchain ragas trulens-eval opensearch-py datasets pandas sqlalchemy psycopg2 langchain-anthropic scikit-learn

# ---- 1. SETUP ---- #
import os
import pandas as pd
from sqlalchemy import create_engine
from langchain.chains import RetrievalQA
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from opensearchpy import OpenSearch
from datasets import Dataset
from ragas.langchain.eval_chain import RagasEvaluator
from ragas.metrics import faithfulness, context_precision, answer_relevancy, answer_correctness
from trulens_eval import TruChain, Feedback, Tru
from trulens_eval.feedback import Groundedness, Hallucination
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Claude setup
llm = ChatAnthropic(
    model="claude-3.5-sonnet",
    temperature=0.0,
    max_tokens=1024
)

# OpenSearch setup
client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_auth=("admin", "admin"),
    use_ssl=False
)

vectorstore = OpenSearchVectorSearch(
    opensearch_client=client,
    index_name="rag-docs",
    embedding_function=OpenAIEmbeddings()
)

retriever = vectorstore.as_retriever()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ---- 2. EXAMPLE DATA ---- #
data = [
    {
        "question": "What is the mechanism of action of metformin?",
        "ground_truth": "Metformin decreases hepatic glucose production and improves insulin sensitivity."
    },
    {
        "question": "What is the recommended starting dose of metformin in type 2 diabetes?",
        "ground_truth": "The usual starting dose is 500 mg once or twice daily with meals."
    }
]

user_feedbacks = [
    {
        "user_comment": "The answer is mostly correct but doesn't mention the effect on intestinal glucose absorption.",
        "user_rating": 3
    },
    {
        "user_comment": "Accurate and clearly explained!",
        "user_rating": 5
    }
]

# ---- 3. GENERATE ANSWERS & RECORD ---- #
results = []

for row, fb in zip(data, user_feedbacks):
    response = qa_chain(row["question"])
    results.append({
        "question": row["question"],
        "answer": response["result"],
        "contexts": [doc.page_content for doc in response["source_documents"]],
        "ground_truth": row["ground_truth"],
        "user_comment": fb["user_comment"],
        "user_rating": fb["user_rating"]
    })

# ---- 4. EVALUATE WITH RAGAS ---- #
df = pd.DataFrame(results)
dataset = Dataset.from_pandas(df)
evaluator = RagasEvaluator(
    metrics=[faithfulness(), context_precision(), answer_relevancy(), answer_correctness()],
    llm=llm
)
scores = evaluator.evaluate(dataset)
print("\nRAGAS Scores:\n", scores)

# ---- 5. STORE FEEDBACK IN POSTGRESQL ---- #
db_uri = "postgresql+psycopg2://username:password@localhost:5432/feedback_db"
engine = create_engine(db_uri)
df.to_sql("rag_feedback", con=engine, if_exists="append", index=False)

# ---- 6. TRULENS: LOG AND LAUNCH DASHBOARD ---- #
tru = Tru()
ground = Groundedness(groundedness_provider=llm)
hallucination = Hallucination(groundedness_provider=llm)

tru_chain = TruChain(
    chain=qa_chain,
    app_id="rag_claude_chain_eval",
    feedbacks=[
        Feedback(ground.groundedness_measure).on_input_output(),
        Feedback(hallucination.hallucination_check).on_input_output()
    ]
)

with tru_chain:
    for row in results:
        qa_chain(row["question"])

# ---- 7. AUTO PROMPT TUNING SUGGESTIONS ---- #
def suggest_prompt_improvements(feedback_text):
    improvement_prompt = f"Suggest a way to improve the prompt based on this user feedback: '{feedback_text}'"
    response = llm.invoke(improvement_prompt)
    return response.content

low_rated = df[df["user_rating"] < 4]

print("\nPrompt Improvement Suggestions:")
for idx, row in low_rated.iterrows():
    suggestion = suggest_prompt_improvements(row["user_comment"])
    print(f"\nQ: {row['question']}\nFeedback: {row['user_comment']}\nSuggestion: {suggestion}")

# ---- 8. AUTO-RETRAIN PROMPT QUALITY PREDICTOR ---- #
feedback_df = pd.read_sql("SELECT question, user_comment, user_rating FROM rag_feedback", con=engine)

# Label good (>=4) vs bad (<4) feedback
feedback_df["label"] = feedback_df["user_rating"].apply(lambda x: 1 if x >= 4 else 0)

# Train model to predict feedback quality based on question + user_comment
feedback_df["text"] = feedback_df["question"] + " " + feedback_df["user_comment"]

vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(feedback_df["text"])
y = feedback_df["label"]

model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer for real-time use
joblib.dump(model, "feedback_classifier.pkl")
joblib.dump(vectorizer, "feedback_vectorizer.pkl")

print("\nAuto-retraining complete: classifier saved to feedback_classifier.pkl")
