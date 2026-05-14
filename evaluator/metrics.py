import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from rag_pipeline import build_rag_pipeline, get_answer_and_contexts
from test_data import TEST_QUESTIONS

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def run_evaluation(pdf_path: str, chunk_size: int = 1000,
                   chunk_overlap: int = 200, k: int = 3,
                   pipeline_type: str = "Naive RAG") -> dict:

    print(f"Building {pipeline_type} pipeline...")
    print(f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, k={k}")

    # Configure RAGAS
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))

    # Build pipeline based on type
    if pipeline_type == "Advanced RAG":
        import sys
        advanced_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'advanced-rag', 'retrieval'
        )
        sys.path.insert(0, advanced_path)
        from advanced_pipeline import build_advanced_pipeline

        run_query = build_advanced_pipeline(pdf_path, chunk_size, chunk_overlap)

        def get_answer_fn(question):
            answer, chunks = run_query(question)
            contexts = [chunk.page_content for chunk in chunks]
            return answer, contexts

    else:
        chain, retriever = build_rag_pipeline(
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            k=k
        )

        def get_answer_fn(question):
            return get_answer_and_contexts(chain, retriever, question)

    print(f"Running {len(TEST_QUESTIONS)} test questions...")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"  Question {i+1}/{len(TEST_QUESTIONS)}: {item['question'][:50]}...")

        answer, context = get_answer_fn(item["question"])

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

    print("Running RAGAS evaluation...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    results_df = results.to_pandas()

    overall_scores = {
        "faithfulness": float(results_df["faithfulness"].mean()),
        "answer_relevancy": float(results_df["answer_relevancy"].mean()),
        "context_precision": float(results_df["context_precision"].mean()),
        "context_recall": float(results_df["context_recall"].mean()),
        "overall": float(results_df[["faithfulness", "answer_relevancy",
                                      "context_precision", "context_recall"]].mean().mean())
    }

    detailed_results = []
    for i, row in results_df.iterrows():
        detailed_results.append({
            "question": questions[i],
            "answer": answers[i],
            "ground_truth": ground_truths[i],
            "faithfulness": float(row["faithfulness"]),
            "answer_relevancy": float(row["answer_relevancy"]),
            "context_precision": float(row["context_precision"]),
            "context_recall": float(row["context_recall"]),
            "contexts": contexts[i]
        })

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_label = "advanced" if pipeline_type == "Advanced RAG" else "naive"
    results_file = os.path.join(
        base_dir, "results",
        f"evaluation_{timestamp}_{pipeline_label}_cs{chunk_size}_k{k}.json"
    )

    output = {
        "timestamp": timestamp,
        "pipeline_type": pipeline_type,
        "settings": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "k": k
        },
        "overall_scores": overall_scores,
        "detailed_results": detailed_results
    }

    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Results saved to {results_file}")

    return output