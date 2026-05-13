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
                   chunk_overlap: int = 200, k: int = 3) -> dict:
    """
    Run full RAGAS evaluation on a RAG pipeline.
    Returns scores and detailed results for each question.
    """

    print(f"Building RAG pipeline...")
    print(f"Settings: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, k={k}")

    # Configure RAGAS with explicit LLM and embeddings
    ragas_llm = LangchainLLMWrapper(ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key
    ))
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key))

    # Build the pipeline
    chain, retriever = build_rag_pipeline(
        pdf_path=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=k
    )

    print(f"Running {len(TEST_QUESTIONS)} test questions...")

    # Collect results for each question
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(TEST_QUESTIONS):
        print(f"  Question {i+1}/{len(TEST_QUESTIONS)}: {item['question'][:50]}...")

        answer, context = get_answer_and_contexts(
            chain, retriever, item["question"]
        )

        questions.append(item["question"])
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(item["ground_truth"])

    print("Running RAGAS evaluation...")

    # Create RAGAS dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # Run evaluation
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

    # Convert to dataframe
    results_df = results.to_pandas()

    # Calculate overall scores
    overall_scores = {
        "faithfulness": float(results_df["faithfulness"].mean()),
        "answer_relevancy": float(results_df["answer_relevancy"].mean()),
        "context_precision": float(results_df["context_precision"].mean()),
        "context_recall": float(results_df["context_recall"].mean()),
        "overall": float(results_df[["faithfulness", "answer_relevancy",
                                      "context_precision", "context_recall"]].mean().mean())
    }

    # Build detailed results
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

    # Save results
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_file = os.path.join(base_dir, "results",
                                f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_cs{chunk_size}_k{k}.json")

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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