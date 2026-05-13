import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'evaluator'))

import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from metrics import run_evaluation

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="wide")
st.title("📊 RAG Evaluation Dashboard")
st.write("Evaluate your RAG pipeline quality using RAGAS metrics.")

# ============ SIDEBAR — SETTINGS ============

st.sidebar.header("⚙️ Pipeline Settings")
st.sidebar.write("Adjust these settings to see how they affect RAG quality.")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100,
    help="Number of characters per chunk. Smaller = more precise retrieval. Larger = more context per chunk."
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=400,
    value=200,
    step=50,
    help="Characters shared between adjacent chunks. Higher overlap reduces context loss at boundaries."
)

k = st.sidebar.slider(
    "Number of Chunks (k)",
    min_value=1,
    max_value=8,
    value=3,
    help="How many chunks to retrieve per question. More chunks = more context but higher cost."
)

# ============ MAIN AREA ============

tab1, tab2, tab3 = st.tabs(["🚀 Run Evaluation", "📈 Results", "🔬 Question Analysis"])

with tab1:
    st.subheader("Run Evaluation")
    st.write("Upload a PDF document and run the evaluation against the test questions.")

    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])

    if uploaded_file:
        # Save uploaded file
        pdf_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', uploaded_file.name
        )
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Uploaded: {uploaded_file.name}")

        st.write("**Current Settings:**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Chunk Size", chunk_size)
        col2.metric("Chunk Overlap", chunk_overlap)
        col3.metric("K (chunks retrieved)", k)

        if st.button("▶️ Run Evaluation", type="primary"):
            with st.spinner("Running evaluation — this takes 2 to 3 minutes..."):
                try:
                    results = run_evaluation(
                        pdf_path=pdf_path,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        k=k
                    )
                    st.session_state.results = results
                    st.success("Evaluation complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Evaluation failed: {str(e)}")
    else:
        st.info("Please upload the Tesla annual report PDF to get started.")

with tab2:
    st.subheader("Evaluation Results")

    if "results" not in st.session_state:
        # Check for saved results
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
        result_files = sorted([
            f for f in os.listdir(results_dir) if f.endswith('.json')
        ], reverse=True)

        if result_files:
            selected_file = st.selectbox("Load previous results", result_files)
            if st.button("Load"):
                with open(os.path.join(results_dir, selected_file)) as f:
                    st.session_state.results = json.load(f)
                st.rerun()
        else:
            st.info("No results yet. Run an evaluation first.")
    else:
        results = st.session_state.results
        scores = results["overall_scores"]
        settings = results["settings"]

        # Settings used
        st.write(f"**Settings:** chunk_size={settings['chunk_size']}, "
                f"chunk_overlap={settings['chunk_overlap']}, "
                f"k={settings['k']}")

        # Overall score
        overall = scores["overall"]
        color = "green" if overall >= 0.8 else "orange" if overall >= 0.6 else "red"
        st.markdown(f"### Overall Score: <span style='color:{color}'>{overall:.2f}</span>",
                   unsafe_allow_html=True)

        # Metric cards
        col1, col2, col3, col4 = st.columns(4)

        def score_color(score):
            if score >= 0.8: return "🟢"
            elif score >= 0.6: return "🟡"
            else: return "🔴"

        col1.metric(
            f"{score_color(scores['faithfulness'])} Faithfulness",
            f"{scores['faithfulness']:.2f}",
            help="Is the answer grounded in the retrieved context?"
        )
        col2.metric(
            f"{score_color(scores['answer_relevancy'])} Answer Relevancy",
            f"{scores['answer_relevancy']:.2f}",
            help="Does the answer address the question?"
        )
        col3.metric(
            f"{score_color(scores['context_precision'])} Context Precision",
            f"{scores['context_precision']:.2f}",
            help="Are the retrieved chunks relevant?"
        )
        col4.metric(
            f"{score_color(scores['context_recall'])} Context Recall",
            f"{scores['context_recall']:.2f}",
            help="Did retrieval find all important information?"
        )

        # Radar chart
        st.subheader("Metric Radar Chart")
        categories = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]
        values = [
            scores["faithfulness"],
            scores["answer_relevancy"],
            scores["context_precision"],
            scores["context_recall"]
        ]

        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line_color='royalblue'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Question-by-Question Analysis")

    if "results" not in st.session_state:
        st.info("Run an evaluation first to see per-question analysis.")
    else:
        results = st.session_state.results
        detailed = results["detailed_results"]

        # Build dataframe
        df = pd.DataFrame([{
            "Question": r["question"][:60] + "...",
            "Faithfulness": round(r["faithfulness"], 2),
            "Answer Relevancy": round(r["answer_relevancy"], 2),
            "Context Precision": round(r["context_precision"], 2),
            "Context Recall": round(r["context_recall"], 2),
        } for r in detailed])

        # Heatmap
        st.subheader("Score Heatmap")
        fig = px.imshow(
            df.set_index("Question")[["Faithfulness", "Answer Relevancy",
                                       "Context Precision", "Context Recall"]],
            color_continuous_scale="RdYlGn",
            zmin=0, zmax=1,
            aspect="auto"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Per question detail
        st.subheader("Question Detail")
        for i, r in enumerate(detailed):
            with st.expander(f"Q{i+1}: {r['question']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Generated Answer:**")
                    st.write(r["answer"])
                    st.write("**Expected Answer:**")
                    st.write(r["ground_truth"])
                with col2:
                    st.write("**Scores:**")
                    st.metric("Faithfulness", f"{r['faithfulness']:.2f}")
                    st.metric("Answer Relevancy", f"{r['answer_relevancy']:.2f}")
                    st.metric("Context Precision", f"{r['context_precision']:.2f}")
                    st.metric("Context Recall", f"{r['context_recall']:.2f}")
                st.write("**Retrieved Chunks:**")
                for j, ctx in enumerate(r["contexts"]):
                    st.text_area(f"Chunk {j+1}", ctx[:300] + "...", height=100, key=f"chunk_{i}_{j}")

if __name__ == "__main__":
    pass