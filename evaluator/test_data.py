# Test dataset for RAG evaluation
# Each entry has a question and the expected correct answer (ground truth)
# We use Tesla's annual report as our test document

TEST_QUESTIONS = [
    {
        "question": "What was Tesla's total revenue in 2023?",
        "ground_truth": "Total revenues in 2023 were $96,773 million."
    },
    {
        "question": "How many vehicles did Tesla deliver in 2023?",
        "ground_truth": "Tesla delivered 1,808,581 vehicles in 2023."
    },
    {
        "question": "What was Tesla's net income in 2023?",
        "ground_truth": "Net income attributable to common stockholders was $14,974 million in 2023."
    },
    {
        "question": "What are Tesla's main business segments?",
        "ground_truth": "Tesla's main business segments are automotive and energy generation and storage."
    },
    {
        "question": "What was Tesla's gross profit margin in 2023?",
        "ground_truth": "The gross margin for energy generation and storage in 2023 was 18.9%."
    },
    {
        "question": "What were Tesla's total operating expenses in 2023?",
        "ground_truth": "Tesla's total operating expenses in 2023 were approximately $8,769 million."
    },
    {
        "question": "What risks does Tesla identify related to competition?",
        "ground_truth": "Tesla identifies risks including increasing competition from established automotive manufacturers and new entrants in the electric vehicle market."
    },
    {
        "question": "What is Tesla's strategy for energy products?",
        "ground_truth": "Tesla's energy generation and storage business includes Powerwall, Powerpack, and Megapack products."
    }
]