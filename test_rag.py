from query_data import query_rag
from langchain_community.llms.ollama import Ollama

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response?
"""

def test_fmea_risk_priority():
    assert query_and_validate(
        question="How is the Risk Priority Number (RPN) calculated in FMEA?",
        expected_response="RPN = Severity × Occurrence × Detection"
    )

def query_and_validate(question: str, expected_response: str):
    actual = query_rag(question)
    prompt = EVAL_PROMPT.format(expected_response=expected_response, actual_response=actual)

    model = Ollama(model="mistral")
    result = model.invoke(prompt).strip().lower()

    print(prompt)
    if "true" in result:
        print("\033[92m✔ VALIDATED\033[0m")
        return True
    elif "false" in result:
        print("\033[91m✘ INVALID\033[0m")
        return False
    else:
        raise ValueError("Invalid response from evaluator.")
