"""Smoke test for the running vLLM decision-model (d3) endpoint at /v1/systemone."""
import json
import time
import requests

ENDPOINT = "http://localhost:8000/v1/systemone"

payload = {
    "model": "local-active",
    "state": (
        "TOPIC: coronavirus origin\n\n"
        "QUESTION: where did COVID-19 originate?\n\n"
        "DOCUMENT TITLE: Origin of SARS-CoV-2\n\n"
        "DOCUMENT: SARS-CoV-2 was first identified in Wuhan, China in late 2019."
    ),
    "questions": {
        "noul": {
            "type": "noul",
            "instructions": "Does this document provide relevant information to answer the question?"
        },
        "score": {
            "type": "score",
            "instructions": "Rate how relevant this document is to the question.",
            "criteria": [
                "0: Not relevant",
                "1: Partially relevant",
                "2: Fully relevant"
            ]
        }
    }
}

if __name__ == "__main__":
    print(f"Connecting to {ENDPOINT}...")
    t0 = time.time()
    resp = requests.post(ENDPOINT, json=payload, timeout=30)
    dur = (time.time() - t0) * 1000

    print(f"Status: {resp.status_code} in {dur:.1f}ms")
    if resp.status_code == 200:
        data = resp.json()
        print("\nDecision Model Response:")
        print(json.dumps(data["answers"], indent=2))
        print("\nVerification: SUCCESS!")
    else:
        print("Error:", resp.text)
