import requests

def compute_ppl_via_server(text: str, server_url: str = "http://localhost:8000/ppl") -> float:
    payload = {"text": text}

    resp = requests.post(server_url, json=payload)

    if resp.status_code == 200:
        return resp.json().get("ppl", 0.0)
    else:
        return 0.0
    
print(compute_ppl_via_server("This is a test sentence."))