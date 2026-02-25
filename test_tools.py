import requests, json

resp = requests.post("http://localhost:11434/api/chat", json={
    "model": "qwen2.5:7b",
    "stream": False,
    "messages": [{"role": "user", "content": "Use the run_python tool to compute the mean of [1,2,3,4,5] and print it."}],
    "tools": [{"type": "function", "function": {
        "name": "run_python",
        "description": "Execute Python code.",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}
    }}]
})
print(json.dumps(resp.json()["message"], indent=2))
