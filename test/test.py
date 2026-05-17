import requests
import json

url = "http://localhost:2000/v1/chat/completions"

context = "我叫dino-y"
question = "你好，你是谁"

payload = {
    "model": "/root/autodl-tmp/model/Qwen3-0.6B",
    "messages": [
        {"role": "user", "content": "介绍下ai infra,请具体点回答"}
        # {"role": "system", "content": "根据文档回答问题"},
        # {"role": "user", "content": f"文档：{context}\n问题：{question}"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": True
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                content = obj["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
            except:
                continue