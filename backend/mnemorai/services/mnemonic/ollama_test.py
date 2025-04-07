# https://github.com/ollama/ollama-python

from ollama import ChatResponse, chat

response: ChatResponse = chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": "Why is the sky blue?",
        },
    ],
)
print(response["message"]["content"])
# or access fields directly from the response object
print(response.message.content)
