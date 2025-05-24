# LLama3 (llamacpp) Chat + Dat1 (OpenAI-compatible)

This is an example of how to create a chatbot using an LLM ([full list of supported models](https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#text-only)) and deploy it to Dat1. 
It uses the [llamacpp](https://github.com/ggerganov/llama.cpp) server to run the model.
It provides an OpenAI-compatible API for the model (create chat completions endpoint).
You can also use multimodal models like [qwen2.5-vl](https://github.com/QwenLM/Qwen2.5-VL) by providing an `mmproj.gguf` file with additional model weights.

## Prerequisites

- [Python 3.6+](https://www.python.org/downloads/)
- [Dat1 Account](https://dat1.co)
- [Dat1 CLI](https://github.com/dat1-co/dat1-cli) installed and logged in

## Deploying the example

1. Download the model weights (in `.gguf` format) and place it in the `model.gguf` file.

2. Download the multimodal model weights and place it in the `mmproj.gguf` file.

3. Deploy the model to Dat1:

```bash
dat1 deploy
```

## Using the model

You can use the model by sending a POST request to the Dat1 endpoint:

```bash
curl --request POST \
  --url https://api.dat1.co/api/v1/inference/llama-cpp-chat/invoke-stream \
  --header 'Content-Type: application/json' \
  --header 'X-API-Key: <your api key>' \
  --data '{"messages": [{"role": "user", "content": "Say this is a test!"}], "temperature": 0.7, "stream": true, "max_tokens": 100 }'
```

The model will stream back the chat completions as they are generated using Server-Sent Events.
