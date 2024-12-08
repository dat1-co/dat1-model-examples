# Kolors + Dat1

This is an example of how to deploy the [Kolors Stable Diffusion model](https://huggingface.co/Kwai-Kolors/Kolors) to Dat1 Platform.

## Prerequisites

- [Python 3.6+](https://www.python.org/downloads/)
- [Dat1 Account](https://dat1.co)
- [Dat1 CLI](https://github.com/dat1-co/dat1-cli) installed and logged in

## Deploying the example

1. Download the model weights using the Hugging Face CLI:

```bash
huggingface-cli download --resume-download Kwai-Kolors/Kolors --local-dir weights/Kolors
```

or Git LFS:
```bash
git lfs clone https://huggingface.co/Kwai-Kolors/Kolors weights/Kolors
```

2. Deploy the model to Dat1:

```bash
dat1 deploy
```

## Using the model

You can use the model by sending a POST request to the Dat1 endpoint:

```bash
curl --request POST 
  --url https://api.dat1.co/api/v1/inference/kolors/invoke 
  --header 'Content-Type: application/json' 
  --header 'X-API-Key: <your api key>' 
  --data '{
	"input": {
		"prompt": "A funny cat"
	}
}'
```
