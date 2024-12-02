# World Conference on Information Security Applications (WISA) 2024

## Abstract
This paper proposes an LLM guardrail framework that incorporates a Zero Trust architecture to validate and control the responses of Large Language Model (LLM) to unethical queries. The proposed framework applies guardrails to harmful inputs to avoid harmful responses and includes four verification steps through Policy Decision Point (PDP) and Policy Enforcement Point (PEP) structures. This structure aims to enhance the reliability and safety of LLM responses. We demonstrate this framework on a fixed model and verify its generality by applying it to various models. Consequently, this allows for evasive responses to a wide range of unethical prompts.

## Proposed Algorithm
![image](image\flowchart.png)
## Usage
### Single GPU
#### 1. File input
```
CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM python3 run.py YOUR_FILE_NAME YOUR_EMBEDDING_MODEL_ID
```
#### 2. Text input
```
CUDA_VISIBLE_DEVICES=YOUR_GPU_NUM python3 run.py "YOUR_INPUT_TEXT" YOUR_EMBEDDING_MODEL_ID
```
The result will be printed to your terminal like below screenshot
![image](https://github.com/nsbg/WISA-2024/assets/53206051/7ed37a2e-2176-4c35-ab93-e2f83bc5c137)
