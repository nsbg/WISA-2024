from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, TextClassificationPipeline

GENERATION_MODEL_ID = "google/gemma-7b-it"

gemma_model     = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_ID, device_map="auto")
gemma_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_ID, add_special_tokens=True)

CLASSIFICATION_MODEL_ID = "checkpoint-24410"

deberta_model     = AutoModelForSequenceClassification.from_pretrained(CLASSIFICATION_MODEL_ID)
deberta_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def classify_prompts(inp):
    pipe = pipeline("text-generation", model=gemma_model, tokenizer=gemma_tokenizer, max_new_tokens=2)

    messages = [
        {
            "role": "user",
            "content": "Please classify the given sentence as either normal or abnormal.\n\nClassification criteria:\n-abnormal: sexual, hate, violence, harassment, self-harm etc.\n-normal: general sentences\n\nResponse format:\nnormal\nor\nabnormal\n\nSentence:\n\n{}".format(inp)
        }
    ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
    )

    return outputs[0]["generated_text"][len(prompt):]

def classify_normal_prompts(user_input):
    pipe = TextClassificationPipeline(model=deberta_model, tokenizer=deberta_tokenizer)

    output = pipe(user_input)

    final_output = output[0]['label'].split('_')[1]

    return final_output
