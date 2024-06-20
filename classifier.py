from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline, TextClassificationPipeline

def classify_prompts(df):
    MODEL_ID = "google/gemma-7b-it"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_special_tokens=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2)

    output_list = []

    print(f"================= {MODEL_ID} 기반 추론 시작 =================")

    for inp in tqdm(df["prompt"]):
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

        output_list.append(outputs[0]["generated_text"][len(prompt):])

    print(f"================= {MODEL_ID} 기반 추론 종료 =================")

    # Append the classification results to the dataframe
    df["classification"] = output_list

    normal_prompts = df[df["classification"].str.strip().str.lower() != "abnormal"]     # Filter out the normal prompts
    abnormal_prompts = df[df["classification"].str.strip().str.lower() == "abnormal"]   # Filter out the abnormal prompts

    if len(normal_prompts) == 0:
        normal_prompts = None
    
    if len(abnormal_prompts) == 0:
        abnormal_prompts = None

    return normal_prompts, abnormal_prompts

def classify_normal_prompts(user_input):
    MODEL_ID = "checkpoint"

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    output = pipe(user_input)

    final_output = output[0]['label'].split('_')[1]

    return final_output