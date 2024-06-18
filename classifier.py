from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_ID = "google/gemma-7b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, add_special_tokens=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2)

def classify_prompts(df):
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

    # Filter out the abnormal prompts
    abnormal_prompts = df[df["classification"].str.strip().str.lower() == "abnormal"]

    return abnormal_prompts