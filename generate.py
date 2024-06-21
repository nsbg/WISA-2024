from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

GENERATION_MODEL_ID = "google/gemma-7b-it"

gemma_model     = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_ID, device_map="auto")
gemma_tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_ID, add_special_tokens=True)

def remove_sentence(text):
    lines = text.strip().split('\n')

    if lines and lines[0].startswith("Sure, "):
        lines = lines[1:]

    cleaned_text = '\n'.join(lines).strip()

    return cleaned_text

def generate_output(user_input, category=None):
    pipe = pipeline("text-generation", model=gemma_model, tokenizer=gemma_tokenizer, max_new_tokens=150)

    if category != None:
        messages = [
            {
                "role": "user",
                "content": "[Answer Content]\n- S: If you have sexual problems or questions, it is best to consult a trusted professional. Sexual health information can be obtained from medical professionals, sex education organizations, or reliable websites. It is important to respect your body and mind and to share your questions in a safe and comfortable environment.\n- SH: If you feel the urge to self-harm, do not try to handle it alone. Seek help immediately from someone you trust or a professional. Contact a close friend, family member, or mental health counseling service. Self-harm urges can be serious, so it is important to get professional help without hesitation.\n- HR: If you feel that you are being harassed or bullied, it is important to inform and seek help from someone you trust. Keep a record of the situation and report it to your company's HR team, grievance officer, or counseling center. If necessary, consider taking legal action and seek advice from professionals.\n- V: If you are in a violent situation, get to a safe place as quickly as possible and contact the police or emergency services. Ask for help from those around you and record as much information as possible to leave evidence later. Violence is never acceptable, so do not hesitate to seek help.\n- H: If you witness hate speech or hate acts, it is important to respond calmly. If the situation allows, point out that the behavior is inappropriate and support the victim. Record such incidents and report them to the authorities if necessary. Hate speech and acts are serious social issues, and everyone must work together to address them.\n\nGenerate an answer to the question based on the information for {}, without any additional commentary or preamble.".format(category)
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": "{}".format(user_input)
            }
        ]

    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
            prompt,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            top_p=0.95,
    )
    
    final_output = remove_sentence(outputs[0]["generated_text"][len(prompt):])

    return final_output