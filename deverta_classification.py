import os
import sys

from tqdm import tqdm
from datetime import datetime

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline
import pandas as pd
import numpy as np

MODEL_PATH = 'checkpoint-24410'

def main(file_name):
    now = datetime.now()

    save_time = ''.join(str(now.date()).split("-"))
    
    df = pd.read_csv(file_name)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)

    outputs = []
    for prompt in tqdm(df['prompt']):
        output = pipe(prompt)
        outputs.append(output[0]['label'].split('_')[1])

    df["toxic"] = outputs

    save_dir = "./result"
    file_path = f"{save_dir}/{save_time}_deverta_results.csv"
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    df.to_csv(file_path)

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)