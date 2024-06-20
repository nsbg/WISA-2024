import pandas as pd

from tqdm import tqdm
from datetime import datetime

from utils import *
from generate import generate_output
from classifier import classify_prompts, classify_normal_prompts
from search import set_model_name, create_vector_db, find_most_similar

ABNORMAL_LABELS = ["S", "H", "SH", "V", "HR"]

def process_file(file_name, model_name):
    now = datetime.now()

    save_time = ''.join(str(now.date()).split("-"))
    
    try:
        df = pd.read_csv(file_name)
    except:
        print("Please upload CSV format file.")

    set_model_name(model_name)

    index, model, metadata = create_vector_db()

    normal_prompts, abnormal_prompts = classify_prompts(df)

    # Verify normal prompts
    if not normal_prompts.empty:
        for prompt in tqdm(normal_prompts['prompt']):
            normal_verification = classify_normal_prompts(prompt)

            if normal_verification not in ABNORMAL_LABELS:
                normal_output = generate_output(prompt)
            else:
                abnormal_category = find_most_similar(prompt, model, index, metadata)
                abnormal_output = generate_output(prompt, abnormal_category)
    else:
        print("No normal prompts found.")

    # Process abnormal prompts
    if not abnormal_prompts.empty:
        abnormal_lists = []

        for prompt in tqdm(abnormal_prompts['prompt']):
            results = find_most_similar(prompt, model, index, metadata)

            combined_result = {
                'input_sentence': prompt, 
                'human_category': abnormal_prompts.loc[abnormal_prompts['prompt'] == prompt, 'category'].values[0]
            }
            
            for result in results:
                combined_result.update(result)

            abnormal_lists.append(combined_result)

        save_dir = "./result"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        modelname_for_save = model_name.split("/")[1]

        file_version = get_version()

        file_path = f"{save_dir}/{save_time}_{modelname_for_save}_result_v{file_version}.xlsx"

        results_df = pd.DataFrame(abnormal_lists)

        apply_excel_style(results_df, file_path)
    else:
        print("No abnormal prompts found.")

def process_text(text, model_name):
    set_model_name(model_name)

    index, model, metadata = create_vector_db()

    df = pd.DataFrame({'prompt': [text]})

    normal_prompts, abnormal_prompts = classify_prompts(df)

    if normal_prompts is None:
        print("No normal prompts found.")
    else:
        for prompt in tqdm(normal_prompts['prompt']):
            normal_verification = classify_normal_prompts(prompt)
            
            if normal_verification not in ABNORMAL_LABELS:
                output = generate_output(prompt)
            else:
                abnormal_category = find_most_similar(prompt, model, index, metadata)
                output = generate_output(prompt, abnormal_category)

        return output
    
    if abnormal_prompts is None:
        print("No abnormal prompts found.")
    else:
        for prompt in tqdm(abnormal_prompts['prompt']):
            results = find_most_similar(prompt, model, index, metadata)
        
            abnormal_category = results[0]["top_1_category"]
            output = generate_output(prompt, abnormal_category)
        
        return output