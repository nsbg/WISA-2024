import pandas as pd

from tqdm       import tqdm
from datetime   import datetime

from utils      import *
from generate   import generate_output
from classifier import classify_prompts, classify_normal_prompts
from search     import set_model_name, create_vector_db, find_most_similar

ABNORMAL_LABELS = ["S", "H", "SH", "V", "HR"]

def process_file(file_name, model_name):
    save_dir = "./result"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    now = datetime.now()

    save_time = ''.join(str(now.date()).split("-"))

    modelname_for_save = model_name.split("/")[1]
    
    try:
        df = pd.read_csv(file_name)
    except:
        print("Please upload CSV format file.")

    set_model_name(model_name)

    index, model, metadata = create_vector_db()

    result_list = []
    
    category_1, category_2, category_3, category_4 = "", "", "", ""

    output = ""
    
    for prompt in tqdm(df["prompt"]):
        category_1 = classify_prompts(prompt)

        if category_1 != "abnormal":
            category_2 = classify_normal_prompts(prompt)

            if category_2 == "normal":
                output = generate_output(prompt)
            else:
                vector_search = find_most_similar(prompt, model, index, metadata)

                category_3 = vector_search[0]["top_1_category"]

                if category_3 == "normal":
                    output = generate_output(prompt)
                else:
                    vector_search = find_most_similar(prompt, model, index, metadata)

                    category_4 = vector_search[0]["top_1_category"]

                    output = generate_output(prompt, category_4)
        else:
            vector_search = find_most_similar(prompt, model, index, metadata)

            category_2 = vector_search[0]["top_1_category"]

            similarity_score = vector_search[0]["top_1_similarity_score"]

            if similarity_score < 0.55:
                category_3 = classify_normal_prompts(prompt)
            else:
                output = generate_output(prompt, category_2)
        
        result_list.append([prompt, category_1, category_2, category_3, category_4, output])
    
    result_df = pd.DataFrame(
        result_list,
        columns=["user_input", "category1", "category2", "category3", "category_4", "model_output"]
    )

    file_version = get_version()

    file_path = f"{save_dir}/{save_time}_{modelname_for_save}_result_v{file_version}.csv"

    result_df.to_csv(file_path, index=False)

def process_text(text, model_name):
    set_model_name(model_name)

    index, model, metadata = create_vector_db()

    classification_result = classify_prompts(text)

    # TODO: category_1 = classify_normal_prompts(text)

    if classification_result != "abnormal":
        category_1 = classify_normal_prompts(text)

        if category_1 not in ABNORMAL_LABELS:
            output = generate_output(text)
        else:
            abnormal_category = find_most_similar(text, model, index, metadata)
            output = generate_output(text, abnormal_category)
    else:
        abnormal_category = find_most_similar(text, model, index, metadata)
        output = generate_output(text, abnormal_category[0]['top_1_category'])
        
    return output
    # if normal_texts is None:
    #     print("No normal prompts found.")
    # else:
    #     for prompt in tqdm(normal_prompts['prompt']):
    #         normal_verification = classify_normal_prompts(prompt)
            
    #         if normal_verification not in ABNORMAL_LABELS:
    #             output = generate_output(prompt)
    #         else:
    #             abnormal_category = find_most_similar(prompt, model, index, metadata)
    #             output = generate_output(prompt, abnormal_category)

    #     return output
    
    # if abnormal_prompts is None:
    #     print("No abnormal prompts found.")
    # else:
    #     for prompt in tqdm(abnormal_prompts['prompt']):
    #         results = find_most_similar(prompt, model, index, metadata)
        
    #         abnormal_category = results[0]["top_1_category"]
    #         output = generate_output(prompt, abnormal_category)
        
    #     return output