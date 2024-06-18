import os
import sys
import pandas as pd

from tqdm import tqdm
from datetime import datetime

from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font

from classifier import classify_prompts
from search import set_model_name, create_vector_db, find_most_similar

def main(file_name, model_name):
    now = datetime.now()

    save_time = ''.join(str(now.date()).split("-"))
    
    df = pd.read_csv(file_name)

    set_model_name(model_name)

    modelname_for_save = model_name.split("/")[1]

    abnormal_prompts = classify_prompts(df)

    if not abnormal_prompts.empty:
        results_list = []

        index, model, metadata = create_vector_db()

        for prompt in tqdm(abnormal_prompts['prompt']):
            results = find_most_similar(prompt, model, index, metadata)

            combined_result = {
                'input_sentence': prompt, 
                'human_category': abnormal_prompts.loc[abnormal_prompts['prompt'] == prompt, 'category'].values[0]
                }
            
            for result in results:
                combined_result.update(result)

            results_list.append(combined_result)

        save_dir = "./result"
        file_path = f"{save_dir}/{save_time}_{modelname_for_save}_results.xlsx"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        results_df = pd.DataFrame(results_list)

        with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
            results_df.to_excel(writer, index=False, sheet_name="Results")

            worksheet = writer.sheets["Results"]

            header_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
            header_font = Font(bold=True)

            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font

        print("Results saved to results.xlsx")
    else:
        print("No abnormal prompts found.")

if __name__ == "__main__":
    file_name = sys.argv[1]
    embed_model_name = sys.argv[2]

    main(file_name, embed_model_name)