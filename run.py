import os
import sys

from process import *

def main(filename_or_text, model_name):
    if os.path.isfile(filename_or_text):
        process_file(filename_or_text, model_name)
    else:
        print(process_text(filename_or_text, model_name))

if __name__ == "__main__":
    user_input = sys.argv[1]
    embed_model = sys.argv[2]

    main(user_input, embed_model)