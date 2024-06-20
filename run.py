import os
import sys

from process import *

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 run.py <file_name or text> <embedding_model_name>")
    else:
        user_input = sys.argv[1]
        model_name = sys.argv[2]

        if os.path.isfile(user_input):
            process_file(user_input, model_name)
        else:
            process_text(user_input, model_name)

if __name__ == "__main__":
    main()