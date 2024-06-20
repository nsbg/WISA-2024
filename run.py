import os
import sys

from process import *

def main():
    user_input = sys.argv[1]
    model_name = sys.argv[2]

    if os.path.isfile(user_input):
        process_file(user_input, model_name)
    else:
        process_text(user_input, model_name)

if __name__ == "__main__":
    main()