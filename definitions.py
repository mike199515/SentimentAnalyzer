import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#print(os.path.abspath(__file__))
def verify_cwd():
    assert(os.getcwd().lower() == ROOT_DIR), "cwd mismatch detected. make sure you run the script from root. current cwd is {}/Expect cwd is {}".format(os.getcwd(),ROOT_DIR)