import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def verify_cwd():
    assert(os.getcwd()==ROOT_DIR), "cwd mismatch detected. make sure you run the script from root. current cwd is {}".format(os.getcwd())