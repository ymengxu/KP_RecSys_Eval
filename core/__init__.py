import os


WORKING_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MAIN_DIRECTORY = os.path.dirname(WORKING_DIRECTORY)
ROOT_DIRECTORY = os.path.dirname(MAIN_DIRECTORY)

DATA_RAW_PATH = os.path.join(MAIN_DIRECTORY, 'data_raw')
DATA_CLEAN_PATH = os.path.join(MAIN_DIRECTORY, 'data_clean')
SAVEMODEL_PATH = os.path.join(MAIN_DIRECTORY, 'savemodel')
RES_PATH = os.path.join(MAIN_DIRECTORY, 'res')

os.makedirs(DATA_RAW_PATH, exist_ok=True)
os.makedirs(DATA_CLEAN_PATH, exist_ok=True)
os.makedirs(SAVEMODEL_PATH, exist_ok=True)
os.makedirs(RES_PATH, exist_ok=True)

SEED = 2023

TEST_K = [5,10,20]