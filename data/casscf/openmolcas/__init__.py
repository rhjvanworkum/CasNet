MOLCAS_PATH = '/home/ruard/code/build/pymolcas'

def get_input_file(basis: str):
    return f'./data/files/input_{basis}.input'

def get_guess_orb_file(basis: str):
    return f'./data/files/orb_{basis}.orb'

def get_seward_input_file(basis: str) -> str:
    return f'./data_storage/seward_{basis}.input'