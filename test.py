import numpy as np


data_idxs = np.load('./data_storage/fulvene_gs_250_inter.npz')['test_idx']

print(data_idxs[-2])