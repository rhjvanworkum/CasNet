import numpy as np

if __name__ == "__main__":  
  name = 'fulvene_gs_250_extra'
  save_path = './data_storage/' + name + '.npz'
  n = 250

  train_split = 0.8
  val_split = 0.1
  test_split = 0.1

  non_test_idxs = np.arange(n - 25)
  np.random.shuffle(non_test_idxs)
  train_idxs = non_test_idxs[:int(train_split * n)]
  val_idxs = non_test_idxs[int(train_split * n):]

  test_idxs = np.arange(n - 25, n)

  np.savez(save_path, 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=test_idxs)