import numpy as np

if __name__ == "__main__":  
  name = 'fulvene_md_250'
  save_path = './data_storage/' + name + '.npz'
  n = 250

  train_split = 0.8
  val_split = 0.1
  test_split = 0.1

  data_idxs = np.arange(225)
  np.random.shuffle(data_idxs)
  train_idxs = data_idxs[:int(train_split * n)]
  val_idxs = data_idxs[int(train_split * n):]
  test_idxs = np.arange(225, 250)

  np.savez(save_path, 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=test_idxs)