import numpy as np
import argparse

def generate_split(train_split: float, val_split: float, test_split: float, n: int, save_path: str, data_idx: bool) -> None:
  np.random.shuffle(data_idx)
  train_idxs = data_idx[:int(train_split * n)]
  val_idxs = data_idx[int(train_split * n):int((train_split + val_split) * n)]
  test_idxs = data_idx[int((train_split + val_split) * n):]

  np.savez(save_path, 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=test_idxs)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--name', type=str)
  parser.add_argument('--n', type=int)
  parser.add_argument('--train_split', type=float)
  parser.add_argument('--val_split', type=float)
  parser.add_argument('--test_split', type=float)
  args = parser.parse_args()
  
  name = 'split_10k'
  save_path = './data_storage/' + name + '.npz'
  
  train_split = 0.8
  val_split = 0.2
  
  tot_size = 9995
  test_size = 50
  
  n = 9945
  data_idx = np.random.choice(np.arange(tot_size - test_size), n)
  
  np.random.shuffle(data_idx)
  # test_idxs = data_idx[:test_size]
  # non_test_idxs = data_idx[test_size:]
  train_idxs = data_idx[:int(train_split * len(data_idx))]
  val_idxs = data_idx[int(train_split * len(data_idx)):]

  np.savez(save_path, 
    train_idx=train_idxs, 
    val_idx=val_idxs,
    test_idx=np.arange(tot_size)[-50:])
  
  
  # if sample:
  #   n = 1200
  #   data_idx = np.random.choice(np.arange(args.n), n)
  # else:
  #   n = args.n
  #   data_idx = np.arange(args.n)

  # generate_split(train_split=args.train_split, 
  #                val_split=args.val_split, 
  #                test_split=args.test_split, 
  #                n=n, 
  #                save_path='./data_storage/' + args.name + '.npz',
  #                data_idx=data_idx)