import numpy as np
import argparse

def generate_split(train_split: float, val_split: float, test_split: float, n: int, save_path: str) -> None:
  data_idx = np.arange(n)
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

  generate_split(train_split=args.train_split, 
                 val_split=args.val_split, 
                 test_split=args.test_split, 
                 n=args.n, 
                 save_path='./data_storage/' + args.name + '.npz')