import torch


def mean_squared_error(pred, targets, basis_set_size):
    pred = pred.reshape(-1, basis_set_size, basis_set_size)
    targets = targets.reshape(-1, basis_set_size, basis_set_size)
    batch_size = pred.shape[0]
    loss = 0
    for i in range(batch_size):
        loss += torch.sum(torch.square(targets[i].flatten() - pred[i].flatten())) / basis_set_size**2
    return loss / batch_size


def symm_matrix_mse(pred, targets, basis_set_size):
    pred = pred.reshape(-1, basis_set_size, basis_set_size)
    targets = targets.reshape(-1, basis_set_size, basis_set_size)

    batch_size = pred.shape[0]
    loss = 0
    for i in range(batch_size):
        H = 0.5 * (pred[i] + pred[i].T)
        loss += torch.sum(torch.square(targets[i].flatten() - H.flatten())) / len(targets[i].flatten())
    return loss / batch_size