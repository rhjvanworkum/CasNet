from ase.db import connect
import torch
import schnetpack.properties as properties
from model.caschnet_so3_model import create_so3_orbital_model
import numpy as np


if __name__ == "__main__":

    with connect('./data_storage/ethene_geom_scan.db') as conn:
        H = conn.get(1).data['F']
        atomic_numbers = conn.get(1)["numbers"]
        positions = conn.get(1)["positions"]

    batch = {
        properties.Z: torch.from_numpy(atomic_numbers[np.newaxis, :]).long(),
        properties.position: torch.from_numpy(positions[np.newaxis, :]).float(),
        'F': torch.from_numpy(H[np.newaxis, :]).float(),
    }

    label = batch['F'] # .copy()

    loss_fn = torch.nn.functional.mse_loss
    basis_set_size = 14
    cutoff = 5.0
    lr = 1e-3

    model = create_so3_orbital_model(loss_function=loss_fn, lr=lr, output_property_key=property, basis_set_size=basis_set_size, cutoff=cutoff)
    model = model.to(torch.float)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # print(model.parameters())
    for param in model.parameters():
        print(param)

    for epoch in range(100):  # loop over the dataset multiple times
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch)
        loss = loss_fn(outputs['F'], label)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss = loss.item()
        print(f'[{epoch + 1}] loss: {running_loss / 2000:.3f}')
        running_loss = 0.0

print('Finished Training')