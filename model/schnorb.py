import math
import numpy as np
import torch
import torch.nn as nn
import schnetpack as spk
from schnorb import SchNOrbProperties

class SchNOrbProperties(Properties):
    ham_prop = 'hamiltonian'
    hamorth_prop = 'hamiltonian_orth'
    ov_trans_prop = 'ov_transform'
    ov_prop = 'overlap'
    en_prop = 'energy'
    f_prop = 'forces'
    psi_prop = 'psi'
    eps_prop = 'eps'


class SingleAtomHamiltonian(nn.Module):

    def __init__(self, orbital_energies, trainable=False):
        super(SingleAtomHamiltonian, self).__init__()

        if trainable:
            self.orbital_energies = nn.Parameter(
                torch.FloatTensor(orbital_energies))
        else:
            self.register_buffer('orbital_energies',
                                 torch.FloatTensor(orbital_energies))

    def forward(self, numbers, basis):
        tmp1 = (basis[:, None, :, 2] > 0).expand(-1, numbers.shape[1], -1)
        tmp2 = numbers[..., None].expand(-1, -1, basis.shape[-2])
        orb_mask = torch.gather(tmp1, 0, tmp2)
        h0 = self.orbital_energies[numbers]
        h0 = torch.masked_select(h0, orb_mask).reshape(numbers.shape[0], 1, -1)
        h0 = h0.expand(-1, h0.shape[2], -1)
        diag = torch.eye(h0.shape[1], device=h0.device)
        h0 = h0 * diag[None]
        return h0

class Hamiltonian(nn.Module):

    def __init__(self, basis_definition, n_cosine_basis, lmax, directions,
                 orbital_energies=None, return_forces=False,
                 quambo=False, create_graph=False,
                 mean=None, stddev=None, max_z=30):
        super(Hamiltonian, self).__init__()
        if return_forces:
            self.derivative = 'forces'
        else:
            self.derivative = None

        self.create_graph = create_graph

        if orbital_energies is None:
            self.h0 = None
        else:
            self.h0 = SingleAtomHamiltonian(orbital_energies, True)
            self.s0 = SingleAtomHamiltonian(np.ones_like(orbital_energies),
                                            True)

        self.register_buffer('basis_definition',
                             torch.LongTensor(basis_definition))
        self.n_types = self.basis_definition.shape[0]
        self.n_orbs = self.basis_definition.shape[1]
        self.n_cosine_basis = n_cosine_basis
        self.quambo = quambo

        directions = directions if directions is not None else 3
        self.offsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)
        self.onsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)

        self.ov_offsitenet = spk.nn.Dense(
            n_cosine_basis * directions * (2 * lmax + 1),
            self.n_orbs ** 2)

        if self.quambo:
            self.ov_onsitenet = spk.nn.Dense(
                n_cosine_basis * directions * (2 * lmax + 1), self.n_orbs ** 2)
        else:
            self.ov_onsitenet = nn.Embedding(max_z, self.n_orbs ** 2,
                                             padding_idx=0)
            self.ov_onsitenet.weight.data = torch.diag_embed(
                torch.ones(max_z, self.n_orbs)
            ).reshape(max_z, self.n_orbs ** 2)
            self.ov_onsitenet.weight.data.zero_()
        self.pairagg = spk.nn.Aggregate(axis=2, mean=True)

        self.atom_net = nn.Sequential(
            spk.nn.Dense(n_cosine_basis, n_cosine_basis // 2,
                         activation=spk.nn.activations.shifted_softplus),
            spk.nn.Dense(n_cosine_basis // 2, 1),
            spk.nn.base.ScaleShift(mean, stddev)
        )
        self.atomagg = spk.nn.Aggregate(axis=1, mean=False)

    def forward(self, inputs):
        Z = inputs['_atomic_numbers']
        nbh = inputs[SchNOrbProperties.neighbors]
        # nbhmask = inputs[Properties.neighbor_mask]
        x0, x, Vijkl = inputs['representation']

        # Vijkl shape: batch, max_atoms, max_nbh, max_lr, feats

        batch = Vijkl.shape[0]
        max_atoms = Vijkl.shape[1]

        orb_mask_i = self.basis_definition[:, :, 2] > 0
        orb_mask_i = orb_mask_i[Z].float()
        orb_mask_i = orb_mask_i.reshape(batch, -1, 1)
        orb_mask_j = orb_mask_i.reshape(batch, 1, -1)
        orb_mask = orb_mask_i * orb_mask_j

        ar = torch.arange(max_atoms, device=nbh.device)[None, :, None].expand(
            nbh.shape[0], -1, 1)
        _, nbh = torch.cat([nbh, ar], dim=2).sort(dim=2)

        Vijkl = Vijkl.reshape(Vijkl.shape[:3] + (-1,))

        H_off = self.offsitenet(Vijkl)
        zeros = torch.zeros((batch, max_atoms, 1, self.n_orbs ** 2),
                            device=H_off.device,
                            dtype=H_off.dtype)
        H_off = torch.cat([H_off, zeros], dim=2)
        H_off = torch.gather(H_off, 2, nbh[..., None].expand(-1, -1, -1,
                                                             self.n_orbs ** 2))

        H_on = self.onsitenet(Vijkl)
        H_on = self.pairagg(H_on)
        id = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[
            None, ..., None]
        H_on = id * H_on[:, :, None]

        H = H_off + H_on

        H = H.reshape(batch, max_atoms, max_atoms, self.n_orbs,
                      self.n_orbs).permute((0, 1, 3, 2, 4))
        H = H.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        H = 0.5 * (H + H.permute((0, 2, 1)))

        # mask padded orbitals
        H = torch.masked_select(H, orb_mask > 0)
        orbs = int(math.sqrt(H.shape[0] / batch))
        H = H.reshape(batch, orbs, orbs)

        if self.h0 is not None:
            H = H + self.h0(Z, self.basis_definition)

        del zeros

        # overlap
        S_off = self.ov_offsitenet(Vijkl)
        zeros = torch.zeros((batch, max_atoms, 1, self.n_orbs ** 2),
                            device=S_off.device,
                            dtype=S_off.dtype)
        S_off = torch.cat([S_off, zeros], dim=2)
        S_off = torch.gather(S_off, 2, nbh[..., None].expand(-1, -1, -1,
                                                             self.n_orbs ** 2))
        del zeros
        if self.quambo:
            S_on = self.ov_onsitenet(Vijkl)
            S_on = self.pairagg(S_on)
        else:
            S_on = self.ov_onsitenet(Z)
        id = torch.eye(max_atoms, device=H_on.device, dtype=H_on.dtype)[
            None, ..., None]
        S_on = id * S_on[:, :, None]

        S = S_off + S_on

        S = S.reshape(batch, max_atoms, max_atoms, self.n_orbs,
                      self.n_orbs).permute((0, 1, 3, 2, 4))
        S = S.reshape(batch, max_atoms * self.n_orbs, max_atoms * self.n_orbs)

        # symmetrize
        S = 0.5 * (S + S.permute((0, 2, 1)))

        # mask padded orbitals
        S = torch.masked_select(S, orb_mask > 0)
        orbs = int(math.sqrt(S.shape[0] / batch))
        S = S.reshape(batch, orbs, orbs)

        if self.s0 is not None:
            S = S + self.s0(Z, self.basis_definition)

        # total energy
        Ei = self.atom_net(x)
        E = self.atomagg(Ei)

        if self.derivative is not None:
            F = -torch.autograd.grad(E, inputs[SchNOrbProperties.R],
                                     grad_outputs=torch.ones_like(E),
                                     create_graph=self.create_graph)[0]
        else:
            F = None

        inputs['F'] = H
        inputs['S'] = S
        inputs['energy'] = E
        inputs['forces'] = F
        return inputs