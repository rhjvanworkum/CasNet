import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from ema import ExponentialMovingAverage
from modules import activation_factory


def str_to_bool(value):
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")


class MultiFeatureEmbedding(nn.Module):
    def __init__(self, max_features, dim, n_ao, activation, n_layers=2, n_features=5):
        super().__init__()
        self.dim = dim
        self.n_ao = n_ao
        self.activation = activation
        self.n_layers = n_layers
        self.n_features = n_features

        self.embeddings = nn.ModuleList(
            [nn.Embedding(max_features, dim) for _ in range(n_features)]
        )

        def sum_reduce(x):
            return x.sum(dim=-1)

        self.reduction = sum_reduce

    def forward(self, x):
        output = torch.empty(
            x.shape[0],
            x.shape[1],
            self.dim,
            self.n_features,
            dtype=torch.float32,
            device=x.device,
        )
        for i, emb in enumerate(self.embeddings):
            output[..., i] = emb(x[..., i])
        return self.reduction(output)


class HamiltonianNet(nn.Module):
    def __init__(
        self, dim, activation, n_ao, n_layers,
    ):
        super().__init__()
        self.activation = activation_factory(activation)
        self.n_layers = n_layers
        self.n_ao = n_ao

        modules = list()

        modules.append(
            MLPMixer(
                num_patches=n_ao,
                num_features=dim,
                num_layers=n_layers,
                num_classes=dim,
                activation=activation,
                mean_reduce=False,
            )
        )
        self.net = nn.Sequential(*modules)
        self.pred_net = nn.Sequential(self.activation(), nn.Linear(dim, n_ao))
        self.add_net = nn.Sequential(self.activation(), nn.Linear(dim, 1))

    def forward(self, reps):
        diag_correction = self.add_net(reps).squeeze(-1)
        reps = self.net(reps)
        output = self.pred_net(reps)
        output = (output.transpose(1, 2) + output) / 2.0
        output = output + torch.diag_embed(diag_correction)

        return output

class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout, activation):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)
        if isinstance(activation, str):
            self.act = activation_factory(activation)()
        else:
            self.act = activation()

    def forward(self, x):
        x = self.dropout1(self.act(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(
        self, num_features, num_patches, expansion_factor, dropout, activation
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor, dropout, activation)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(
        self, num_features, num_patches, expansion_factor, dropout, activation
    ):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout, activation)

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x = self.norm(x)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(
        self, num_features, num_patches, expansion_factor, dropout, activation
    ):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, num_features, expansion_factor, dropout, activation
        )
        self.channel_mixer = ChannelMixer(
            num_patches, num_features, expansion_factor, dropout, activation
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class MLPMixer(nn.Module):
    def __init__(
        self,
        num_patches,
        num_features=128,
        expansion_factor=2,
        num_layers=8,
        num_classes=13,
        dropout=0.0,
        activation=F.gelu,
        mean_reduce=False,
    ):
        super().__init__()
        # per-patch fully-connected is equivalent to strided conv2d
        self.mean_reduce = mean_reduce
        self.mixers = nn.Sequential(
            *[
                MixerLayer(
                    num_patches, num_features, expansion_factor, dropout, activation
                )
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(x)
        # embedding.shape == (batch_size, num_patches, num_features)
        if self.mean_reduce:
            embedding = embedding.mean(dim=1)
        logits = self.classifier(embedding)
        return logits


class OrbitalMixer(pl.LightningModule):
    def __init__(
        self,
        dim: int = 1024,
        activation: str = "GELU",
        n_layers: int = 6,
        n_overlap_layers: int = 2,
        learning_rate: float = 3e-4,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        step_lr: int = 300000,
        gamma_lr: float = 0.8,
        n_ao: int = 72,
        n_atoms: float = 9,
        num_occ: int = 13,
        expand_factor: int = 2,
        weight_decay: float = 0.0,
        run_non_ema_val: bool = True,
        max_features: int = 25,
        **kwargs,
    ):
        super().__init__()
        params = [
            k for k, v in locals().items() if isinstance(v, (int, str, float, bool))
        ] + [k for k, v in kwargs.items() if isinstance(v, (int, str, float, bool))]
        self.save_hyperparameters(*params)

        self.overlap_embed = torch.nn.Sequential(
            torch.nn.Linear(n_ao, expand_factor * dim),
            activation_factory(activation)(),
            torch.nn.Linear(expand_factor * dim, dim),
            MLPMixer(
                num_patches=n_ao,
                num_features=dim,
                num_layers=n_overlap_layers,
                num_classes=dim,
                activation=activation,
                mean_reduce=False,
            ),
        )
        self.orbital_embed = MultiFeatureEmbedding(
            max_features, dim, n_ao=n_ao, activation=activation
        )
        self.mo_energy_coeff_net = HamiltonianNet(
            dim=dim, activation=activation, n_ao=n_ao, n_layers=n_layers,
        )

    def forward(
        self, orbitals, s1es,
    ):
        orbital_embedding = self.overlap_embed(s1es)
        orbital_embedding = orbital_embedding + self.orbital_embed(orbitals)
        hamiltonian = self.mo_energy_coeff_net(orbital_embedding)

        return hamiltonian

    @staticmethod
    def get_mo_coeffs_energies_from_hamiltonian(hamiltonian, overlap):
        evs, evecs = torch.linalg.eigh(overlap)
        S_half = evecs.bmm(torch.diag_embed(evs ** (-0.5))).bmm(evecs.transpose(1, 2))
        F = S_half.transpose(1, 2).bmm(hamiltonian).bmm(S_half)
        mo_energies, mo_coeffs = torch.linalg.eigh(F)
        mo_coeffs = S_half.bmm(mo_coeffs)
        return mo_coeffs, mo_energies

    def calc_loss(
        self, orbitals, s1es, true_hamiltonian,
    ):
        pred_hamiltonian = self(orbitals, s1es)
        ham_loss = F.mse_loss(pred_hamiltonian, true_hamiltonian)
        loss_dict = {
            "hamiltonian_loss": ham_loss if self.training else ham_loss.detach(),
            "total_loss": ham_loss if self.training else ham_loss.detach(),
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.calc_loss(*batch)
        for loss_name, value in loss_dict.items():
            self.log(f"train_{loss_name}", value, on_step=True)
        return loss_dict["total_loss"]

    def validation_step(self, batch, batch_idx):
        if self.hparams.use_ema:
            with self.ema.average_parameters():
                loss_dict = self.calc_loss(*batch)
                for loss_name, value in loss_dict.items():
                    self.log(f"val_{loss_name}_EMA", value, on_step=True)
            if self.hparams.run_non_ema_val:
                loss_dict = self.calc_loss(*batch)
                for loss_name, value in loss_dict.items():
                    self.log(f"val_{loss_name}", value, on_step=True)
        else:
            loss_dict = self.calc_loss(*batch)
            for loss_name, value in loss_dict.items():
                self.log(f"val_{loss_name}", value, on_step=True)

    def configure_optimizers(self):
        if self.hparams.use_ema:
            self.ema = ExponentialMovingAverage(
                self.parameters(), decay=self.hparams.ema_decay
            )
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = dict()
        lr_scheduler["scheduler"] = torch.optim.lr_scheduler.StepLR(
            opt, step_size=self.hparams.step_lr, gamma=self.hparams.gamma_lr
        )
        lr_scheduler["interval"] = "step"

        return [opt], lr_scheduler

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.use_ema:
            self.ema.update(self.parameters())

    def on_save_checkpoint(self, checkpoint):
        if hasattr(self, "ema"):
            checkpoint["ema"] = self.ema.state_dict()
            return checkpoint

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint.keys():
            self.ema = ExponentialMovingAverage(
                self.parameters(), self.hparams.ema_decay
            )
            self.ema.load_state_dict(checkpoint["ema"])

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("OrbitalMixer")
        parser.add_argument("--dim", type=int, default=1024)
        parser.add_argument("--activation", type=str, default="GELU")

        parser.add_argument("--n_layers", type=int, default=6)
        parser.add_argument("--n_overlap_layers", type=int, default=2)
        parser.add_argument("--expand_factor", type=float, default=2)

        parser.add_argument("--learning_rate", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=0.0)
        parser.add_argument("--step_lr", type=int, default=300000)
        parser.add_argument("--gamma_lr", type=float, default=0.8)

        parser.add_argument("--use_ema", type=str_to_bool, default=True)
        parser.add_argument("--ema_decay", type=float, default=0.999)
        parser.add_argument("--run_non_ema_val", type=str_to_bool, default=True)

        parser.add_argument("--max_features", type=int, default=25)

        return parent_parser