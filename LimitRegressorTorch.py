import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Callable, Optional


class LimitRegressor(nn.Module):
    def __init__(self, base_net: nn.Module,
                 phi_type: str = "tanh", phi_c: float = 0.9,
                 alpha: float = 0.9,
                 eps: float = 1e-6, k_max: int = 100,
                 device: Optional[torch.device] = None):
        
        super().__init__()
        self.base_net = base_net
        assert phi_type in ("tanh", "linear"), "phi_type must be 'tanh' or 'linear'"
        self.phi_type = phi_type
        self.phi_c = float(phi_c)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.k_max = int(k_max)
        self.device = device if device is not None else torch.device('cpu')

    def phi(self, z: torch.Tensor) -> torch.Tensor:
        if self.phi_type == "tanh":
            return self.phi_c * torch.tanh(z)
        else:
            return self.phi_c * z

    def compute_y_star(self, x: torch.Tensor, unroll: bool = True) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.base_net(x)
        out_dim = out.shape[1:]

        y = torch.zeros((batch_size, *out_dim), device=x.device, dtype=out.dtype)

        if unroll:
            for k in range(self.k_max):
                update = self.alpha * self.phi(out - y)
                y_next = y + update
                if torch.max(torch.abs(y_next - y)) < self.eps:
                    y = y_next
                    break
                y = y_next
            return y
        else:
            with torch.no_grad():
                for k in range(self.k_max):
                    update = self.alpha * self.phi(out - y)
                    y_next = y + update
                    if torch.max(torch.abs(y_next - y)) < self.eps:
                        y = y_next
                        break
                    y = y_next
            return y.detach()

    def forward(self, x: torch.Tensor, unroll: bool = True) -> torch.Tensor:
        return self.compute_y_star(x, unroll=unroll)



def train_limit_regressor(model: LimitRegressor,
                          train_loader: DataLoader,
                          optimizer: torch.optim.Optimizer,
                          criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          epochs: int = 10,
                          mode: str = "unroll",
                          device: Optional[torch.device] = None,
                          verbose: bool = True):

    assert mode in ("unroll", "approx"), "mode must be 'unroll' or 'approx'"
    device = device if device is not None else torch.device('cpu')
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0

        for x_batch, t_batch in train_loader:
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)

            if mode == "unroll":
                # y* computed with autograd tape
                y_star = model(x_batch, unroll=True)
                loss = criterion(y_star, t_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:  
                y_star = model(x_batch, unroll=False)  
                y_for_grad = y_star.clone().detach().requires_grad_(True)
                loss_y = criterion(y_for_grad, t_batch)
                dl_dy = torch.autograd.grad(loss_y, y_for_grad, retain_graph=False)[0]
                F_out = model.base_net(x_batch)
                param_list = [p for p in model.base_net.parameters() if p.requires_grad]
                optimizer.zero_grad()
                grads = torch.autograd.grad(outputs=F_out, inputs=param_list, grad_outputs=dl_dy, retain_graph=False, allow_unused=True)
                for p, g in zip(param_list, grads):
                    if g is None:
                        p.grad = torch.zeros_like(p)
                    else:
                        p.grad = g
                optimizer.step()
                loss = criterion(y_star, t_batch)

            total_loss += float(loss.detach().cpu().item()) * x_batch.shape[0]
            count += x_batch.shape[0]

        avg_loss = total_loss / max(1, count)
        if verbose:
            print(f"Epoch {epoch}/{epochs} — avg_loss: {avg_loss:.6f}")

    return model
