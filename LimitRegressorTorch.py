from dataclasses import dataclass
from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

def make_activation(name: Literal["identity","tanh","relu","gelu"]):
    if name == "identity":
        return nn.Identity()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int,...]=(64,64), act="tanh"):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), make_activation(act)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

@dataclass
class LimitRegressorConfig:
    input_dim: int
    output_dim: int
    hidden: Tuple[int,...] = (64,64)
    act_F: str = "tanh"
    phi: str = "tanh"
    alpha: float = 0.2
    eps: float = 1e-6
    k_max: int = 1000
    lr: float = 1e-3
    loss: Literal["mse"] = "mse"
    device: str = "cpu"

class LimitRegressor(nn.Module):
    def __init__(self, cfg: LimitRegressorConfig):
        super().__init__()
        self.cfg = cfg
        self.F = MLP(cfg.input_dim, cfg.output_dim, cfg.hidden, act=cfg.act_F)
        self.phi_fn = make_activation(cfg.phi)

    @torch.no_grad()
    def fixed_point(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        y = torch.zeros(B, self.cfg.output_dim, device=x.device, dtype=x.dtype)
        F_x = self.F(x)
        alpha = self.cfg.alpha
        eps = self.cfg.eps
        for _ in range(self.cfg.k_max):
            y_next = y + alpha * self.phi_fn(F_x - y)
            if torch.max(torch.norm(y_next - y, dim=1)) < eps:
                y = y_next
                break
            y = y_next
        return y

    def predict(self, x: torch.Tensor, detach: bool = True) -> torch.Tensor:
        x = x.to(self.cfg.device)
        y_star = self.fixed_point(x)
        return y_star.detach().clone() if detach else y_star

    def training_step(self, x: torch.Tensor, t: torch.Tensor):
        x = x.to(self.cfg.device)
        t = t.to(self.cfg.device)
        with torch.no_grad():
            y_star = self.fixed_point(x)
        F_x = self.F(x)
        if self.cfg.loss == "mse":
            loss = F.mse_loss(F_x, t)
        else:
            raise ValueError("Only mse loss implemented here.")
        return loss, y_star.detach(), F_x

    def fit(self, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 128, max_epochs: int = 200, delta: float = 1e-5, verbose: bool = True, val_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, patience: int = 20):
        self.to(self.cfg.device)
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        def batches(tensor, bs):
            for i in range(0, tensor.shape[0], bs):
                yield i, min(i+bs, tensor.shape[0])
        best_val = float("inf")
        no_improve = 0
        prev_epoch_loss = None
        for epoch in range(max_epochs):
            self.train()
            total = 0.0
            count = 0
            for i, j in batches(X, batch_size):
                x = X[i:j].to(self.cfg.device)
                y = Y[i:j].to(self.cfg.device)
                opt.zero_grad()
                loss, y_star, F_x = self.training_step(x, y)
                loss.backward()
                opt.step()
                total += loss.item() * (j - i)
                count += (j - i)
            epoch_loss = total / count
            if val_data is not None:
                self.eval()
                with torch.no_grad():
                    x_val, y_val = val_data
                    pred_val = self.fixed_point(x_val.to(self.cfg.device))
                    val_mse = F.mse_loss(pred_val, y_val.to(self.cfg.device)).item()
            else:
                val_mse = epoch_loss
            if verbose:
                print(f"Epoch {epoch+1:03d} | train_loss={epoch_loss:.6f} | val_mse={val_mse:.6f}")
            if prev_epoch_loss is not None and abs(epoch_loss - prev_epoch_loss) < delta:
                if verbose:
                    print("Early stop by delta.")
                break
            prev_epoch_loss = epoch_loss
            if val_data is not None:
                if val_mse + 1e-9 < best_val:
                    best_val = val_mse
                    no_improve = 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        if verbose:
                            print("Early stop by patience.")
                        self.load_state_dict(best_state)
                        break