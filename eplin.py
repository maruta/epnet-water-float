import torch
import torch.nn as nn
import torch.nn.functional


class eplin(nn.Module):
    def __init__(self, nx, nu, ny, hp=10, hf=10):
        super(eplin, self).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.hp = hp
        self.hf = hf

        self.estimator = nn.Sequential(
            nn.Linear(hp*(ny+nu), nx),
            nn.Identity()
        )
        self.predictor = nn.Sequential(
            nn.Linear(nx+hf*nu,hf*ny),
            nn.Identity()
        )

    def forward(self, up, yp, uf):
        input_vector = torch.cat((yp, up), 1)
        x = self.estimator(input_vector)
        xuf = torch.cat((x, uf), 1)
        yf = self.predictor(xuf)
        return yf

    def estimate(self, yp, up):
        input_vector = torch.cat((yp, up), 1)
        return self.estimator(input_vector)

    def predict(self, x, uf):
        xuf = torch.cat((x, uf), 1)
        return self.predictor(xuf)
