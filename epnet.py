import torch
import torch.nn as nn
import torch.nn.functional


class epnet(nn.Module):
    def __init__(self, nx, nu, ny, hp=10, hf=10, ne=32):
        super(epnet, self).__init__()
        self.nx = nx
        self.nu = nu
        self.ny = ny
        self.hp = hp
        self.hf = hf
        self.ne = ne

        self.estimator = nn.Sequential(
            nn.Linear(hp*(ny+nu), ne),
            nn.ReLU(),
            nn.Linear(ne, ne),
            nn.ReLU(),
            nn.Linear(ne, ne),
            nn.ReLU(),
            nn.Linear(ne, nx),
            nn.Identity()
        )
        self.predictor = nn.Sequential(
            nn.Linear(nx+hf*nu, ne),
            nn.ReLU(),
            nn.Linear(ne, ne),
            nn.ReLU(),
            nn.Linear(ne, ne),
            nn.ReLU(),
            nn.Linear(ne, hf*ny),
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
