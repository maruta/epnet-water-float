# Recommended to open with VS Code with the Python extension installed.
# %%
# load modules

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from epnet import epnet
from eplin import eplin
from utils import *
import time
from datetime import datetime
import copy
from mat4py import loadmat
from pathlib import Path
from scipy.io import savemat
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is NOT available')

# %%
# parameters

hp = 300
hf = 300
nxhat = 3

# %%
# load data
outpath = 'results'
Path(outpath).mkdir(parents=True, exist_ok=True)
data = loadmat('data/data-1.mat')
y = np.array(data['y'])

# In this case, there is only one output,
# but the code assumes multiple outputs,
# so it is extended to a two-dimensional
# array with single elements.
y = y[:, None]

# In this case, there is no input,
# so prepare a 2-dimensional array
# with empty elements
u = np.zeros((y.size, 0))

nu = u.shape[1]
ny = y.shape[1]
N = y.shape[0]-hp-hf

fig = plt.figure(figsize=(6, 3))
plt.plot(y)
plt.xlabel(r'$t$', fontsize=15)
plt.ylabel(r'$y$', fontsize=15)


origin = hp

Yp = torch.tensor(pHankel(y, origin, hp, N),
                  dtype=torch.float32).to('cuda')
Up = torch.tensor(pHankel(u, origin, hp, N),
                  dtype=torch.float32).to('cuda')
Yf = torch.tensor(fHankel(y, origin, hf, N),
                  dtype=torch.float32).to('cuda')
Uf = torch.tensor(fHankel(u, origin, hf, N),
                  dtype=torch.float32).to('cuda')

split_size = 0.9
train_size = int(N*split_size)

train_dataloader = DumbDataLoader(
    Up[:train_size], Yp[:train_size], Uf[:train_size], Yf[:train_size], batch_size=int(train_size/4))

val_dataloader = DumbDataLoader(
    Up[train_size:], Yp[train_size:], Uf[train_size:], Yf[train_size:], batch_size=N-train_size, shuffle=False)

# %%
# Training

model = epnet(nx=nxhat, nu=nu, ny=ny, hf=hf, hp=hp).to(device)

# The use of linear nets results in a subspace identification method based on the gradient method.
# model = eplin(nx=nxhat,nu=nu,ny=ny,hf=hf,hp=hp).to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

best_val_loss = np.Inf
writer = SummaryWriter()

for epoch in range(200000):
    running_loss = 0.0
    val_loss = 0.0
    start = time.time()

    for (up, yp, uf, yf_target) in train_dataloader:
        optimizer.zero_grad()

        yf = model(up, yp, uf)
        loss = criterion(yf, yf_target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    for (up, yp, uf, yf_target) in val_dataloader:  # validation check
        yf = model(up, yp, uf)
        loss = criterion(yf, yf_target)
        val_loss += loss.item()

    # for termination criteria
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stagnation_counter = 0
        best_model = copy.deepcopy(model.state_dict())
    else:
        stagnation_counter += 1

    writer.add_scalar('running_loss', running_loss, epoch+1)
    writer.add_scalar('validation_loss', val_loss, epoch+1)
    writer.add_scalar('best_loss', best_val_loss, epoch+1)
    writer.add_scalar('stagnation_counter', stagnation_counter, epoch+1)

    print("epoch:", epoch, " training",
          running_loss, " validation", val_loss, " best", best_val_loss, " stagnation_counter", stagnation_counter)

    # termination criteria
    if stagnation_counter >= 10000:
        break

writer.close()
model.load_state_dict(best_model)
print('Finished Training')


name = '{}/model_{} nx_{} hp_{} hf_{} ts_{}.pkl'.format(
    outpath,
    type(model).__name__,
    nxhat,
    model.hp,
    model.hf,
    format(datetime.now(), '%Y%m%d%H%M%S'))
torch.save(model, name)
