# %%
import torch
from epnet import epnet
from eplin import eplin
import matplotlib.pyplot as plt
from PIL import Image
from mat4py import loadmat
import numpy as np
from pathlib import Path
import time
from os import listdir
from os.path import isfile, join
from matplotlib.animation import FuncAnimation


# %%
# load model

if torch.cuda.is_available():
    device = 'cuda'
    print('GPU is available')
else:
    device = 'cpu'
    print('GPU is NOT available')


# %%
dirname = 'results'
files = [f for f in listdir(dirname) if isfile(join(dirname,f))]

outpath = 'movies'
Path(outpath).mkdir(parents=True, exist_ok=True)

for filename in files:
    model = torch.load(join(dirname,filename))
    movie_filename = join('movies',filename[:-4]+".mp4")
    if isfile(movie_filename):
        print("skip: {} exists\n".format(movie_filename))
        continue
    print("parse {}\n".format(filename))
    model.to(device).eval()

    # import data
    data = loadmat('data/data-1.mat')
    y = np.array(data['y'])
    y = y[:,None]
    u = np.zeros((y.size,0))
    Ts = 1.0/60
    t = np.arange(y.size)*Ts
    nu = u.shape[1]
    ny = y.shape[1]
    hp = model.hp
    hf = model.hf

    # check model response
    k0 = hp*2+1000
    test_length = 3000
    plt.tight_layout()

    fig, ax = plt.subplots(1)

    def update_plot(k):
        ax.clear()
        yp = torch.tensor(y[k-hp:k].flatten(), device=device, dtype=torch.float32)
        up = torch.tensor(u[k-hp:k].flatten(), device=device, dtype=torch.float32)
        xt = model.estimate(yp[None, :], up[None, :])[0]
        yf = model.predict(xt[None,:], up[None,:])[0].detach().cpu().numpy()
        ax.plot(t[k],y[k][0],'ko')
        ax.plot(t[k-hp*2:k+hp*2+1],y[k-hp*2:k+hp*2+1].flatten(),'k:',linewidth=1)
        ax.plot(t[k:k+hf],yf,linewidth=2)
        ax.plot(t[k-hp:k],yp.detach().cpu().numpy(),linewidth=2)
        ax.set_xlim(t[k-hp*2],t[k+hp*2+1])
        yl = ax.get_ylim()
        ax.set_ylim((yl[0]+yl[1])/2-0.3,(yl[0]+yl[1])/2+0.3)
        ax.axvline(t[k],0,1,color='k',linewidth=0.5)
        ax.set_xlabel('t')
        ax.set_ylabel('y(t)')

    # Create animation
    anim = FuncAnimation(fig,
                        update_plot,
                        frames=np.arange(k0, k0+test_length))
    
    # Save animation
    anim.save(movie_filename,
            dpi=150,
            fps=60,
            writer='ffmpeg')
    

