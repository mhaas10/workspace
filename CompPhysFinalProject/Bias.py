#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from numpy import random as rng
from tqdm import tqdm
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation



### These functions are copied from the Notebook
### See animation code at bottom

cmap = ListedColormap(['white', 'green'])

# Uniform Lattice creation
def normal_lattice(N, M, value=0):
    '''
    This function returns an N (rows) x M (columns) lattice with identical values value
    '''
    return np.full((N, M), value)

# Random Lattice creation
def random_lattice(N, M):
    '''
    This function returns an N (rows) x M (columns) lattice with randomized spin values 0 or 1
    '''
    return rng.choice((-1, 1), (N, M))

# plot a lattice
def plot_lattice(lattice, ax=None):
    """
    Create a visualization of a lattice 
    """
    if ax is None:
        ax = plt.gca()
    
    edgecolor = 'black' if max(lattice.shape) < 25 else None
    im = ax.pcolormesh(lattice, cmap=cmap, edgecolor=edgecolor, vmax=1, vmin=0)
    ax.set_aspect('equal')
    return im


# Let's see if we can try and exactly image this with our code.

# In[115]:

def MomoLattice(sizex,sizey):
    lattice = normal_lattice(sizex,sizey)
    for i in range(sizex):
        lattice[0][i] = 1
        lattice[i][0] = 1
        lattice[sizex-1][i] = 1
        lattice[i][sizex-1] = 1
    return lattice

def singleStepBound(lattice, B, S, locs,vals) :
    output = MomoLattice(len(lattice),len(lattice[0]))
    for i in np.arange(2,len(lattice)-2):
        for j in np.arange(2,len(lattice[i])-2):
            output[i][j] = pointChange(lattice,i,j,B,S,locs,vals)
    return output


def pointChange(lattice,i,j,BornNums,SurvNums,locs,vals):
    if lattice[i][j] == 1:
        for num in SurvNums:
            if neighborSums(lattice,i,j,locs,vals) == num:
                return 1
        return 0
    else:
        for num in BornNums:
            if neighborSums(lattice,i,j,locs,vals) == num:
                return 1
        return 0
    
def neighborSums(lattice, i , j, locs, vals):
    ip = (i + 1) % len(lattice)
    im = (i - 1) % len(lattice)
    jp = (j + 1) % len(lattice[0])
    jm = (j - 1) % len(lattice[0])
    weight = [1 for i in range(8)]
    for k in range(len(locs)):
        weight[locs[k]] = vals[k]
    
    x = (weight[0]*lattice[im][jm] + weight[1]*lattice[i][jm] + weight[2]*lattice[ip][jm] + weight[3]*lattice[im][j] + weight[4]*lattice[ip][j] + weight[5]*lattice[im][jp] + weight[6]*lattice[i][jp] + weight[7]*lattice[ip][jp])
    return x
def singleStepAnimation(lattice, B, S, locs,vals,boundary = False) :
    if boundary:
        output = MomoLattice(len(lattice),len(lattice[0]))
        for i in np.arange(2,len(lattice)-2):
            for j in np.arange(2,len(lattice[i])-2):
                output[i][j] = pointChange(lattice,i,j,B,S,locs,vals)
        lattice = output
        return lattice
    else:
        output = normal_lattice(len(lattice),len(lattice[0]))
        for i in range(len(lattice)):
            for j in np.arange(len(lattice[i])):
                output[i][j] = pointChange(lattice,i,j,B,S,locs,vals)
        lattice = output
        return lattice
def PutGlider(lattice,i,j):
    lattice[i% len(lattice)][j% len(lattice[0])] = 1
    lattice[i% len(lattice)][(j+2) % len(lattice[0])] = 1
    lattice[(i-1)% len(lattice)][(j+1)% len(lattice[0])] = 1
    lattice[(i-1)% len(lattice)][(j+2)% len(lattice[0])] = 1
    lattice[(i+1)% len(lattice)][(j+2)% len(lattice[0])] = 1
    return lattice
def GliderLat(size):
    lattice = normal_lattice(size,size)
    lattice = PutGlider(lattice, int(size/2),int(size/2))
    return lattice                   

                   
N, M = 50, 50

B=[3,4]
S=[2,3]
locs = [0,1,3]
vals = [1,1,1]

history = []
size = 201
half = int(size/2)
lattice = normal_lattice(size,size)
for i in range(10):
    lattice[half+i][half-i] = 1
    lattice[half + i][half + i ] =1
    lattice[half-i][half-i] =1 
    lattice[half-i][half+i] =1 
store_every = 1
Nsteps = size
boundary = False

def updatefig(frame):
    dataImg = history[frame]
    im.set_array(np.ravel(dataImg))

    return [im]

history = []
for t in tqdm(range(Nsteps)):
    lattice = singleStepAnimation(lattice, B, S, locs,vals,boundary) 
    if t % store_every == 0:
            history.append(lattice.copy())

history = np.asarray(history)

### Create an animation
fig, ax = plt.subplots()
im = plot_lattice(history[0], ax=ax)
anim = animation.FuncAnimation(fig, updatefig, np.arange(0, len(history)-1), repeat=True, interval=150, blit = True)
plt.show()



