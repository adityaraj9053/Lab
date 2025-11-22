import numpy as np
import math, random
import matplotlib.pyplot as plt
from PIL import Image

# This will loadscrambled_lena.mat
def load_scrambled_lena(path):
    with open(path, 'r', encoding='latin1') as f:
        lines = [ln.rstrip('\n') for ln in f]
    # find dims line
    for i, ln in enumerate(lines):
        if ln.strip().startswith('# ndims:'):
            dims_line = lines[i+1].strip()
            break
    dims = list(map(int, dims_line.split()))
    data_tokens = []
    for ln in lines[i+2:]:
        ln_strip = ln.strip()
        if ln_strip == '' or ln_strip.startswith('#'):
            continue
        data_tokens.extend(ln_strip.split())
    arr = np.array(list(map(int, data_tokens)), dtype=np.uint8)
    arr = arr[:dims[0]*dims[1]]
    img_gray = arr.reshape((dims[0], dims[1]))
    img_color = np.stack([img_gray]*3, axis=-1)  # convert to 3-channel
    return img_color

# Jigsaw Puzzle
class JigsawAgent:
    def __init__(self, image, grid_size=4):
        self.image = image
        self.grid_size = grid_size
        self.h, self.w, _ = image.shape
        self.tile_h = self.h // grid_size
        self.tile_w = self.w // grid_size

        # split into tiles
        self.tiles = []
        for r in range(grid_size):
            for c in range(grid_size):
                tile = image[r*self.tile_h:(r+1)*self.tile_h,
                             c*self.tile_w:(c+1)*self.tile_w].copy()
                self.tiles.append(tile)
        self.n_tiles = len(self.tiles)

        # precompute borders
        self.left_cols = np.array([t[:,0,:].ravel() for t in self.tiles])
        self.right_cols = np.array([t[:,-1,:].ravel() for t in self.tiles])
        self.top_rows = np.array([t[0,:,:].ravel() for t in self.tiles])
        self.bottom_rows = np.array([t[-1,:,:].ravel() for t in self.tiles])

        # precompute pairwise seam costs
        self.cost_horiz = np.zeros((self.n_tiles, self.n_tiles), dtype=np.int64)
        self.cost_vert = np.zeros((self.n_tiles, self.n_tiles), dtype=np.int64)
        for a in range(self.n_tiles):
            dh = self.right_cols[a] - self.left_cols
            self.cost_horiz[a,:] = np.sum(dh*dh, axis=1)
            dv = self.bottom_rows[a] - self.top_rows
            self.cost_vert[a,:] = np.sum(dv*dv, axis=1)

    def pos_index(self, r, c):
        return r*self.grid_size + c

    def energy(self, perm):
        e = 0
        g = self.grid_size
        for r in range(g):
            for c in range(g-1):
                p = self.pos_index(r,c)
                q = self.pos_index(r,c+1)
                e += self.cost_horiz[perm[p], perm[q]]
        for r in range(g-1):
            for c in range(g):
                p = self.pos_index(r,c)
                q = self.pos_index(r+1,c)
                e += self.cost_vert[perm[p], perm[q]]
        return e

    def anneal(self, max_iters=50000, T0=1e5, T_end=1e-2):
        perm = np.arange(self.n_tiles)
        np.random.shuffle(perm)
        current, current_e = perm, self.energy(perm)
        best, best_e = current.copy(), current_e

        def T_sched(t):
            return T0 * (T_end/T0)**(t/max_iters)

        for t in range(max_iters):
            T = T_sched(t)
            a, b = np.random.randint(0, self.n_tiles, size=2)
            if a == b: continue
            neighbor = current.copy()
            neighbor[a], neighbor[b] = neighbor[b], neighbor[a]
            e2 = self.energy(neighbor)
            dE = e2 - current_e
            if dE <= 0 or random.random() < math.exp(-dE/(T+1e-12)):
                current, current_e = neighbor, e2
                if current_e < best_e:
                    best, best_e = current.copy(), current_e

            if (t+1) % (max_iters//5) == 0:
                print(f"iter {t+1}, T={T:.1e}, currentE={current_e:.2e}, bestE={best_e:.2e}")

        return best, best_e

    def reconstruct(self, perm):
        g = self.grid_size
        recon = np.zeros_like(self.image)
        for idx, tile_id in enumerate(perm):
            r, c = divmod(idx, g)
            recon[r*self.tile_h:(r+1)*self.tile_h,
                  c*self.tile_w:(c+1)*self.tile_w] = self.tiles[tile_id]
        return recon

img = load_scrambled_lena("scrambled_lena.mat")
agent = JigsawAgent(img, grid_size=4)
best_perm, best_energy = agent.anneal()
result = agent.reconstruct(best_perm)

plt.imshow(result)
plt.axis('off')
plt.show()
