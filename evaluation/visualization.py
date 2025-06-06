import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_3d_masks_marching_cubes(mask):
    verts, faces, normals, _ = measure.marching_cubes(mask, level=0.5)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    mesh = Poly3DCollection(verts[faces], alpha=0.7)
    mesh.set_facecolor([0.5, 0.5, 1])
    ax.add_collection3d(mesh)

    ax.set_xlim(0, mask.shape[2])
    ax.set_ylim(0, mask.shape[1])
    ax.set_zlim(0, mask.shape[0])
    plt.tight_layout()
    plt.show()
