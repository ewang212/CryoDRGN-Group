import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from utils import hartley_transform, fft, ifft
from plotting import visualize_3D, visualize_2D

def get_nearest_neighbor_indices(original_coords, volume_size):
    discrete_coords = np.round(original_coords).astype(int)
    discrete_coords = np.clip(discrete_coords, 0, volume_size - 1)
    return discrete_coords


def get_all_original_coords(grid_size, rotation_matrices):
    all_original_coords = []
    i = 0
    for rotation_matrix in rotation_matrices:
        grid = np.array(
            np.meshgrid(
                np.arange(-grid_size // 2, grid_size // 2),
                np.arange(-grid_size // 2, grid_size // 2),
                [0],
                indexing="ij",
            )
        ).reshape(3, -1)

        R = rotation_matrix
        original_coords = R @ grid
        original_coords += grid_size // 2
        original_coords = original_coords.transpose(1, 0)
        all_original_coords.append(original_coords)
        i += 1
    return all_original_coords


def get_volume(grid_size, rotation_matrices, slices, flag_is_fourier):
    all_original_coords = get_all_original_coords(grid_size, rotation_matrices)
    vol = np.zeros((grid_size, grid_size, grid_size), dtype=np.complex128)
    weight_volume = np.zeros((grid_size, grid_size, grid_size), dtype=np.complex128)
    for i, original_coords in enumerate(all_original_coords):
        discrete_coords = get_nearest_neighbor_indices(
            original_coords, grid_size
        ).transpose(1, 0)
        for j, (x, y, z) in enumerate(discrete_coords.transpose(1, 0)):
            slice = slices[i].reshape(grid_size, grid_size)
            if flag_is_fourier:
                slice = hartley_transform(slice)
            slice = fft(slice) / np.sqrt(np.prod(slice.shape))
            vol[x, y, z] = slice.flatten()[j]
            j += 1
            weight_volume[x, y, z] += 1

    weight_volume[weight_volume == 0] = 1
    vol_norm = vol / weight_volume
    return vol_norm
