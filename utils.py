import numpy as np
import os
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from scipy.ndimage import affine_transform
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import zoom
import mrcfile
import matplotlib.pyplot as plt
import yaml

def load_config(config_folder, config_file, config_name):
    """
    Loads a configuration using a pipeline of configuration handlers.

    Parameters:
        config_folder (str): Path to the folder containing the configuration file(s)
        config_file (str): Name of the primary configuration file to load.
        config_name (str): Name of the configuration to retrieve within the file

    Returns:
        config (dict): The loaded configuration as a dictionary
    """
            
    pipe = ConfigPipeline(
        [
            YamlConfig(
                config_file, config_name=config_name, config_folder=config_folder
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(config_folder=config_folder),
        ]
    )
    config = pipe.read_conf()
    return config

def save_checkpoint(model, epoch, train_loss, optimizer, path, val_loss=None, kl_loss=None, recon_loss=None):
    """
    Saves a checkpoint of the model

    Parameters:
        model (torch.nn.Module): The PyTorch model to save.
        epoch (int): Current epoch number
        train_loss (float): Training loss at the current epoch
        optimizer (torch.optim.Optimizer): The optimizer whose state will be saved
        path (str): File path where the checkpoint will be saved
        val_loss (float, optional): Validation loss at the current epoch 
        kl_loss (float, optional): The KL divergence loss at the current epoch 
        recon_loss (float, optional): The reconstruction loss at the current epoch
    """
    checkpoint = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(), 
        'train_loss': train_loss,
        'val_loss': val_loss,
        'kl_loss': kl_loss,
        'recond_loss': recon_loss,
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")
    

def load_checkpoint(model, optimizer, filename):
    """
    Loads a checkpoint for the model

    Parameters:
        model (torch.nn.Module): Pytorch model to which the checkpoint will be loaded
        optimizer (torch.optim.Optimizer): optimizer associated with the model
        filename (): The path to the saved checkpoint file

    Returns:
        epoch (int): The epoch number at which the checkpoint was saved
    """
    file_size = os.path.getsize(filename) 
    if not os.path.exists(filename):
        print(f"Checkpoint file {filename} does not exist.")
    else:
        print(f"Checkpoint file {filename} found.")
    checkpoint = torch.load(filename, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def hartley_transform(object):
    """
    Applies a Hartley Transform to the input

    Parameters:
        object (numpy.ndarray): Data to be Hartley Transformed

    Returns:
        Hartley Representation of the input
    """
    object = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(object))) / np.sqrt(
        np.prod(object.shape)
    )
    object = object.real - object.imag
    return object

def fft(object):
    """
    Applies a Fourier Transform to the input

    Parameters:
        object (numpy.ndarray): Data to be Fourier Transformed

    Returns:
        Fourier Representation of the input
    """
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(object)))

def ifft(object):
    """
    Applies an inverse Fourier Transform to the input

    Parameters:
        object (numpy.ndarray): Data to be inverse Fourier Transformed

    Returns:
        Inverse Fourier Representation of the input
    """
    return np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(object)))

def generate_rand_axis(size):
    """
    Generates a random axis 

    Parameters:
        size (int): Number of elements in the random vector

    Returns:
        random_axis (numpy.ndarray): Axis of rotation
    """
    random_axis = np.random.normal(size=size)
    random_axis /= np.linalg.norm(random_axis) 
    return random_axis

def get_rotation_matrix(angle, axis):
    """
    Calculates a 3 x 3 rotation matrix given a rotation angle and axis

    Parameters:
        angle (float): angle of rotation 
        axis (numpy.ndarray): A 3D vector representing the axis of rotation.
        
    Returns:
        R (numpy.ndarray): 3 x 3 rotation matrix
    """
    angle = np.radians(angle) 
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis) 
    
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    one_minus_cos = 1 - cos_theta
    
    x, y, z = axis
    R = np.array([
        [cos_theta + x**2 * one_minus_cos, x * y * one_minus_cos - z * sin_theta, x * z * one_minus_cos + y * sin_theta],
        [y * x * one_minus_cos + z * sin_theta, cos_theta + y**2 * one_minus_cos, y * z * one_minus_cos - x * sin_theta],
        [z * x * one_minus_cos - y * sin_theta, z * y * one_minus_cos + x * sin_theta, cos_theta + z**2 * one_minus_cos]
    ])
    
    return R
            
def rotate_arbitrary_axis(fprotein, rotation_matrix, order=0, mode='constant', cval=0.0):
    """
    Rotates an input volume by a rotation matrix in the Fourier space

    Parameters: 
        fprotein (numpy.ndarray): Protein in Fourier space
        rotation_matrix (numpy.ndarray): 3 x 3 rotation matrix along random axis
        order (int): Type of interpolation, default Nearest-Neighbor
        mode (char): Denotes how points outside boundaries of the input are filled
        cval (float): Value to fill past edges of the input if mode='constant'

    Returns: 
        rotated_fprotein (): Rotated protein in Fourier space
    """
    center = np.array(fprotein.shape) / 2
    offset = center - np.dot(rotation_matrix, center)
    affine_matrix = np.eye(4)
    affine_matrix[:3, :3] = rotation_matrix
    affine_matrix[:3, 3] = offset
    rotated_fprotein = affine_transform(
        fprotein,
        matrix=affine_matrix[:3, :3], 
        offset=affine_matrix[:3, 3], 
        order=order,
        mode=mode,
        cval=cval
    )
    return rotated_fprotein


def calculate_fsc(volume1, volume2, resolution_shells=20):
    """
    Calculate the Fourier Shell Correlation (FSC) between two Fourier volumes.

    Parameters:
        volume1 (numpy.ndarray): First 3D Fourier volume.
        volume2 (numpy.ndarray): Second 3D Fourier volume.
        resolution_shells (int): Number of radial frequency shells.

    Returns:
        shell_radii (numpy.ndarray): Radii of the Fourier shells.
        fsc_values (numpy.ndarray): FSC values for each shell.
    """
    assert volume1.shape == volume2.shape, "Volumes must have the same dimensions."
    size = volume1.shape[0]
    center = size // 2

    kx, ky, kz = np.meshgrid(
        np.arange(-center, center),
        np.arange(-center, center),
        np.arange(-center, center),
        indexing="ij",
    )
    radii = np.sqrt(kx**2 + ky**2 + kz**2)

    max_radius = np.max(radii)
    shell_edges = np.linspace(0, max_radius, resolution_shells + 1)
    fsc_values = []
    shell_radii = []

    for i in range(len(shell_edges) - 1):
        mask = (radii >= shell_edges[i]) & (radii < shell_edges[i + 1])
        numerator = np.sum(volume1[mask] * np.conj(volume2[mask]))
        denominator = np.sqrt(np.sum(np.abs(volume1[mask])**2) * np.sum(np.abs(volume2[mask])**2))
        if denominator > 0:
            fsc = np.abs(numerator) / denominator
        else:
            fsc = 0
        fsc_values.append(fsc)
        shell_radii.append((shell_edges[i] + shell_edges[i + 1]) / 2)

    return np.array(shell_radii), np.array(fsc_values)

def calculate_auc_fsc(shell_radii, fsc_values):
    """
    Calculate the AUC of the FSC curve using the trapezoidal rule.

    Parameters:
        shell_radii (numpy.ndarray): Radii of the Fourier shells.
        fsc_values (numpy.ndarray): FSC values for each shell.

    Returns:
        auc_fsc (float): The area under the FSC curve.
    """
    auc_fsc = 0.0
    for i in range(1, len(shell_radii)):
        width = shell_radii[i] - shell_radii[i - 1]
        height = (fsc_values[i] + fsc_values[i - 1]) / 2
        auc_fsc += width * height
    return auc_fsc

def evaluate_fsc(vol_f, mrc_file_path, grid_size):
    """
    Plots the FSC Curve against frequency shells. 

    Parameters:
        vol_f (numpy.ndarray): 3D volume predicted by the model. 
        mrc_file_path (string): Filepath storing the ground truth 3D density
        grid_size (int): Dimension of the desired 3D grid. 
    """
    with mrcfile.open(mrc_file_path, permissive=True) as mrc:
        protein_raw = mrc.data
        zoom_factors = (grid_size / 128, grid_size / 128, grid_size / 128)
        protein_raw = zoom(protein_raw, zoom_factors, order=3)
        fprotein = fft(protein_raw)
        
    shell_radii, fsc_values = calculate_fsc(vol_f, fprotein)
    normalized_radii = shell_radii / (np.max(shell_radii))
    auc_fsc = calculate_auc_fsc(normalized_radii, fsc_values)
    
    plt.figure()
    plt.plot(range(len(fsc_values)), fsc_values, marker='o')
    plt.xlabel('Shell Index (Frequency Order)')
    plt.ylabel('FSC to Ground Truth')
    plt.title(f'FSC Curve with AUC-FSC as {auc_fsc}')
    plt.show()
    plt.savefig("FSC_evaluation.png")
    
