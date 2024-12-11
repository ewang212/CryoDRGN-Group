import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from utils import hartley_transform, ifft

def visualize_3D(grid_size, data, thresh, azim, title):
    """
    Visualizes a 3D scatter plot of the data above a specified threshold
    
    Parameters:
        grid_size (int) : The size of the 3D grid
        
        data (numpy.ndarray) : A 3D array containing the data to visualize 
        
        thresh (float) : A fraction (0.0 to 1.0) of the maximum value in data
        
        azim (float) : The azimuthal angle (in degrees) for the 3D plot view

        title (str): Title of the plot
    """
    data = data.real
    threshold = thresh * data.max()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    x, y, z = np.where(data > threshold)
    ax.scatter(x, y, z, c=data[data > threshold], cmap="gray")
    ax.view_init(elev=30, azim=azim)
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.set_zlim(0, grid_size)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(title)
    plt.savefig(f"fig_{title}.png")
    plt.show()

def visualize_2D(data, title):
    """
    Visualizes a 2D array as an image with a title and labeled axes.

    Parameters:
        data (numpy.ndarray or torch.Tensor) : A 2D array to be visualized

        title (str) : Title of the plot
    """
    plt.imshow(data)
    plt.title(title)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def plot_losses(loss1, loss2, label1, label2, save_file_path, title):
    """
    Plots two sets of loss values over training epochs

    Parameters:
        loss1 (list or numpy.ndarray) : The first set of loss values (e.g., training loss)
    
        loss2 (list or numpy.ndarray) : The second set of loss values (e.g., validation loss)
    
        label1 (str) : Label for first set of loss values 
    
        label2 (str) : Label for the second set of loss values
    
        save_file_path (str) : The file path where to save plot 
    
        title (str) : Title of the plot
    """
    epochs = list(range(1, len(loss1) + 1))
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss1, label=label1, marker='o')
    plt.plot(epochs, loss2, label=label2, marker='o')
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_file_path)


def plot_imgs(x, x_tmp, output, output_tmp, grid_size, save_file_path):
    """
    Plots and compares input and output images in a 4-panel grid and saves the plot. 
    Fourier input is compared to Fourier output, and real input is compared to real output. 

    Parameters:
        x (torch.Tensor) : Input data in Hartley space.
    
        x_tmp (numpy.ndarray or torch.Tensor) : Input projection data (real space)
    
        output (torch.Tensor) : Output data in Hartley space. 
    
        output_tmp (numpy.ndarray or torch.Tensor) : The output projection data (real space)
    
        grid_size (int) : size of the input image
    
        save_file_path (str) : Path to save the image
    """
    plt.subplot(1, 4, 1)
    plt.imshow(x_tmp)
    plt.title('Input Proj')
    plt.subplot(1, 4, 2)
    plt.imshow(x.cpu().detach().numpy()[0].reshape(grid_size, grid_size))
    plt.title('Input Hartley')
    plt.subplot(1, 4, 3)
    plt.imshow(output_tmp)
    plt.title('Output Slice')
    plt.subplot(1, 4, 4)
    plt.imshow(output.cpu().detach().numpy()[0].reshape(grid_size, grid_size))
    plt.title('Output Hartley')
    plt.savefig(f"{save_file_path}.png")


def plotting_validation(x_val, output_val, output_val_r, grid_size):
    """
    Visualizes and compares validation data and plots input 
    and output images for both correct and random rotations

    Parameters:
    x_val (torch.Tensor) : Input validation data, in Hartley space

    output_val (torch.Tensor) : The model output corresponding to the correct rotation
    This is in the Hartley space. 
    (note that a hartley transform is applied to this, effectively moving it to the real space)

    output_val_r (torch.Tensor) : The model output corresponding to a random rotation
    This is in the Hartley space. 

    grid_size (int) : size of the grid and the input images
    """
    x_tmp = hartley_transform(x_val.cpu().detach().numpy()[0].reshape(grid_size, grid_size))
    output_tmp = hartley_transform(output_val.cpu().detach().numpy()[0].reshape(grid_size, grid_size))
    plot_imgs(x_val, x_tmp, output_val, output_tmp, grid_size, "2D_Visualization_Correct_Rotation")

    output_tmp_r = hartley_transform(output_val_r.cpu().detach().numpy()[0].reshape(grid_size, grid_size))
    plot_imgs(x_val, x_tmp, output_val_r, output_tmp_r, grid_size, "2D_Visualization_Random_Rotation")

