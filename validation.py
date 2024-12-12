from plotting import visualize_3D, plotting_validation
import torch
import torch.fft
import torch.utils.data
import torch
import torch.optim as optim
import argparse
from models import VAE, loss_function
from dataset import Data
from utils import load_config, load_checkpoint, ifft, evaluate_fsc
import numpy as np
from d3_reconstruction import get_volume

# seed for reproducibility
np.random.seed(20)
torch.manual_seed(20)


parser = argparse.ArgumentParser(description="Process a filename as an input argument.")
parser.add_argument(
    "--config_file",      
    type=str,           
    required=False,        
    default="config.yaml", 
    help="Experiment file with parameter values."
)

args = parser.parse_args()

config = load_config(config_folder="config", config_file=args.config_file, config_name="default")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = Data(args.config_file)
test_data_loader = data.get_validation_loader()

checkpoint_dir = config.saving.checkpoints_dir

grid_size = config.data.D
x_dim = grid_size ** 2
hidden_dim = config.model.hidden_dim
z_dim = config.model.z_dim
n_encoder_layers = config.model.n_encoder_layers
n_decoder_layers = config.model.n_decoder_layers
batch_size = config.data.batch_size
lr = config.model.lr
num_epochs = config.model.epochs
alpha_recon = config.model.alpha_recon
beta_kl = config.model.beta_kl
d_model = config.model.d_model

if config.model.model_type == "CryoDRGN VAE":
    model = VAE(x_dim, hidden_dim, z_dim, batch_size, grid_size, n_encoder_layers, n_decoder_layers, positional_encoding=config.model.positional_encoding).to(device)
elif config.model.model_type == "Spatial VAE":
    raise ValueError(f"Validation not Implemented for : {config.model.model_type} due to Poor Training Performance")
else:
    raise ValueError(f"Unknown model type: {config.model.model_type}")
    
optimizer = optim.Adam(model.parameters(), lr=lr)

checkpoint_path = f"{checkpoint_dir}/{config.saving.checkpoint_filename}"
print(checkpoint_path)
start_epoch = load_checkpoint(model, optimizer, checkpoint_path)


model.eval()
validation_loss = 0
validation_loss_r = 0
all_rotations = []
all_slices = []
all_slices_r = []
with torch.no_grad():
    for images, phi in test_data_loader:
        rotations = phi.clone().detach().to(device)
        x_val = images.view(batch_size, x_dim)
        output_val, z_val, mu_val, logvar_val = model(x_val, rotations)
        output_val_r, z_val_r, mu_val_r, logvar_val_r = model(x_val, torch.randn_like(rotations))

        if config.plotting.plot_validation:
            plotting_validation(x_val, output_val, output_val_r, grid_size)

        if config.model.model_type == "CryoDRGN VAE":
            val_loss = loss_function(output_val, x_val, mu_val, logvar_val, [], [], alpha_recon, beta_kl)
            val_loss_r = loss_function(output_val_r, x_val, mu_val_r, logvar_val_r, [], [], alpha_recon, beta_kl)
        for rotation in rotations:
            all_rotations.append(rotation.cpu().detach().numpy())
        for output in output_val:
            all_slices.append(output.cpu().detach().numpy())
        for output in output_val_r:
            all_slices_r.append(output.cpu().detach().numpy())
        
        validation_loss += val_loss.item()
        validation_loss_r += val_loss_r.item()

    avg_val_loss = validation_loss / len(test_data_loader)
    avg_val_loss_r = validation_loss_r / len(test_data_loader)
    print(f"Val loss: {avg_val_loss}, Val loss r: {avg_val_loss_r}")


#Generate 3D reconstruction for outputs with correct rotations
#as well as outputs with random rotations. 

vol_f = get_volume(grid_size, all_rotations, all_slices, True)
vol = ifft(vol_f)
visualize_3D(grid_size, vol, 0.4, 130, "correct_000.mrc")
np.save("volume_right.npy", vol)


vol_f_r = get_volume(grid_size, all_rotations, all_slices_r, True)
vol_r = ifft(vol_f_r)

visualize_3D(grid_size, vol_r, 0.4, 130, "random_000.mrc")
np.save("volume_random.npy", vol_r)

#Evaluate Fourier Shell Correlation 
mrc_file_path = config.data.mrc_file_path
evaluate_fsc(vol_f, mrc_file_path, grid_size)

if config.model.model_type == "CryoDRGN VAE":
    print("Images and Data Generated. Please check for Validation")

