import os
import torch
import torch.fft
import torch.utils.data
import torch
import torch.optim as optim
import argparse
from models import VAE, loss_function, VAE_spatial, loss_function2
from dataset import Data
from utils import load_config, save_checkpoint
import numpy as np
from plotting import plot_losses

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
train_data_loader = data.get_training_loader()
test_data_loader = data.get_validation_loader()

checkpoint_dir = config.saving.checkpoints_dir
os.makedirs(checkpoint_dir, exist_ok=True)
save_dir = config.saving.save_file_dir
if len(save_dir) > 0:
    os.makedirs(save_dir, exist_ok=True)


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
    model = VAE_spatial(config).to(device)
else:
    raise ValueError(f"Unknown model type: {config.model.model_type}")
    
optimizer = optim.Adam(model.parameters(), lr=lr)

start_epoch = 0

recon_losses = []
kl_losses = []
losses = []
train_losses = []
val_losses = []
epochs = []

# Training Loop
model.set_batch_size(batch_size)
for epoch in range(num_epochs):
    epoch += start_epoch
    epochs.append(epoch)
    epoch_loss = 0
    for images, phi in train_data_loader:
        rotations = phi.clone().detach().to(device)
        x = images.view(batch_size, x_dim).to(device)
        optimizer.zero_grad()
    
        if config.model.model_type == "CryoDRGN VAE":
            output, z, mu, logvar = model(x, rotations)
            loss = loss_function(output, x, mu, logvar, recon_losses, kl_losses, alpha_recon, beta_kl)
        elif config.model.model_type == "Spatial VAE":
            output, rotations, mu, log_var = model(x)
            loss = loss_function2(output, x, mu, log_var)
        else:
            raise ValueError(f"Not implemented yet model type: {config.model.model_type}")
    
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validation phase
    if config.model.validation:
        model.eval()
        validation_loss = 0
        validation_loss_r = 0
        with torch.no_grad():
            for images, phi in test_data_loader:
                x_val = images.view(batch_size, x_dim)
                if config.model.model_type == "CryoDRGN VAE":
                    rotations = phi.clone().detach().to(device)
                    output_val, z_val, mu_val, logvar_val = model(x_val, rotations) #compare outputs with correct rotations to random rotations
                    output_val_r, z_val_r, mu_val_r, logvar_val_r = model(x_val, torch.randn_like(rotations))
                    val_loss = loss_function(output_val, x_val, mu_val, logvar_val, [], [], alpha_recon, beta_kl)
                    val_loss_r = loss_function(output_val_r, x_val, mu_val_r, logvar_val_r, [], [], alpha_recon, beta_kl)
                    validation_loss_r += val_loss_r.item()
                elif config.model.model_type == "Spatial VAE":
                    output_val, rotations, mu, log_var  = model(x_val)
                    val_loss = loss_function(output_val, x_val, mu, log_var)
                    validation_loss += val_loss.item()
                else:
                    raise ValueError(f"Not implemented yet model type: {config.model.model_type}")
                
                validation_loss += val_loss.item()
        
        avg_val_loss = validation_loss / len(test_data_loader)
        if config.model.model_type == "CryoDRGN Vae":
            avg_val_loss_r = validation_loss_r / len(test_data_loader)
            print(f"Val loss: {avg_val_loss}, Val loss random rotation: {avg_val_loss_r}")
        else:
            print(f"Val loss: {avg_val_loss}")
        
        val_losses.append(avg_val_loss)
    else:
        avg_val_loss = None

    train_loss = epoch_loss/ len(train_data_loader)
    train_losses.append(epoch_loss/len(train_data_loader))
    
    plot_losses(val_losses, train_losses, "Validation Loss", "Training Loss", "Training_Validation_Loss", "Training vs Validation Loss")

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss}", end='')
    if config.model.validation:
        print(f", Validation Loss: {avg_val_loss}\n")
    else:
        print()

    if epoch % 10 == 0 and epoch != 0:
        if config.saving.save_checkpoints:
            save_checkpoint(model, epoch, train_loss, optimizer, f"{checkpoint_dir}/vae_{epoch}.pth", avg_val_loss)
