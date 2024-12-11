import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def quaternion_to_rotation_matrix(q):
    rotation = R.from_quat(q.detach().cpu().numpy())  
    return rotation.as_matrix()
    
def ffwd(dim=16, expand=2) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(dim, expand * dim, bias=False), 
        nn.ReLU(), 
        nn.Linear(expand * dim, dim, bias=False), 
    ).to(device)

def ffwd2(dim=16, expand=2) -> nn.Sequential:
    # block
    return nn.Sequential(
        nn.Linear(dim, expand * dim, bias=True).to(device),
        nn.ReLU(),
        nn.Linear(expand * dim, dim, bias=True).to(device),
        nn.LayerNorm(dim)
    ).to(device)


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs, include_input=True, log_sampling=True, input_dim=3):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.input_dim = input_dim

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0, num_freqs - 1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0, 2. ** (num_freqs - 1), steps=num_freqs)
        self.register_buffer('freq_bands_buffer', self.freq_bands)

    def forward(self, coords):
        freq_bands = self.freq_bands_buffer.to(coords.device)
        coords_expanded = coords.unsqueeze(-1)  
        freq_bands = freq_bands.view(*([1] * (coords.dim() - 1)), 1, -1)
        scaled_coords = coords_expanded * freq_bands
        sin_enc = torch.sin(scaled_coords)
        cos_enc = torch.cos(scaled_coords)
        enc = torch.cat([sin_enc, cos_enc], dim=-1) 
        enc = enc.view(*coords.shape[:-1], self.input_dim * 2 * self.num_freqs)

        if self.include_input:
            enc = torch.cat([coords, enc], dim=-1) 
        return enc


class VAE(nn.Module):
    def __init__(self, x_dim=16384, hidden_dim=256, z_dim=384, batch_size = 50, grid_size=128, n_encoder_layers=2, n_decoder_layers=2, positional_encoding=False):
        super(VAE, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.x_dim = x_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.grid_size = grid_size
        self.positional_encoding = positional_encoding
        
        if self.z_dim > self.grid_size ** 2: 
            raise ValueError(f"z_dim ({self.z_dim}) cannot be greater than D*D ({grid_size * grid_size}).")

        # Encoder
        self.enc_l1 = nn.Linear(self.x_dim, self.hidden_dim)
        self.enc_layers = nn.ModuleList([
            ffwd(self.hidden_dim, expand=2) for _ in range(n_encoder_layers)
        ])
        self.enc_final_layer_mu = nn.Linear(self.hidden_dim, self.z_dim)
        self.enc_final_layer_log_var = nn.Linear(self.hidden_dim, self.z_dim, bias = False)
        self.enc_linear = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)   
    
        # Decoder
        if self.positional_encoding:
            self.dec_l1 = nn.Linear(self.z_dim+63, self.hidden_dim).to(self.device) 
        else:
            self.dec_l1 = nn.Linear(self.z_dim+3, self.hidden_dim).to(self.device) 

        self.dec_layers = nn.ModuleList([
            ffwd(self.hidden_dim, expand=2) for _ in range(n_decoder_layers)
        ])
        self.dec_final_layer = nn.Linear(self.hidden_dim, 1).to(self.device)

        self.positional_encoding_layer = PositionalEncoding(
            num_freqs=10,
            include_input=True,
            log_sampling=True,
            input_dim=3
        )
    
    
    def get_grid(self):
        x, y = np.meshgrid(
                np.linspace(-1, 1, self.grid_size, endpoint=True),
                np.linspace(-1, 1, self.grid_size, endpoint=True),
            )
        coords = np.stack([x.ravel(), y.ravel(), np.zeros(self.grid_size**2)], 1).astype(np.float32)
        coords = torch.from_numpy(coords).unsqueeze(0).repeat(self.batch_size, 1, 1)
        return coords.to(device)
        
    
    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    
    def encoder(self, x):
        x = x.to(self.device)
        x = torch.relu(self.enc_l1(x))
        x = self.enc_linear(x)
        for layer in self.enc_layers:
            x = x + layer(x)

        mu = self.enc_final_layer_mu(x)
        log_var = self.enc_final_layer_log_var(x)
       
        return mu.to(self.device), log_var.to(self.device)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z


    def decoder(self, z):
        z = z.to(self.device)
        batch_size, num_subsamples, z_dim = z.shape
        x = torch.relu(self.dec_l1(z))
        for layer in self.dec_layers:
            x = x + layer(x) 
        x = self.dec_final_layer(x)
        x = x.view(batch_size, num_subsamples)

        return x, z


    def forward(self, x, rotations, eval=False):
        coords = self.get_grid()
        rotations = rotations.to(self.device)
        coords = torch.matmul(coords, rotations.float())

        mu, log_var = self.encoder(x)

        z = self.reparameterize(mu, log_var)
        z = z.unsqueeze(1).expand(-1, 2500, -1)
        z = 2 * (z - torch.min(z)) / (torch.max(z) - torch.min(z)) - 1

        if self.positional_encoding:
            coords = self.positional_encoding_layer(coords)
            
        z = torch.concat((z, coords), dim=-1)
        y, z = self.decoder(z)
        return y.to(self.device), z.to(self.device), mu.to(self.device), log_var.to(self.device) 


class VAE_spatial(nn.Module):
    def __init__(self, config):
        super(VAE_spatial, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.D = config.data.D
        self.grid_size_hor = self.D
        self.grid_size_vert = self.D
        self.batch_size = config.data.batch_size
        self.rotation_latent_size = 4

        self.x_dim = self.D * self.D
        self.hidden_dim = config.model.hidden_dim
        self.z_dim = config.model.z_dim
        self.spatial = config.model.spatial
        self.pose_mlp = config.model.pose_mlp
        if self.spatial:
            self.z_dim += self.rotation_latent_size

        self.n_layers = 1

        # Encoder
        if self.spatial: 
            self.enc_l1 = nn.Linear(self.x_dim, self.hidden_dim + self.rotation_latent_size, bias=False).to(self.device)
            self.enc_layers = nn.ModuleList(
                [ffwd2(self.hidden_dim + self.rotation_latent_size , expand=4) for _ in range(self.n_layers)]
            )
            self.enc_final_layer_mu = nn.Linear(self.hidden_dim + 4, self.z_dim).to(self.device)
            self.enc_final_layer_log_var = nn.Linear(self.hidden_dim + 4, self.z_dim, bias=True).to(self.device)
        else:
            self.enc_l1 = nn.Linear(self.x_dim, self.hidden_dim, bias=False).to(self.device)
            self.enc_layers = nn.ModuleList(
                [ffwd2(self.hidden_dim, expand=4) for _ in range(self.n_layers)]
            )
            self.enc_final_layer_mu = nn.Linear(self.hidden_dim, self.z_dim).to(self.device)
            self.enc_final_layer_log_var = nn.Linear(self.hidden_dim, self.z_dim, bias=True).to(self.device)
            

        # Project coords
        self.proj_coords = nn.Linear(63, self.hidden_dim).to(self.device)   # self.hidden_dim
        self.proj_layers = nn.ModuleList([ 
            ffwd2(dim=self.hidden_dim, expand=4) for _ in range(self.n_layers)
        ])

        # Decoder
        if self.spatial:
            self.dec_l1 = nn.Linear(self.z_dim - self.rotation_latent_size, self.hidden_dim, bias=False).to(self.device)
            self.dec_layers = nn.ModuleList([ffwd2(self.hidden_dim, expand=4) for _ in range(self.n_layers)])
            self.dec_final_layer = nn.Linear(self.hidden_dim, 1, bias=True).to(self.device)
        else:
            self.dec_l1 = nn.Linear(self.z_dim, self.hidden_dim, bias=False).to(self.device)
            self.dec_layers = nn.ModuleList([ffwd2(self.hidden_dim, expand=4) for _ in range(self.n_layers)])
            self.dec_final_layer = nn.Linear(self.hidden_dim, 1, bias=True).to(self.device)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
                num_freqs=10,
                include_input=True,
                log_sampling=True,
                input_dim=3
            )

        self.rotation_mlp = nn.Sequential(
            nn.Linear(self.z_dim, self.hidden_dim),  
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 4) 
            ).to(self.device)
            

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def get_grid(self):
        x, y = np.meshgrid(
                np.linspace(-1, 1, self.D, endpoint=True),
                np.linspace(-1, 1, self.D, endpoint=True),
            )
        coords = np.stack([x.ravel(), y.ravel(), np.zeros(self.D * self.D)], 1).astype(np.float32)
        coords = torch.from_numpy(coords).unsqueeze(0).repeat(self.batch_size, 1, 1)
        return coords.to(self.device) # B x D^2 x 3
 

    def decoder(self, z_cat):
        z_cat = z_cat.to(self.device)
        batch_size, num_subsamples, z_cat_dim = z_cat.shape
        y = self.dec_l1(z_cat)
        for layer in self.dec_layers:
            y = y + layer(y)
        y = self.dec_final_layer(y)
        y = y.view(batch_size, num_subsamples)
        
        return y

    
    def encoder(self, x):
        x = x.to(self.device)
        z = self.enc_l1(x).to(self.device)
        for layer in self.enc_layers:
            z = z + layer(z)
        mu = self.enc_final_layer_mu(z)
        log_var = self.enc_final_layer_log_var(z)
        return mu.to(self.device), log_var.to(self.device)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)  # Equivalent to torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(std)
        z = eps * std + mu
        return z

    def forward(self, x, eval=False):

        x = x.to(self.device)

        # ENCODER
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)


        if self.spatial: 
            if self.pose_mlp: 
                latent_q = self.rotation_mlp(z)
            else: 
                latent_q = z[:, :self.rotation_latent_size]
                 
            rotation_from_quaternion = torch.tensor(quaternion_to_rotation_matrix(latent_q)).to(self.device)
     
            z = z[:, self.rotation_latent_size:]  
            self.z_dim = self.z_dim - self.rotation_latent_size
        
        
        z = z.unsqueeze(1).expand(-1, self.D* self.D, -1)

        
        x_coords = self.get_grid()

        if self.spatial: 
            rotations = rotation_from_quaternion.to(self.device)
            x_coords = torch.matmul(x_coords, rotations.float())
        else:
             x_coords = torch.matmul(x_coords, rotations.float())
            
        
        x_coords = self.positional_encoding(x_coords)
        x_coords = self.proj_coords(x_coords)
        for layer in self.proj_layers:
            x_coords = x_coords + layer(x_coords)

        # DECODER
        z_cat = z + x_coords
        y = self.decoder(z_cat)

        return y.to(self.device), rotation_from_quaternion, mu, log_var 


def loss_function(output, x, mu, logvar, recon_losses=[], kl_losses=[], alpha_recon=1, beta_kl=.1):
    output = output.to(device)
    x = x.to(device)
    mu = mu.to(device)
    logvar = logvar.to(device)
    recon_loss = F.mse_loss(output, x, reduction='mean') 
    z_kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    recon_losses.append(recon_loss.item())
    kl_losses.append(z_kl_loss.item())
    return alpha_recon * recon_loss +  beta_kl * z_kl_loss 


def loss_function2(output, x, mu, logvar, alpha_recon=1, beta_kl=0.1):
    output = output.to(device)
    x = x.to(device)
    mu = mu.to(device)
    logvar = logvar.to(device)

    recon_loss = F.mse_loss(output, x, reduction='mean') 

    z_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0) 

    if torch.isnan(z_kl_loss):
        raise RuntimeError("KL Loss is NaN!!")
   
    return alpha_recon * recon_loss +  beta_kl * z_kl_loss
