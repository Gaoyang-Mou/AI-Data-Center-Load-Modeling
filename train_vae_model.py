import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import logging
import sys
import os
import time

# --- Configuration ---
INPUT_FEATHER_FILE = 'merged_data_cell_a_pdu6_pdu7_approx100machines_30d.feather' # Your processed data

# VAE Configuration
SEQUENCE_LENGTH = 96  # Length of the input/output time series sequences
TARGET_PDU = 'pdu6'   # Focus on one PDU for simplicity first

# Select features to use for the VAE. VAEs learn to reconstruct these.
# For a VAE, the input and output are typically the same features.
FEATURE_COLUMNS_VAE = ['pdu_sum_cpu_usage', 'production_power_util'] # Example features to model

LATENT_DIM = 16  # Dimensionality of the latent space (hyperparameter)
LSTM_HIDDEN_DIM_VAE = 64 # Hidden dimensions for LSTMs in encoder/decoder
LSTM_NUM_LAYERS_VAE = 1 # Number of LSTM layers
DROPOUT_PROB = 0.1

# Training Parameters
LEARNING_RATE_VAE = 0.001
EPOCHS_VAE = 100 # VAEs might need more epochs
BATCH_SIZE_VAE = 32
KLD_WEIGHT = 0.00025 # Weight for the KL divergence part of the loss (important hyperparameter)

# Output directory for model and plots
OUTPUT_DIR_VAE = "vae_model_output"
os.makedirs(OUTPUT_DIR_VAE, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Function for Sequence Creation (can be reused) ---
def create_sequences_for_autoencoder(input_data, sequence_length):
    """
    Creates sequences for autoencoder-like models (input is also the target for reconstruction).
    input_data: NumPy array of features (already scaled).
    sequence_length: Length of the input/output sequences.
    """
    X = []
    for i in range(len(input_data) - sequence_length + 1): # +1 to use all data
        X.append(input_data[i:(i + sequence_length), :])
    return np.array(X)





# --- PyTorch VAE Model Definition ---
class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim, latent_dim, num_layers=1, dropout_prob=0.1):
        super(TimeSeriesVAE, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        # Takes sequence_length x input_dim -> outputs features for latent space params
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim) # To output mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # To output log_var

        # Decoder
        # Takes latent_dim -> outputs sequence_length x input_dim
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim) # Initial projection from latent space
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.decoder_output_fc = nn.Linear(hidden_dim, input_dim) # To reconstruct original features

    def encode(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, (h_n, c_n) = self.encoder_lstm(x)
        # Use the last hidden state of the LSTM
        last_hidden_state = h_n[-1] # If num_layers > 1, h_n is (num_layers, batch, hidden_dim)
                                    # If num_layers = 1, h_n is (1, batch, hidden_dim), so h_n[0] or h_n[-1]
        mu = self.fc_mu(last_hidden_state)
        logvar = self.fc_logvar(last_hidden_state)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample epsilon from standard normal
        return mu + eps * std

    def decode(self, z):
        # z shape: (batch, latent_dim)
        # We need to feed a sequence to the LSTM decoder.
        # One way is to repeat the latent vector z for each time step of the output sequence.
        hidden_init = self.decoder_fc(z) # Project z to hidden_dim
        
        # Prepare input for decoder LSTM: repeat hidden_init sequence_length times
        # Shape should be (batch, seq_len, hidden_dim_for_decoder_lstm_input)
        # Here, we are using the projected z as the *first* input to an LSTM
        # that then generates the sequence. Or, treat hidden_init as initial hidden state.

        # Let's try using hidden_init as the initial hidden state for the decoder LSTM
        # and feed a dummy start-of-sequence input (e.g., zeros)
        # This approach is common: use z to initialize LSTM state, then unroll.
        h_dec_init = hidden_init.unsqueeze(0).repeat(self.num_layers, 1, 1) # (num_layers, batch, hidden_dim)
        c_dec_init = torch.zeros_like(h_dec_init) # (num_layers, batch, hidden_dim)

        # Decoder LSTM input can be a learned "start token" or just zeros, repeated for seq_length
        # For reconstruction, a common trick is to teacher-force or use zeros as input.
        # Let's use a simpler approach for now: just unroll based on z.
        # Simpler decoder: project z and use it as input to LSTM for each step (less common for VAE generation)
        # A more standard way: use z to condition the initial hidden state of an autoregressive decoder LSTM.

        # For a non-autoregressive decoder based on LSTMs (simpler for VAE reconstruction task):
        # Repeat z for each time step as input to decoder_lstm
        # This is a simplification. A true generative LSTM decoder would often be autoregressive.
        decoder_lstm_input = hidden_init.unsqueeze(1).repeat(1, self.sequence_length, 1)
        # decoder_lstm_input shape: (batch, seq_len, hidden_dim)
        
        lstm_out_dec, _ = self.decoder_lstm(decoder_lstm_input, (h_dec_init, c_dec_init))
        # lstm_out_dec shape: (batch, seq_len, hidden_dim)
        
        reconstruction = torch.sigmoid(self.decoder_output_fc(lstm_out_dec)) # Sigmoid if data scaled [0,1]
        return reconstruction

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

# VAE Loss function
def vae_loss_function(recon_x, x, mu, logvar, kld_weight):
    # Reconstruction loss (e.g., Mean Squared Error)
    # Ensure x is shaped the same as recon_x if it was flattened for an MLP decoder
    # Here, x and recon_x should be (batch, seq_len, feature_dim)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum') # Sum over all elements

    # KL divergence
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_weight * kld, recon_loss, kld




def main_vae():
    logging.info("--- Starting PyTorch VAE Model Training Pipeline ---")
    overall_start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Processed Data
    logging.info(f"Loading data from {INPUT_FEATHER_FILE}...")
    # ... (same data loading as in train_lstm_pytorch.py) ...
    try:
        df_full_merged = pd.read_feather(INPUT_FEATHER_FILE)
    except Exception as e:
        logging.error(f"Could not load {INPUT_FEATHER_FILE}. Did you run data processing script? Error: {e}")
        return
    logging.info(f"Data loaded successfully. Shape: {df_full_merged.shape}")

    # 2. Filter for Target PDU and Sort
    logging.info(f"Filtering data for PDU: {TARGET_PDU}")
    df_pdu_data = df_full_merged[df_full_merged['pdu'] == TARGET_PDU].copy()
    if df_pdu_data.empty:
        logging.error(f"No data found for PDU: {TARGET_PDU}. Avail: {df_full_merged['pdu'].unique()}")
        return
    df_pdu_data.set_index('datetime', inplace=True)
    df_pdu_data.sort_index(inplace=True)
    logging.info(f"Data for PDU {TARGET_PDU} shape: {df_pdu_data.shape}")

    # 3. Feature Selection (for VAE, input and output are these features)
    logging.info(f"Selecting features for VAE: {FEATURE_COLUMNS_VAE}")
    if not all(col in df_pdu_data.columns for col in FEATURE_COLUMNS_VAE):
        logging.error(f"One or more VAE FEATURE_COLUMNS not in DataFrame. Available: {df_pdu_data.columns}")
        return
    features_df_vae = df_pdu_data[FEATURE_COLUMNS_VAE]

    # 4. Normalization
    logging.info("Normalizing data for VAE...")
    scaler_vae = MinMaxScaler(feature_range=(0, 1)) # VAEs often use sigmoid in output, so [0,1] is good
    scaled_data_vae = scaler_vae.fit_transform(features_df_vae.values)

    # 5. Sequence Preparation for VAE
    # For a VAE, each sequence is an input, and the target is the same sequence (reconstruction)
    logging.info(f"Creating sequences for VAE with length: {SEQUENCE_LENGTH}")
    sequences_vae = create_sequences_for_autoencoder(scaled_data_vae, SEQUENCE_LENGTH)
    if sequences_vae.shape[0] == 0:
        logging.error("Failed to create VAE sequences. Not enough data.")
        return
    logging.info(f"VAE sequences created. Shape: {sequences_vae.shape}") # (num_samples, seq_len, num_features)

    # 6. Train/Test Split (Optional for VAE if only interested in generation, but good for checking reconstruction)
    # For VAE, X_train is sequences_vae, and y_train is also sequences_vae (target is reconstruction)
    train_sequences, test_sequences = train_test_split(sequences_vae, test_size=0.2, shuffle=False) # Shuffle False for time series context
    logging.info(f"Train sequences: {train_sequences.shape}, Test sequences: {test_sequences.shape}")

    # 7. Convert to PyTorch Tensors and Create DataLoaders
    X_train_vae_tensor = torch.from_numpy(train_sequences).float().to(device)
    # Target is the same as input for reconstruction
    y_train_vae_tensor = torch.from_numpy(train_sequences).float().to(device)

    X_test_vae_tensor = torch.from_numpy(test_sequences).float().to(device)
    y_test_vae_tensor = torch.from_numpy(test_sequences).float().to(device)

    train_dataset_vae = TensorDataset(X_train_vae_tensor, y_train_vae_tensor)
    train_loader_vae = DataLoader(train_dataset_vae, batch_size=BATCH_SIZE_VAE, shuffle=True) # Shuffle for VAE training

    test_dataset_vae = TensorDataset(X_test_vae_tensor, y_test_vae_tensor)
    test_loader_vae = DataLoader(test_dataset_vae, batch_size=BATCH_SIZE_VAE, shuffle=False)


    # 8. Build, Train, and Evaluate VAE Model
    logging.info("Building PyTorch VAE model...")
    model_input_dim_vae = scaled_data_vae.shape[1] # Number of features
    vae_model = TimeSeriesVAE(
        input_dim=model_input_dim_vae,
        sequence_length=SEQUENCE_LENGTH,
        hidden_dim=LSTM_HIDDEN_DIM_VAE,
        latent_dim=LATENT_DIM,
        num_layers=LSTM_NUM_LAYERS_VAE,
        dropout_prob=DROPOUT_PROB
    )
    vae_model.to(device)
    logging.info(vae_model)

    optimizer_vae = optim.Adam(vae_model.parameters(), lr=LEARNING_RATE_VAE)

    logging.info(f"Training PyTorch VAE model for {EPOCHS_VAE} epochs...")
    train_start_time_actual_vae = time.time()
    
    # For storing losses
    epoch_losses = []
    recon_losses_epoch = []
    kld_losses_epoch = []

    for epoch in range(EPOCHS_VAE):
        vae_model.train()
        total_epoch_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0
        for batch_idx, (data_batch, _) in enumerate(train_loader_vae): # Target is ignored as it's same as data_batch
            data_batch = data_batch.to(device) # Input is data_batch
            optimizer_vae.zero_grad()
            
            recon_batch, mu, logvar = vae_model(data_batch)
            loss, recon_loss_val, kld_val = vae_loss_function(recon_batch, data_batch, mu, logvar, KLD_WEIGHT)
            
            loss.backward()
            optimizer_vae.step()
            
            total_epoch_loss += loss.item()
            total_recon_loss += recon_loss_val.item()
            total_kld_loss += kld_val.item()
        
        avg_epoch_loss = total_epoch_loss / len(train_loader_vae.dataset) # Avg loss per sample
        avg_recon_loss = total_recon_loss / len(train_loader_vae.dataset)
        avg_kld_loss = total_kld_loss / len(train_loader_vae.dataset)

        epoch_losses.append(avg_epoch_loss)
        recon_losses_epoch.append(avg_recon_loss)
        kld_losses_epoch.append(avg_kld_loss)

        logging.info(f"Epoch {epoch+1}/{EPOCHS_VAE}, Total Loss: {avg_epoch_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, KLD: {avg_kld_loss:.6f}")

    logging.info(f"VAE model training finished in {time.time() - train_start_time_actual_vae:.2f} seconds.")
    
    # Save the trained VAE model
    vae_model_save_path = os.path.join(OUTPUT_DIR_VAE, f"vae_model_pdu_{TARGET_PDU}.pth")
    torch.save(vae_model.state_dict(), vae_model_save_path)
    logging.info(f"VAE Model state saved to {vae_model_save_path}")

    # 9. (Optional) Evaluate VAE - e.g., reconstruction on test set
    vae_model.eval()
    test_loss_total_vae = 0
    with torch.no_grad():
        for data_batch_test, _ in test_loader_vae:
            data_batch_test = data_batch_test.to(device)
            recon_batch_test, mu_test, logvar_test = vae_model(data_batch_test)
            loss_test, _, _ = vae_loss_function(recon_batch_test, data_batch_test, mu_test, logvar_test, KLD_WEIGHT)
            test_loss_total_vae += loss_test.item()
    avg_test_loss_vae = test_loss_total_vae / len(test_loader_vae.dataset)
    logging.info(f"VAE Test Set Average Loss: {avg_test_loss_vae:.6f}")

    # Plotting losses
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_losses, label='Total VAE Loss')
    plt.plot(recon_losses_epoch, label='Reconstruction Loss')
    plt.plot(kld_losses_epoch, label='KLD Loss')
    plt.title('VAE Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR_VAE, f"vae_loss_curve_pdu_{TARGET_PDU}.png"))
    logging.info(f"VAE Loss curve saved to {os.path.join(OUTPUT_DIR_VAE, f'vae_loss_curve_pdu_{TARGET_PDU}.png')}")

    # Visualize some reconstructions
    if X_test_vae_tensor.shape[0] > 0: # Use the PyTorch tensor's shape
        num_samples_to_plot = min(5, X_test_vae_tensor.shape[0])
        random_indices = np.random.choice(X_test_vae_tensor.shape[0], num_samples_to_plot, replace=False)

        # Select samples directly from the PyTorch tensor, which is already on the correct device
        original_samples_for_plot = X_test_vae_tensor[random_indices] 

        with torch.no_grad():
            # Ensure model is on the same device as original_samples_for_plot
            vae_model.to(device) # Should already be on device, but good to be sure
            reconstructed_samples, _, _ = vae_model(original_samples_for_plot) # Pass the tensor

        # Inverse transform for plotting if desired (requires scaler_vae to be fitted on multi-feature data)
        # For simplicity, plotting scaled versions or first feature
        original_samples_plot = scaler_vae.inverse_transform(original_samples_for_plot[0].cpu().numpy().reshape(-1, model_input_dim_vae))[:,0] # first sample, first feature
        reconstructed_samples_plot = scaler_vae.inverse_transform(reconstructed_samples[0].cpu().numpy().reshape(-1, model_input_dim_vae))[:,0] # first sample, first feature

        plt.figure(figsize=(12, 6))
        plt.plot(original_samples_plot, label=f'Original Sample (1st feature, PDU {TARGET_PDU})')
        plt.plot(reconstructed_samples_plot, label=f'Reconstructed Sample (1st feature, PDU {TARGET_PDU})', linestyle='--')
        plt.title('VAE Reconstruction Example')
        plt.xlabel('Time Step (15-min interval in sequence)')
        plt.ylabel(f"{FEATURE_COLUMNS_VAE[0]} (Original Scale)")
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR_VAE, f"vae_reconstruction_pdu_{TARGET_PDU}.png"))
        logging.info(f"VAE Reconstruction plot saved.")

    # How to Generate New Samples with VAE:
    # with torch.no_grad():
    #    noise = torch.randn(NUM_SAMPLES_TO_GENERATE, LATENT_DIM).to(device)
    #    generated_samples_scaled = vae_model.decode(noise) # Pass through decoder
    #    generated_samples_original_scale = scaler_vae.inverse_transform(generated_samples_scaled.cpu().numpy().reshape(-1, model_input_dim_vae))
    # logging.info(f"Generated {generated_samples_original_scale.shape[0]} samples of shape {generated_samples_original_scale.shape[1:]}")

    logging.info(f"--- PyTorch VAE Model Training Pipeline Finished in {time.time() - overall_start_time:.2f} seconds ---")


if __name__ == "__main__":
    main_vae()