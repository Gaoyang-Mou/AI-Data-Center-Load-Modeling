import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split # For splitting data
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import logging
import sys
import os
import time # For timing training

# --- Configuration ---

# Input file from the previous processing script
INPUT_FEATHER_FILE = 'merged_data_cell_a_pdu6_pdu7_approx100machines_30d.feather' # Or your specific output filename

# LSTM Configuration
SEQUENCE_LENGTH = 96  # Number of past 15-min intervals to use for prediction (e.g., 96 for 1 day)
TARGET_PDU = 'pdu6'   # Focus on one PDU. PDUs available based on previous script: ['pdu6', 'pdu7']
                      # Update this if your merged file has different PDUs or you want to target another.

# Select features and target. These must exist in your INPUT_FEATHER_FILE.
# Available columns in example 'merged_data_cell_a_pdu6_pdu7_1d.feather':
# 'datetime', 'pdu', 'cell', 'pdu_avg_cpu_usage', 'pdu_sum_cpu_usage',
# 'pdu_avg_memory_usage', 'pdu_sum_memory_usage', 'machine_count',
# 'measured_power_util', 'production_power_util'
FEATURE_COLUMNS = ['pdu_sum_cpu_usage', 'production_power_util'] # Example: use these as input features
TARGET_COLUMN = 'production_power_util' # Example: predict future production power utilization

# PyTorch Model Hyperparameters
INPUT_DIM_MODEL = len(FEATURE_COLUMNS) # Will be set based on selected features
HIDDEN_DIM = 50
NUM_LAYERS = 2 # Number of stacked LSTM layers
OUTPUT_DIM_MODEL = 1 # Predicting a single value
DROPOUT_PROB = 0.2

# Training Parameters
LEARNING_RATE = 0.001
EPOCHS = 50 # Start with a reasonable number, can increase later
BATCH_SIZE = 32

# Output directory for model and plots
OUTPUT_DIR = "pytorch_lstm_model_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

# --- Helper Functions ---
def create_sequences_pytorch(input_data, target_data_col, sequence_length):
    """
    Creates sequences for LSTM with PyTorch.
    input_data: NumPy array of features (already scaled).
    target_data_col: NumPy array of the target column (already scaled, 1D).
    sequence_length: Length of the input sequences.
    """
    X, y = [], []
    for i in range(len(input_data) - sequence_length):
        X.append(input_data[i:(i + sequence_length), :]) # Input sequence (all features)
        y.append(target_data_col[i + sequence_length])    # Target value (from the target column)
    return np.array(X), np.array(y)

# --- PyTorch Model Definition ---
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout_prob=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layer
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 25)
        self.relu = nn.ReLU() # Activation function
        self.fc2 = nn.Linear(25, output_dim)

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        # (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # We need to detach the hidden state to prevent gradients from flowing all the way back
        # if we were to do truncated backpropagation through time (common for stateful LSTMs).
        # For stateless LSTMs, it's less critical for each batch but good practice.
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # out: tensor of shape (batch_size, seq_length, hidden_dim)
        # We only want the output of the last time step for prediction
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# --- Main Script ---
def main():
    logging.info("--- Starting PyTorch LSTM Model Training Pipeline ---")
    overall_start_time = time.time()

    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Load Processed Data
    logging.info(f"Loading data from {INPUT_FEATHER_FILE}...")
    if not os.path.exists(INPUT_FEATHER_FILE):
        logging.error(f"Input file not found: {INPUT_FEATHER_FILE}.")
        return
    try:
        df_full_merged = pd.read_feather(INPUT_FEATHER_FILE)
        logging.info(f"Data loaded successfully. Shape: {df_full_merged.shape}")
    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        return

    # 2. Filter for the Target PDU and Sort
    logging.info(f"Filtering data for PDU: {TARGET_PDU}")
    df_pdu_data = df_full_merged[df_full_merged['pdu'] == TARGET_PDU].copy()
    if df_pdu_data.empty:
        logging.error(f"No data found for PDU: {TARGET_PDU}. Available PDUs: {df_full_merged['pdu'].unique()}")
        return
    
    df_pdu_data.set_index('datetime', inplace=True)
    df_pdu_data.sort_index(inplace=True) # Ensure data is sorted by time
    logging.info(f"Data for PDU {TARGET_PDU} shape: {df_pdu_data.shape}")
    
    if df_pdu_data.shape[0] < SEQUENCE_LENGTH + 2: # Need enough data for at least one sequence and train/test split
        logging.error(f"Not enough data points for PDU {TARGET_PDU} to create sequences and split. (Found {df_pdu_data.shape[0]})")
        return

    # 3. Feature Selection
    logging.info(f"Selecting features: {FEATURE_COLUMNS} and target: {TARGET_COLUMN}")
    if not all(col in df_pdu_data.columns for col in FEATURE_COLUMNS) or TARGET_COLUMN not in df_pdu_data.columns:
        logging.error(f"One or more specified FEATURE_COLUMNS or TARGET_COLUMN not in DataFrame. Available: {df_pdu_data.columns}")
        return

    features_df = df_pdu_data[FEATURE_COLUMNS]
    target_series = df_pdu_data[TARGET_COLUMN] # This will be a pandas Series

    # 4. Normalization
    logging.info("Normalizing data...")
    # Scale features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features_df.values)

    # Scale target separately (important for correct inverse transform of predictions)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target_series.values.reshape(-1, 1))

    # 5. Sequence Preparation
    logging.info(f"Creating sequences with length: {SEQUENCE_LENGTH}")
    X, y = create_sequences_pytorch(scaled_features, scaled_target.flatten(), SEQUENCE_LENGTH)
    if X.shape[0] == 0:
        logging.error("Failed to create sequences. X is empty.")
        return
    logging.info(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}")

    # 6. Train/Test Split
    logging.info("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    logging.info(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        logging.error("Not enough data for training or testing after split.")
        return

    # 7. Convert to PyTorch Tensors and Create DataLoaders
    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).float().unsqueeze(1).to(device) # Add feature dim for target
    X_test_tensor = torch.from_numpy(X_test).float().to(device)
    y_test_tensor = torch.from_numpy(y_test).float().unsqueeze(1).to(device)   # Add feature dim for target

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # shuffle=True for typical non-timeseries

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 8. Build, Train, and Evaluate LSTM Model
    logging.info("Building PyTorch LSTM model...")
    model_input_dim = X_train_tensor.shape[2] # Number of features
    model = LSTMModel(input_dim=model_input_dim, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS, output_dim=OUTPUT_DIM_MODEL, dropout_prob=DROPOUT_PROB)
    model.to(device) # Move model to the selected device (GPU or CPU)
    logging.info(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    logging.info(f"Training PyTorch LSTM model for {EPOCHS} epochs...")
    train_start_time_actual = time.time()
    train_losses = []
    # Simple early stopping parameters
    best_val_loss = float('inf')
    patience_counter = 0
    patience_epochs = 5 # Stop if val_loss doesn't improve for 5 epochs

    for epoch in range(EPOCHS):
        model.train()
        epoch_train_loss = 0
        for seq_batch, label_batch in train_loader:
            # Data is already on device from TensorDataset if created directly on device,
            # but good practice if DataLoader loads from CPU memory
            # seq_batch, label_batch = seq_batch.to(device), label_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(seq_batch)
            loss = criterion(outputs, label_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        avg_epoch_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_epoch_train_loss)
        logging.info(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_epoch_train_loss:.6f}")

        # Basic early stopping based on training loss (replace with validation loss if you add a val_loader)
        if avg_epoch_train_loss < best_val_loss:
            best_val_loss = avg_epoch_train_loss
            patience_counter = 0
            # Save the best model
            best_model_save_path = os.path.join(OUTPUT_DIR, f"pytorch_lstm_model_pdu_{TARGET_PDU}_best.pth")
            torch.save(model.state_dict(), best_model_save_path)
            logging.info(f"Best model saved to {best_model_save_path} at epoch {epoch+1}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience_epochs:
            logging.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
            
    logging.info(f"Model training finished in {time.time() - train_start_time_actual:.2f} seconds.")

    # Load the best model for evaluation if early stopping was used
    if os.path.exists(best_model_save_path):
        model.load_state_dict(torch.load(best_model_save_path, map_location=device))
        logging.info(f"Loaded best model from {best_model_save_path} for evaluation.")


    # Evaluation
    model.eval()
    test_loss_items = []
    all_y_pred_scaled = []
    all_y_test_scaled = []

    with torch.no_grad():
        for seq_batch, label_batch in test_loader:
            # seq_batch, label_batch = seq_batch.to(device), label_batch.to(device)
            outputs = model(seq_batch)
            loss = criterion(outputs, label_batch)
            test_loss_items.append(loss.item())
            all_y_pred_scaled.extend(outputs.cpu().numpy().flatten())
            all_y_test_scaled.extend(label_batch.cpu().numpy().flatten())

    avg_test_loss = np.mean(test_loss_items)
    logging.info(f"Test Loss (MSE, scaled): {avg_test_loss:.6f}")

    # Inverse transform predictions and actuals
    y_pred_inversed = target_scaler.inverse_transform(np.array(all_y_pred_scaled).reshape(-1, 1)).flatten()
    y_test_inversed = target_scaler.inverse_transform(np.array(all_y_test_scaled).reshape(-1, 1)).flatten()

    # 9. Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    # Add validation losses here if you implement a validation loop
    plt.title('Model Loss During Training')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(OUTPUT_DIR, f"pytorch_lstm_loss_curve_pdu_{TARGET_PDU}.png"))
    logging.info(f"Loss curve saved to {os.path.join(OUTPUT_DIR, f'pytorch_lstm_loss_curve_pdu_{TARGET_PDU}.png')}")

    plt.figure(figsize=(14, 7))
    plt.plot(y_test_inversed, label='Actual Values', color='blue', alpha=0.7)
    plt.plot(y_pred_inversed, label='Predicted Values', color='red', linestyle='--', alpha=0.7)
    plt.title(f'PyTorch LSTM Predictions vs Actual for {TARGET_COLUMN} (PDU: {TARGET_PDU})')
    plt.ylabel(TARGET_COLUMN + " (Original Scale)")
    plt.xlabel('Time Steps (15-min intervals in test set)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"pytorch_lstm_predictions_pdu_{TARGET_PDU}.png"))
    logging.info(f"Predictions plot saved to {os.path.join(OUTPUT_DIR, f'pytorch_lstm_predictions_pdu_{TARGET_PDU}.png')}")

    logging.info(f"--- PyTorch LSTM Model Training Pipeline Finished in {time.time() - overall_start_time:.2f} seconds ---")

if __name__ == "__main__":
    main()