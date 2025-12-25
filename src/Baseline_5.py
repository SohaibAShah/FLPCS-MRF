import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import copy

# ==================== CONFIGURATION ====================
class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Config
    SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    TOTAL_CLIENTS = 12
    
    # Training Config
    ROUNDS = 30
    NUM_SELECTED_CLIENTS = 6
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 0.001
    
    # MoE Config
    NUM_EXPERTS = 8          # Number of experts in the mixture
    TOP_K = 2                # Number of experts active per example
    AUX_LOSS_COEF = 0.01     # Weight for load balancing loss
    NOISY_GATING = True      # Add noise to gating for exploration

# ==================== SEED REPRODUCIBILITY ====================
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
print(f"Running on device: {Config.DEVICE}")

# ==================== DATASET CLASS ====================
class CustomDataset(Dataset):
    def __init__(self, csv_features, img1_features, img2_features, labels):
        self.csv = torch.tensor(csv_features, dtype=torch.float32)
        # Ensure channel dimension is present: (N, H, W, C) -> (N, C, H, W) done in model or here
        # Input provided is (N, 32, 32, 1), we keep it as is and permute in model
        self.img1 = torch.tensor(img1_features, dtype=torch.float32)
        self.img2 = torch.tensor(img2_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.csv[idx], self.img1[idx], self.img2[idx], self.labels[idx]

# ==================== DATA LOADING LOGIC ====================
def load_clients_data():
    """
    Loads data for all subjects, performing alignment, scaling, and splitting.
    """
    data = {}
    print("Loading multimodal data...")
    
    for idx, sub in enumerate(Config.SUBJECTS):
        try:
            # --- TRAIN LOADING ---
            train_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_train.csv', skiprows=1)
            train_csv.dropna(inplace=True)
            train_csv.drop_duplicates(inplace=True)
            # Drop unused infrared columns
            drop_cols = ['Infrared 1','Infrared 2','Infrared 3','Infrared 4','Infrared 5','Infrared 6']
            train_csv.drop([c for c in drop_cols if c in train_csv.columns], axis=1, inplace=True)
            na_cols = train_csv.columns[train_csv.isnull().any()]
            train_csv.drop(na_cols, axis=1, inplace=True)
            train_csv.set_index('Time', inplace=True)

            img1_train = np.load(f'./dataset/Sensor + Image/{sub}_image_1_train.npy') / 255.0
            img2_train = np.load(f'./dataset/Sensor + Image/{sub}_image_2_train.npy') / 255.0
            label_train = np.load(f'./dataset/Sensor + Image/{sub}_label_1_train.npy')
            name_train = np.load(f'./dataset/Sensor + Image/{sub}_name_1_train.npy')

            # Align Sensor and Images based on Time Index
            valid_idx = np.isin(name_train, train_csv.index)
            img1_train = img1_train[valid_idx]
            img2_train = img2_train[valid_idx]
            label_train = label_train[valid_idx]
            train_data = train_csv.loc[name_train[valid_idx]].values
            
            # Clean Labels
            label_train[label_train == 20] = 0

            # Scale Sensor Data
            X_train_csv = train_data[:, :-1] # Exclude label col if present in csv array
            y_train = label_train.astype(np.int64)
            scaler = StandardScaler()
            X_train_csv = scaler.fit_transform(X_train_csv)

            img1_train = img1_train.reshape(-1, 32, 32, 1)
            img2_train = img2_train.reshape(-1, 32, 32, 1)

            # --- TEST LOADING ---
            test_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_test.csv', skiprows=1)
            test_csv.dropna(inplace=True)
            test_csv.drop_duplicates(inplace=True)
            test_csv.drop([c for c in drop_cols if c in test_csv.columns], axis=1, inplace=True)
            test_csv.drop(na_cols, axis=1, inplace=True)
            test_csv.set_index('Time', inplace=True)

            img1_test = np.load(f'./dataset/Sensor + Image/{sub}_image_1_test.npy') / 255.0
            img2_test = np.load(f'./dataset/Sensor + Image/{sub}_image_2_test.npy') / 255.0
            label_test = np.load(f'./dataset/Sensor + Image/{sub}_label_1_test.npy')
            name_test = np.load(f'./dataset/Sensor + Image/{sub}_name_1_test.npy')

            valid_idx_test = np.isin(name_test, test_csv.index)
            img1_test = img1_test[valid_idx_test]
            img2_test = img2_test[valid_idx_test]
            label_test = label_test[valid_idx_test]
            test_data = test_csv.loc[name_test[valid_idx_test]].values
            
            label_test[label_test == 20] = 0

            X_test_csv = test_data[:, :-1]
            y_test = label_test.astype(np.int64)
            X_test_csv = scaler.transform(X_test_csv) # Use train scaler

            img1_test = img1_test.reshape(-1, 32, 32, 1)
            img2_test = img2_test.reshape(-1, 32, 32, 1)

            data[idx] = {
                'train': (X_train_csv, img1_train, img2_train, y_train),
                'test': (X_test_csv, img1_test, img2_test, y_test)
            }
        except Exception as e:
            print(f"Error loading client {sub}: {e}")
            continue
            
    return data

# ==================== MOE MODEL ARCHITECTURE ====================

class SensorEncoder(nn.Module):
    """Encodes tabular/sensor data into an embedding."""
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class ImageEncoder(nn.Module):
    """Encodes 32x32 images into an embedding."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # Expects (N, H, W, C), needs permutation to (N, C, H, W)
        x = x.permute(0, 3, 1, 2) 
        return self.net(x)

class Expert(nn.Module):
    """
    A single expert module.
    Each expert specializes in processing the fused multimodal embedding.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU() # Output is feature representation, not logits yet
        )

    def forward(self, x):
        return self.net(x)

class TopKRouter(nn.Module):
    """
    Routes inputs to top-k experts and calculates load balancing loss.
    """
    def __init__(self, input_dim, num_experts, k=2, noisy=True):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy = noisy
        self.softplus = nn.Softplus()
        
        # Trainable noise parameters
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))

    def forward(self, x):
        # x: (batch_size, input_dim)
        clean_logits = self.gate(x)
        
        if self.noisy and self.training:
            noise_logits = x @ self.w_noise
            noise = torch.randn_like(clean_logits) * self.softplus(noise_logits)
            logits = clean_logits + noise
        else:
            logits = clean_logits

        # Calculate routing probabilities (for load balancing)
        # We use softmax over all experts to get probability distribution
        probs = F.softmax(logits, dim=1)

        # Select Top-K
        top_k_vals, top_k_indices = torch.topk(logits, self.k, dim=1)
        
        # Normalize weights for the selected k experts
        top_k_gates = F.softmax(top_k_vals, dim=1)

        # --- Load Balancing Loss Calculation ---
        # 1. Importance: Sum of probabilities assigned to each expert across batch
        importance = probs.sum(0)
        
        # 2. Load: Count how often each expert was selected (hard selection)
        # Create a mask of selected experts
        batch_size = x.size(0)
        mask = torch.zeros((batch_size, self.num_experts), device=x.device)
        mask.scatter_(1, top_k_indices, 1.0)
        load = mask.sum(0)
        
        # Aux Loss = mean(importance * load) * (num_experts^2)
        # We want uniform distribution, minimizing variance of importance and load
        aux_loss = (importance * load).mean() * (self.num_experts ** 2)

        return top_k_gates, top_k_indices, aux_loss

class FedMoEModel(nn.Module):
    """
    Full Multimodal Federated Mixture of Experts Model.
    Structure:
    1. Sensor & Image Encoders (Shared feature extraction)
    2. MoE Layer (Fusion & Specialization)
    3. Classifier Head
    """
    def __init__(self, csv_input_dim, num_classes=12):
        super().__init__()
        
        embed_dim = 256
        fusion_dim = embed_dim * 3 # CSV + IMG1 + IMG2
        
        # Shared Feature Extractors
        self.csv_encoder = SensorEncoder(csv_input_dim, embed_dim)
        self.img_encoder = ImageEncoder(embed_dim) # Shared for both images

        # MoE Components
        self.router = TopKRouter(fusion_dim, Config.NUM_EXPERTS, k=Config.TOP_K, noisy=Config.NOISY_GATING)
        
        # Experts (ModuleList allows independent updates)
        self.experts = nn.ModuleList([
            Expert(fusion_dim, 512, 256) for _ in range(Config.NUM_EXPERTS)
        ])
        
        # Final Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_csv, x_img1, x_img2):
        # 1. Feature Encoding
        h_csv = self.csv_encoder(x_csv)
        h_img1 = self.img_encoder(x_img1)
        h_img2 = self.img_encoder(x_img2)
        
        # 2. Fusion (Concatenation)
        x_fused = torch.cat([h_csv, h_img1, h_img2], dim=1) # (B, fusion_dim)
        
        # 3. Gating / Routing
        # gates: (B, k), indices: (B, k), aux_loss: scalar
        gates, expert_indices, aux_loss = self.router(x_fused)
        
        # 4. Expert Processing (Sparse Execution)
        # Result accumulator
        final_expert_output = torch.zeros(x_fused.size(0), 256, device=x_fused.device)
        
        # Iterate over the k selected experts
        for k_idx in range(Config.TOP_K):
            selected_experts = expert_indices[:, k_idx] # Which expert is the k-th choice for each sample?
            selected_weights = gates[:, k_idx].unsqueeze(1) # Weight for that choice
            
            # Efficiently batch process per unique expert index
            # (Instead of looping 0..E, we can mask, but looping 0..E is cleaner in PyTorch Eager mode)
            for e_id in range(Config.NUM_EXPERTS):
                # Find samples in batch that chose expert e_id as their k-th choice
                mask = (selected_experts == e_id)
                if mask.any():
                    # Process only relevant samples
                    inp_subset = x_fused[mask]
                    out_subset = self.experts[e_id](inp_subset)
                    
                    # Accumulate weighted output
                    # final_out[mask] += weight * expert_out
                    final_expert_output[mask] += selected_weights[mask] * out_subset
        
        # 5. Classification
        logits = self.classifier(final_expert_output)
        
        return logits, aux_loss

# ==================== FEDERATED LEARNING HELPERS ====================

def train_client(global_model, client_data):
    """
    Trains the MoE model locally on a specific client.
    Returns: Updated model state_dict and metrics.
    """
    # Unpack data
    X_csv, X1, X2, y = client_data['train']
    train_loader = DataLoader(CustomDataset(X_csv, X1, X2, y), 
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # Initialize Local Model
    local_model = copy.deepcopy(global_model)
    local_model.train()
    
    optimizer = torch.optim.Adam(local_model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for _ in range(Config.LOCAL_EPOCHS):
        batch_losses = []
        for batch in train_loader:
            csv_b, img1_b, img2_b, y_b = [x.to(Config.DEVICE) for x in batch]
            
            optimizer.zero_grad()
            
            # Forward pass returns logits AND auxiliary load balancing loss
            logits, aux_loss = local_model(csv_b, img1_b, img2_b)
            
            # Primary task loss
            task_loss = criterion(logits, y_b)
            
            # Total loss
            total_loss = task_loss + (Config.AUX_LOSS_COEF * aux_loss)
            
            total_loss.backward()
            optimizer.step()
            
            batch_losses.append(total_loss.item())
        epoch_losses.append(sum(batch_losses)/len(batch_losses))
        
    return local_model.state_dict(), sum(epoch_losses)/len(epoch_losses)

def evaluate_global(model, clients_data, validation_split=False):
    """
    Evaluates the global model on all clients' test sets.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for cid in clients_data:
            # Determine which set to use
            if validation_split:
                data_tuple = clients_data[cid]['train'] # In a real scenario, split train for val
            else:
                data_tuple = clients_data[cid]['test']
                
            X_csv, X1, X2, y = data_tuple
            test_loader = DataLoader(CustomDataset(X_csv, X1, X2, y), batch_size=Config.BATCH_SIZE)
            
            for batch in test_loader:
                csv_b, img1_b, img2_b, y_b = [x.to(Config.DEVICE) for x in batch]
                logits, _ = model(csv_b, img1_b, img2_b) # Ignore aux loss during eval
                preds = logits.argmax(dim=1)
                correct += preds.eq(y_b).sum().item()
                total += y_b.size(0)
                
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc

def aggregate_weights(weights_list):
    """
    FedAvg Aggregation: Component-wise average of model weights.
    """
    avg_weights = copy.deepcopy(weights_list[0])
    
    for key in avg_weights.keys():
        # Check type to handle buffers (like num_batches_tracked) vs parameters
        if avg_weights[key].is_floating_point():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
        else:
            # For integer buffers, cast to float, mean, cast back
            tmp = avg_weights[key].float()
            for i in range(1, len(weights_list)):
                tmp += weights_list[i][key].float()
            avg_weights[key] = (tmp / len(weights_list)).to(avg_weights[key].dtype)
            
    return avg_weights

# ==================== MAIN TRAINING LOOP ====================
def main():
    # 1. Load Data
    clients_data = load_clients_data()
    if not clients_data:
        print("No data loaded. Exiting.")
        return

    # 2. Initialize Global Model
    # Determine CSV dimension from first client
    csv_dim = clients_data[0]['train'][0].shape[1]
    print(f"Initializing FedMoE Model with {Config.NUM_EXPERTS} experts (Top-{Config.TOP_K})...")
    print(f"CSV Feature Dimension: {csv_dim}")
    
    global_model = FedMoEModel(csv_dim, num_classes=12).to(Config.DEVICE)
    
    best_acc = 0.0
    
    print("\n=== Starting Federated MoE Training ===")
    
    for round_num in range(1, Config.ROUNDS + 1):
        print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")
        
        # 3. Client Selection (Random for standard FL)
        # To implement Pareto, you would calculate metrics here first. 
        # Using Random selection as per standard baseline request first.
        selected_clients = random.sample(range(Config.TOTAL_CLIENTS), Config.NUM_SELECTED_CLIENTS)
        print(f"Selected Clients: {selected_clients}")
        
        local_weights_list = []
        local_losses = []
        
        # 4. Local Training
        for client_id in selected_clients:
            w, loss = train_client(global_model, clients_data[client_id])
            local_weights_list.append(w)
            local_losses.append(loss)
            print(f"  Client {client_id} | Loss: {loss:.4f}")
            
        # 5. Aggregation (FedAvg)
        # In FedMoE, we average the Router and ALL Experts.
        # This aligns the experts globally (Expert 1 on Client A aligns with Expert 1 on Client B).
        global_weights = aggregate_weights(local_weights_list)
        global_model.load_state_dict(global_weights)
        
        # 6. Global Evaluation
        test_acc = evaluate_global(global_model, clients_data)
        print(f"Global Test Accuracy: {test_acc:.2f}%")
        
        # 7. Save Best Model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(global_model.state_dict(), "best_fedmoe_model.pth")
            print("  [Saved Best Model]")
            
    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()