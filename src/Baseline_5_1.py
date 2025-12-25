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
    
    # Missing Modality Simulation Profiles
    # 0: Full Multimodal, 1: Sensor Only, 2: Image Only
    CLIENT_PROFILES = {
        0: 'Full', 1: 'Full', 2: 'Full', 3: 'Full',
        4: 'Sensor_Only', 5: 'Sensor_Only', 6: 'Sensor_Only', 7: 'Sensor_Only',
        8: 'Image_Only', 9: 'Image_Only', 10: 'Image_Only', 11: 'Image_Only'
    }

    # Training Config
    ROUNDS = 30
    NUM_SELECTED_CLIENTS = 6
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 0.001
    
    # MoE Config
    NUM_EXPERTS = 8          
    TOP_K = 2                
    AUX_LOSS_COEF = 0.01     
    NOISY_GATING = True      

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
        self.img1 = torch.tensor(img1_features, dtype=torch.float32)
        self.img2 = torch.tensor(img2_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.csv[idx], self.img1[idx], self.img2[idx], self.labels[idx]

# ==================== DATA LOADING WITH HETEROGENEITY ====================
def load_clients_data_heterogeneous():
    """
    Loads data and applies Zero-Filling to simulate missing modalities.
    """
    data = {}
    print("\n=== Loading Data & Simulating Modality Heterogeneity ===")
    
    for idx, sub in enumerate(Config.SUBJECTS):
        profile = Config.CLIENT_PROFILES.get(idx, 'Full')
        print(f"Client {idx} (Subject {sub}): Profile -> {profile}")

        try:
            # --- RAW DATA LOADING ---
            train_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_train.csv', skiprows=1)
            train_csv.dropna(inplace=True); train_csv.drop_duplicates(inplace=True)
            drop_cols = ['Infrared 1','Infrared 2','Infrared 3','Infrared 4','Infrared 5','Infrared 6']
            train_csv.drop([c for c in drop_cols if c in train_csv.columns], axis=1, inplace=True)
            na_cols = train_csv.columns[train_csv.isnull().any()]
            train_csv.drop(na_cols, axis=1, inplace=True)
            train_csv.set_index('Time', inplace=True)

            img1_train = np.load(f'./dataset/Sensor + Image/{sub}_image_1_train.npy') / 255.0
            img2_train = np.load(f'./dataset/Sensor + Image/{sub}_image_2_train.npy') / 255.0
            label_train = np.load(f'./dataset/Sensor + Image/{sub}_label_1_train.npy')
            name_train = np.load(f'./dataset/Sensor + Image/{sub}_name_1_train.npy')

            # Align
            valid_idx = np.isin(name_train, train_csv.index)
            img1_train = img1_train[valid_idx]
            img2_train = img2_train[valid_idx]
            label_train = label_train[valid_idx]
            train_data = train_csv.loc[name_train[valid_idx]].values
            label_train[label_train == 20] = 0

            # Scale Sensor Data
            X_train_csv = train_data[:, :-1]
            y_train = label_train.astype(np.int64)
            scaler = StandardScaler()
            X_train_csv = scaler.fit_transform(X_train_csv)

            img1_train = img1_train.reshape(-1, 32, 32, 1)
            img2_train = img2_train.reshape(-1, 32, 32, 1)

            # --- TEST DATA LOADING ---
            test_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_test.csv', skiprows=1)
            test_csv.dropna(inplace=True); test_csv.drop_duplicates(inplace=True)
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
            X_test_csv = scaler.transform(X_test_csv)

            img1_test = img1_test.reshape(-1, 32, 32, 1)
            img2_test = img2_test.reshape(-1, 32, 32, 1)

            # ============================================================
            # MISSING MODALITY SIMULATION (ZERO-FILLING)
            # ============================================================
            if profile == 'Sensor_Only':
                # Zero out images
                print(f"  -> Masking Images for Client {idx}")
                img1_train = np.zeros_like(img1_train)
                img2_train = np.zeros_like(img2_train)
                img1_test = np.zeros_like(img1_test)
                img2_test = np.zeros_like(img2_test)
            
            elif profile == 'Image_Only':
                # Zero out sensor CSV
                print(f"  -> Masking Sensor Data for Client {idx}")
                X_train_csv = np.zeros_like(X_train_csv)
                X_test_csv = np.zeros_like(X_test_csv)
            # ============================================================

            data[idx] = {
                'train': (X_train_csv, img1_train, img2_train, y_train),
                'test': (X_test_csv, img1_test, img2_test, y_test)
            }
        except Exception as e:
            print(f"Error loading client {sub}: {e}")
            continue
            
    return data

# ==================== MOE MODEL ARCHITECTURE (UNCHANGED) ====================
class SensorEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU(), nn.Dropout(0.2)
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) 
        return self.net(x)

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class TopKRouter(nn.Module):
    def __init__(self, input_dim, num_experts, k=2, noisy=True):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts
        self.k = k
        self.noisy = noisy
        self.softplus = nn.Softplus()
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts))

    def forward(self, x):
        clean_logits = self.gate(x)
        if self.noisy and self.training:
            noise_logits = x @ self.w_noise
            noise = torch.randn_like(clean_logits) * self.softplus(noise_logits)
            logits = clean_logits + noise
        else:
            logits = clean_logits

        probs = F.softmax(logits, dim=1)
        top_k_vals, top_k_indices = torch.topk(logits, self.k, dim=1)
        top_k_gates = F.softmax(top_k_vals, dim=1)

        # Load balancing loss
        importance = probs.sum(0)
        batch_size = x.size(0)
        mask = torch.zeros((batch_size, self.num_experts), device=x.device)
        mask.scatter_(1, top_k_indices, 1.0)
        load = mask.sum(0)
        aux_loss = (importance * load).mean() * (self.num_experts ** 2)

        return top_k_gates, top_k_indices, aux_loss

class FedMoEModel(nn.Module):
    def __init__(self, csv_input_dim, num_classes=12):
        super().__init__()
        embed_dim = 256
        fusion_dim = embed_dim * 3 
        
        self.csv_encoder = SensorEncoder(csv_input_dim, embed_dim)
        self.img_encoder = ImageEncoder(embed_dim) 

        self.router = TopKRouter(fusion_dim, Config.NUM_EXPERTS, k=Config.TOP_K, noisy=Config.NOISY_GATING)
        self.experts = nn.ModuleList([Expert(fusion_dim, 512, 256) for _ in range(Config.NUM_EXPERTS)])
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x_csv, x_img1, x_img2):
        h_csv = self.csv_encoder(x_csv)
        h_img1 = self.img_encoder(x_img1)
        h_img2 = self.img_encoder(x_img2)
        
        x_fused = torch.cat([h_csv, h_img1, h_img2], dim=1) 
        gates, expert_indices, aux_loss = self.router(x_fused)
        
        final_expert_output = torch.zeros(x_fused.size(0), 256, device=x_fused.device)
        
        for k_idx in range(Config.TOP_K):
            selected_experts = expert_indices[:, k_idx] 
            selected_weights = gates[:, k_idx].unsqueeze(1) 
            
            for e_id in range(Config.NUM_EXPERTS):
                mask = (selected_experts == e_id)
                if mask.any():
                    inp_subset = x_fused[mask]
                    out_subset = self.experts[e_id](inp_subset)
                    final_expert_output[mask] += selected_weights[mask] * out_subset
        
        logits = self.classifier(final_expert_output)
        return logits, aux_loss

# ==================== TRAINING HELPERS ====================

def train_client(global_model, client_data):
    X_csv, X1, X2, y = client_data['train']
    train_loader = DataLoader(CustomDataset(X_csv, X1, X2, y), 
                              batch_size=Config.BATCH_SIZE, shuffle=True)
    
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
            logits, aux_loss = local_model(csv_b, img1_b, img2_b)
            task_loss = criterion(logits, y_b)
            total_loss = task_loss + (Config.AUX_LOSS_COEF * aux_loss)
            total_loss.backward()
            optimizer.step()
            batch_losses.append(total_loss.item())
        epoch_losses.append(sum(batch_losses)/len(batch_losses))
    return local_model.state_dict(), sum(epoch_losses)/len(epoch_losses)

def evaluate_global(model, clients_data):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cid in clients_data:
            X_csv, X1, X2, y = clients_data[cid]['test']
            test_loader = DataLoader(CustomDataset(X_csv, X1, X2, y), batch_size=Config.BATCH_SIZE)
            for batch in test_loader:
                csv_b, img1_b, img2_b, y_b = [x.to(Config.DEVICE) for x in batch]
                logits, _ = model(csv_b, img1_b, img2_b)
                preds = logits.argmax(dim=1)
                correct += preds.eq(y_b).sum().item()
                total += y_b.size(0)
    acc = 100.0 * correct / total if total > 0 else 0.0
    return acc

def aggregate_weights(weights_list):
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights.keys():
        if avg_weights[key].is_floating_point():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(weights_list))
        else:
            tmp = avg_weights[key].float()
            for i in range(1, len(weights_list)):
                tmp += weights_list[i][key].float()
            avg_weights[key] = (tmp / len(weights_list)).to(avg_weights[key].dtype)
    return avg_weights

# ==================== MAIN ====================
if __name__ == "__main__":
    # 1. Load Data with Heterogeneity
    clients_data = load_clients_data_heterogeneous()
    if not clients_data:
        print("No data loaded. Exiting.")
    else:
        csv_dim = clients_data[0]['train'][0].shape[1]
        print(f"\nInitializing FedMoE Model with CSV Dim: {csv_dim}")
        
        global_model = FedMoEModel(csv_dim, num_classes=12).to(Config.DEVICE)
        best_acc = 0.0
        
        print("\n=== Starting Federated MoE Training (Heterogeneous Modalities) ===")
        for round_num in range(1, Config.ROUNDS + 1):
            print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")
            
            selected_clients = random.sample(range(Config.TOTAL_CLIENTS), Config.NUM_SELECTED_CLIENTS)
            print(f"Selected Clients: {selected_clients}")
            
            local_weights_list = []
            for client_id in selected_clients:
                w, loss = train_client(global_model, clients_data[client_id])
                local_weights_list.append(w)
                print(f"  Client {client_id} | Loss: {loss:.4f}")
            
            global_weights = aggregate_weights(local_weights_list)
            global_model.load_state_dict(global_weights)
            
            test_acc = evaluate_global(global_model, clients_data)
            print(f"Global Test Accuracy: {test_acc:.2f}%")
            
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(global_model.state_dict(), "best_fedmoe_hetero.pth")
                print("  [Saved Best Model]")
                
        print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")