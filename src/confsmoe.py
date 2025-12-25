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
    
    # Dataset
    SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    TOTAL_CLIENTS = 12
    NUM_CLASSES = 12
    
    # Missing Modality Simulation Profiles
    # 0: Full Multimodal, 1: Sensor Only, 2: Image Only
    # This simulates arbitrary severe missingness across the network
    CLIENT_PROFILES = {
        0: 'Full', 1: 'Full', 2: 'Full', 3: 'Full',          # Full Modality Clients
        4: 'Sensor_Only', 5: 'Sensor_Only', 6: 'Sensor_Only', 7: 'Sensor_Only', # Missing Cameras
        8: 'Image_Only', 9: 'Image_Only', 10: 'Image_Only', 11: 'Image_Only'    # Missing Sensors
    }

    # Training
    ROUNDS = 30
    NUM_SELECTED_CLIENTS = 6
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 0.001
    
    # ConfSMoE Specifics
    NUM_EXPERTS = 8          # Number of experts
    TOP_K = 2                # Top-k experts to activate
    CONFIDENCE_TEMP = 0.1    # Temperature for confidence scaling (if needed)

# ==================== SEED & UTILS ====================
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
class ConfSMoEDataset(Dataset):
    def __init__(self, csv_features, img1_features, img2_features, labels, mask):
        self.csv = torch.tensor(csv_features, dtype=torch.float32)
        self.img1 = torch.tensor(img1_features, dtype=torch.float32)
        self.img2 = torch.tensor(img2_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).reshape(-1)
        # Mask is a fixed profile for the client (e.g., [1,0,0] for Sensor Only)
        self.mask = torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # We enforce zero-filling here based on the client's profile mask
        c = self.csv[idx] * self.mask[0]
        i1 = self.img1[idx] * self.mask[1]
        i2 = self.img2[idx] * self.mask[2]
        return c, i1, i2, self.labels[idx], self.mask

# ==================== DATA LOADING (With Heterogeneity) ====================
def load_data_heterogeneous():
    data = {}
    print("\n=== Loading Data with Simulated Arbitrary Missing Modalities ===")
    
    for idx, sub in enumerate(Config.SUBJECTS):
        profile = Config.CLIENT_PROFILES.get(idx, 'Full')
        
        # Define mask based on profile: [Sensor, Img1, Img2]
        if profile == 'Full':
            mask = [1, 1, 1]
        elif profile == 'Sensor_Only':
            mask = [1, 0, 0]
        elif profile == 'Image_Only':
            mask = [0, 1, 1]
        else:
            mask = [1, 1, 1] # Default
            
        print(f"Client {idx} (Subject {sub}): Profile {profile} -> Mask {mask}")

        try:
            # --- Load Raw Data ---
            train_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_train.csv', skiprows=1)
            train_csv.dropna(inplace=True); train_csv.drop_duplicates(inplace=True)
            drop_cols = ['Infrared 1','Infrared 2','Infrared 3','Infrared 4','Infrared 5','Infrared 6']
            train_csv.drop([c for c in drop_cols if c in train_csv.columns], axis=1, inplace=True)
            train_csv.drop(train_csv.columns[train_csv.isnull().any()], axis=1, inplace=True)
            train_csv.set_index('Time', inplace=True)

            img1_tr = np.load(f'./dataset/Sensor + Image/{sub}_image_1_train.npy') / 255.0
            img2_tr = np.load(f'./dataset/Sensor + Image/{sub}_image_2_train.npy') / 255.0
            lbl_tr = np.load(f'./dataset/Sensor + Image/{sub}_label_1_train.npy')
            name_tr = np.load(f'./dataset/Sensor + Image/{sub}_name_1_train.npy')

            # Align
            valid = np.isin(name_tr, train_csv.index)
            img1_tr, img2_tr, lbl_tr = img1_tr[valid], img2_tr[valid], lbl_tr[valid]
            csv_tr = train_csv.loc[name_tr[valid]].values
            lbl_tr[lbl_tr == 20] = 0

            # Scale
            scaler = StandardScaler()
            X_train_csv = scaler.fit_transform(csv_tr[:, :-1])
            y_train = lbl_tr.astype(np.int64)
            img1_train = img1_tr.reshape(-1, 32, 32, 1)
            img2_train = img2_tr.reshape(-1, 32, 32, 1)

            # Test Data (Assume Test is CLEAN/FULL for evaluation, or use same mask)
            # For robust evaluation, we usually test on full data to see if imputation works,
            # but FL often assumes client hardware is consistent train/test. 
            # We will apply the SAME mask to test to simulate hardware limitation.
            
            test_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_test.csv', skiprows=1)
            test_csv.dropna(inplace=True); test_csv.drop_duplicates(inplace=True)
            test_csv.drop([c for c in drop_cols if c in test_csv.columns], axis=1, inplace=True)
            test_csv.drop(test_csv.columns[test_csv.isnull().any()], axis=1, inplace=True)
            test_csv.set_index('Time', inplace=True)

            img1_te = np.load(f'./dataset/Sensor + Image/{sub}_image_1_test.npy') / 255.0
            img2_te = np.load(f'./dataset/Sensor + Image/{sub}_image_2_test.npy') / 255.0
            lbl_te = np.load(f'./dataset/Sensor + Image/{sub}_label_1_test.npy')
            name_te = np.load(f'./dataset/Sensor + Image/{sub}_name_1_test.npy')

            valid_te = np.isin(name_te, test_csv.index)
            img1_te, img2_te, lbl_te = img1_te[valid_te], img2_te[valid_te], lbl_te[valid_te]
            csv_te = test_csv.loc[name_te[valid_te]].values
            lbl_te[lbl_te == 20] = 0

            X_test_csv = scaler.transform(csv_te[:, :-1])
            y_test = lbl_te.astype(np.int64)
            img1_test = img1_te.reshape(-1, 32, 32, 1)
            img2_test = img2_te.reshape(-1, 32, 32, 1)

            data[idx] = {
                'train': (X_train_csv, img1_train, img2_train, y_train, mask),
                'test': (X_test_csv, img1_test, img2_test, y_test, mask)
            }
        except Exception as e:
            print(f"Skipping Client {idx} due to error: {e}")
            continue

    return data

# ==================== CONF-SMOE MODEL COMPONENTS ====================

class SensorEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
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
            nn.Linear(64*8*8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.net(x)

class ImputationModule(nn.Module):
    """
    Two-Stage Imputation:
    1. Cross-Modal Generation: Tries to generate missing modality from available ones.
    2. Expert Guidance: Refined during training via backprop from experts.
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # Simple projection network to map available features to missing feature space
        # Input is concatenation of 3 modalities (some zeroed). Output is 3 imputed embeddings.
        self.imputer = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim * 3)
        )

    def forward(self, h_csv, h_img1, h_img2, mask):
        # Concatenate inputs (zeros where missing)
        fused_input = torch.cat([h_csv, h_img1, h_img2], dim=1) # (B, 3*D)
        
        # Generate imputed features
        imputed_raw = self.imputer(fused_input)
        
        # Split back
        imp_csv, imp_img1, imp_img2 = torch.split(imputed_raw, self.embed_dim, dim=1)
        
        # REPLACE missing with imputed, KEEP present original
        # Mask is 1 for present, 0 for missing
        # If mask=1: use h_csv. If mask=0: use imp_csv.
        
        mask_csv = mask[:, 0].unsqueeze(1)
        mask_img1 = mask[:, 1].unsqueeze(1)
        mask_img2 = mask[:, 2].unsqueeze(1)
        
        final_csv = h_csv * mask_csv + imp_csv * (1 - mask_csv)
        final_img1 = h_img1 * mask_img1 + imp_img1 * (1 - mask_img1)
        final_img2 = h_img2 * mask_img2 + imp_img2 * (1 - mask_img2)
        
        return final_csv, final_img1, final_img2

class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, output_dim) # Output logits for class prediction directly from expert
        )
    def forward(self, x): return self.net(x)

class ConfidenceGuidedGate(nn.Module):
    """
    Paper Innovation:
    1. Detaches routing from softmax.
    2. Scores represent 'Confidence' of expert i for input x.
    3. No load balancing loss required.
    """
    def __init__(self, input_dim, num_experts, k=2):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.k = k

    def forward(self, x):
        # Calculate raw scores
        logits = self.gate(x)
        
        # ConfSMoE Key: Use Sigmoid to get independent confidence scores [0, 1]
        # instead of Softmax which forces sum to 1.
        confidence_scores = torch.sigmoid(logits)
        
        # Select Top-K based on highest confidence
        top_k_vals, top_k_indices = torch.topk(confidence_scores, self.k, dim=1)
        
        # In ConfSMoE, we weight the expert output by its confidence score
        # No re-normalization (softmax) needed usually, or simple normalization
        # Paper suggests using the confidence score directly as the gating weight.
        return top_k_vals, top_k_indices

class ConfSMoEModel(nn.Module):
    def __init__(self, csv_input_dim, num_classes=12):
        super().__init__()
        self.embed_dim = 256
        fusion_dim = self.embed_dim * 3
        
        # Encoders
        self.enc_csv = SensorEncoder(csv_input_dim, self.embed_dim)
        self.enc_img = ImageEncoder(self.embed_dim)
        
        # Imputation
        self.imputer = ImputationModule(self.embed_dim)
        
        # Gating
        self.gate = ConfidenceGuidedGate(fusion_dim, Config.NUM_EXPERTS, Config.TOP_K)
        
        # Experts
        # Each expert takes fused input and predicts classes (or features)
        self.experts = nn.ModuleList([
            Expert(fusion_dim, num_classes) for _ in range(Config.NUM_EXPERTS)
        ])

    def forward(self, x_csv, x_img1, x_img2, mask):
        # 1. Encode (Zero-filled inputs result in zero embeddings mostly)
        h_csv = self.enc_csv(x_csv) * mask[:, 0].unsqueeze(1)
        h_img1 = self.enc_img(x_img1) * mask[:, 1].unsqueeze(1)
        h_img2 = self.enc_img(x_img2) * mask[:, 2].unsqueeze(1)
        
        # 2. Two-Stage Imputation
        # Replaces zeroed embeddings with imputed ones based on cross-modal correlations
        h_c_imp, h_i1_imp, h_i2_imp = self.imputer(h_csv, h_img1, h_img2, mask)
        
        # 3. Fusion
        # Now we have a 'complete' representation even if original data was missing
        x_fused = torch.cat([h_c_imp, h_i1_imp, h_i2_imp], dim=1)
        
        # 4. Confidence-Guided Routing
        # weights: (B, K), indices: (B, K)
        weights, indices = self.gate(x_fused)
        
        # 5. Sparse Dispatch
        final_logits = torch.zeros(x_fused.size(0), Config.NUM_CLASSES, device=x_fused.device)
        
        for k_idx in range(Config.TOP_K):
            k_experts = indices[:, k_idx]
            k_weights = weights[:, k_idx].unsqueeze(1)
            
            # Efficient implementation: Iterate all experts, check mask
            for e_id in range(Config.NUM_EXPERTS):
                batch_mask = (k_experts == e_id)
                if batch_mask.any():
                    inp_subset = x_fused[batch_mask]
                    expert_out = self.experts[e_id](inp_subset)
                    final_logits[batch_mask] += k_weights[batch_mask] * expert_out
        
        return final_logits

# ==================== TRAINING HELPERS ====================

def train_client(global_model, client_data):
    """
    Local training for ConfSMoE.
    No Load Balancing Loss is used (Key Feature of ConfSMoE).
    """
    # Unpack including mask
    X_csv, img1, img2, y, mask = client_data['train']
    dataset = ConfSMoEDataset(X_csv, img1, img2, y, mask)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    epoch_losses = []
    
    for _ in range(Config.LOCAL_EPOCHS):
        batch_losses = []
        for batch in loader:
            c, i1, i2, lbl, m = [x.to(Config.DEVICE) for x in batch]
            optimizer.zero_grad()
            
            # Forward pass with Imputation + Routing
            logits = local_model(c, i1, i2, m)
            
            # Only Classification Loss (ConfSMoE removes aux loss)
            loss = criterion(logits, lbl)
            
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_losses.append(sum(batch_losses)/len(batch_losses))
        
    return local_model.state_dict(), sum(epoch_losses)/len(epoch_losses)

def evaluate_global(model, clients_data):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cid in clients_data:
            # Use test set (which also has masks applied in load_data_heterogeneous)
            X_csv, img1, img2, y, mask = clients_data[cid]['test']
            dataset = ConfSMoEDataset(X_csv, img1, img2, y, mask)
            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)
            
            for batch in loader:
                c, i1, i2, lbl, m = [x.to(Config.DEVICE) for x in batch]
                logits = model(c, i1, i2, m)
                preds = logits.argmax(dim=1)
                correct += preds.eq(lbl).sum().item()
                total += lbl.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def fed_avg(weights_list):
    avg_w = copy.deepcopy(weights_list[0])
    for key in avg_w.keys():
        if avg_w[key].is_floating_point():
            for i in range(1, len(weights_list)):
                avg_w[key] += weights_list[i][key]
            avg_w[key] = torch.div(avg_w[key], len(weights_list))
        else:
            tmp = avg_w[key].float()
            for i in range(1, len(weights_list)):
                tmp += weights_list[i][key].float()
            avg_w[key] = (tmp / len(weights_list)).to(avg_w[key].dtype)
    return avg_w

# ==================== MAIN ====================
if __name__ == "__main__":
    # 1. Load Data
    all_data = load_data_heterogeneous()
    if len(all_data) == 0: exit()
    
    # 2. Initialize ConfSMoE Global Model
    csv_dim = all_data[0]['train'][0].shape[1]
    global_model = ConfSMoEModel(csv_dim, Config.NUM_CLASSES).to(Config.DEVICE)
    
    best_acc = 0.0
    print("\n=== Starting ConfSMoE Federated Training ===")
    print("Feature: Confidence-Guided Gating (No Softmax, No Aux Loss)")
    print("Feature: Two-Stage Imputation for Missing Modalities")
    
    for round_num in range(1, Config.ROUNDS + 1):
        print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")
        
        # Pareto Client Selection (Simplified to Random for standalone script, extensible)
        # To make it Pareto, calculate resource metrics here and filter `selected_clients`
        selected_clients = random.sample(range(Config.TOTAL_CLIENTS), Config.NUM_SELECTED_CLIENTS)
        
        local_weights = []
        for cid in selected_clients:
            w, loss = train_client(global_model, all_data[cid])
            local_weights.append(w)
            print(f"Client {cid} ({Config.CLIENT_PROFILES.get(cid)}) Loss: {loss:.4f}")
        
        # FedAvg Aggregation
        # Aggregates Experts, Gate, and Imputation Module
        global_w = fed_avg(local_weights)
        global_model.load_state_dict(global_w)
        
        # Evaluation
        acc = evaluate_global(global_model, all_data)
        print(f"Global Test Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), "best_confsmoe.pth")
            print("  [Saved Best Model]")

    print(f"\nTraining Complete. Best Accuracy: {best_acc:.2f}%")