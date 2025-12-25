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
    
    # Missing Modality Simulation Masks [Sensor, Img1, Img2]
    # 1 = Present, 0 = Missing (Zero-filled)
    CLIENT_MASKS = {
        0: [1, 1, 1], 1: [1, 1, 1], 2: [1, 1, 1], 3: [1, 1, 1],  # Full (Proto Generators)
        4: [1, 0, 0], 5: [1, 0, 0], 6: [1, 0, 0], 7: [1, 0, 0],  # Sensor Only
        8: [0, 1, 1], 9: [0, 1, 1], 10: [0, 1, 1], 11: [0, 1, 1] # Images Only
    }
    
    # Training
    ROUNDS = 30
    NUM_SELECTED_CLIENTS = 6
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 0.001
    
    # MFCPL Hyperparameters
    LAMBDA_REG = 0.5   # Weight for Prototype Regularization
    LAMBDA_ALIGN = 0.1 # Weight for Cross-Modal Alignment
    LAMBDA_CON = 0.1   # Weight for Contrastive Loss
    TEMP = 0.07        # Temperature for Contrastive Loss

# ==================== UTILS ====================
def set_seed(seed=Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# ==================== DATASET ====================
class MFCPLDataset(Dataset):
    def __init__(self, csv_data, img1_data, img2_data, labels, mask=[1, 1, 1]):
        self.csv = torch.tensor(csv_data, dtype=torch.float32)
        self.img1 = torch.tensor(img1_data, dtype=torch.float32)
        self.img2 = torch.tensor(img2_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).reshape(-1)
        self.mask = torch.tensor(mask, dtype=torch.float32) # [1, 1, 1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Apply Zero-Filling based on mask
        c = self.csv[idx] * self.mask[0]
        i1 = self.img1[idx] * self.mask[1]
        i2 = self.img2[idx] * self.mask[2]
        return c, i1, i2, self.labels[idx], self.mask

# ==================== DATA LOADING ====================
def load_data():
    data_store = {}
    print(f"Loading data from ./dataset/Sensor + Image/ ...")
    
    for idx, sub in enumerate(Config.SUBJECTS):
        try:
            # --- Load Train ---
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
            csv_tr_scaled = scaler.fit_transform(csv_tr[:, :-1])
            y_tr = lbl_tr.astype(np.int64)
            img1_tr = img1_tr.reshape(-1, 32, 32, 1)
            img2_tr = img2_tr.reshape(-1, 32, 32, 1)

            # --- Load Test ---
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

            csv_te_scaled = scaler.transform(csv_te[:, :-1])
            y_te = lbl_te.astype(np.int64)
            img1_te = img1_te.reshape(-1, 32, 32, 1)
            img2_te = img2_te.reshape(-1, 32, 32, 1)

            data_store[idx] = {
                'train': (csv_tr_scaled, img1_tr, img2_tr, y_tr),
                'test': (csv_te_scaled, img1_te, img2_te, y_te)
            }
        except Exception as e:
            print(f"Skipping Subject {sub}: {e}")
    return data_store

# ==================== MFCPL MODEL ARCHITECTURE ====================

class SensorEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Linear(256, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32*8*8, embed_dim), nn.BatchNorm1d(embed_dim), nn.ReLU()
        )
    def forward(self, x):
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        return self.net(x)

class MFCPLModel(nn.Module):
    def __init__(self, csv_dim, num_classes=12):
        super().__init__()
        self.embed_dim = 128
        
        # 1. Encoders
        self.enc_csv = SensorEncoder(csv_dim, self.embed_dim)
        self.enc_img1 = ImageEncoder(self.embed_dim)
        self.enc_img2 = ImageEncoder(self.embed_dim)
        
        # 2. Projectors (Paper: Shared for Proto Reg, Specific for Contrastive)
        self.proj_shared = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        # Specific projectors (one per modality logic)
        self.proj_spec_csv = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_spec_img1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.proj_spec_img2 = nn.Linear(self.embed_dim, self.embed_dim)
        
        # 3. Fusion & Classifier (Attention-based Fusion to handle zeros better)
        self.fusion_att = nn.Linear(self.embed_dim, 1)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x_csv, x_img1, x_img2, mask):
        # Mask: [Batch, 3]
        
        # Encoding
        # Note: If input is all zeros (missing), encoder output might not be exactly zero due to biases.
        # We enforce zeroing out features of missing modalities after encoding.
        
        h_csv = self.enc_csv(x_csv) * mask[:, 0].unsqueeze(1)
        h_img1 = self.enc_img1(x_img1) * mask[:, 1].unsqueeze(1)
        h_img2 = self.enc_img2(x_img2) * mask[:, 2].unsqueeze(1)
        
        # --- Shared Projections (For L_reg and L_align) ---
        z_shared_csv = self.proj_shared(h_csv) * mask[:, 0].unsqueeze(1)
        z_shared_img1 = self.proj_shared(h_img1) * mask[:, 1].unsqueeze(1)
        z_shared_img2 = self.proj_shared(h_img2) * mask[:, 2].unsqueeze(1)
        
        # --- Specific Projections (For L_con) ---
        z_spec_csv = self.proj_spec_csv(h_csv) * mask[:, 0].unsqueeze(1)
        z_spec_img1 = self.proj_spec_img1(h_img1) * mask[:, 1].unsqueeze(1)
        z_spec_img2 = self.proj_spec_img2(h_img2) * mask[:, 2].unsqueeze(1)
        
        # --- Fusion for Classification ---
        # Simple Attention: sum(w_m * h_m) / sum(w_m)
        # We assume missing modalities have 0 features.
        # We construct a robust average.
        
        stack_feats = torch.stack([h_csv, h_img1, h_img2], dim=1) # (B, 3, D)
        # Count available modalities
        avail_count = mask.sum(dim=1, keepdim=True).clamp(min=1.0) # (B, 1)
        
        fused_feat = stack_feats.sum(dim=1) / avail_count # Mean of available features
        
        logits = self.classifier(fused_feat)
        
        return logits, {
            'shared': [z_shared_csv, z_shared_img1, z_shared_img2],
            'specific': [z_spec_csv, z_spec_img1, z_spec_img2],
            'fused': fused_feat
        }

# ==================== PROTOTYPE BANK ====================
class PrototypeBank:
    def __init__(self, num_classes, embed_dim, device):
        self.prototypes = torch.zeros(num_classes, embed_dim).to(device)
        self.num_classes = num_classes
        self.device = device
        self.counts = torch.zeros(num_classes).to(device)

    def clear(self):
        self.prototypes.fill_(0)
        self.counts.fill_(0)

    def update(self, embeddings, labels):
        # embeddings: (B, D) - typically fused or shared representation from full clients
        # labels: (B,)
        with torch.no_grad():
            for c in range(self.num_classes):
                mask = (labels == c)
                if mask.sum() > 0:
                    class_feats = embeddings[mask]
                    mean_feat = class_feats.mean(dim=0)
                    self.prototypes[c] += mean_feat
                    self.counts[c] += 1

    def average_and_normalize(self):
        # Finalize prototypes for the round
        mask = self.counts > 0
        self.prototypes[mask] /= self.counts[mask].unsqueeze(1)
        # Normalize for cosine similarity
        self.prototypes = F.normalize(self.prototypes, dim=1)
        return self.prototypes.clone()

# ==================== LOSS FUNCTIONS ====================

def mfcpl_loss(model_out, labels, mask, prototypes, config):
    """
    Computes total MFCPL loss: L_cls + L_reg + L_align + L_con
    """
    logits, feats = model_out
    shared_feats = feats['shared'] # [csv, img1, img2]
    spec_feats = feats['specific']
    
    # 1. Classification Loss
    criterion = nn.CrossEntropyLoss()
    l_cls = criterion(logits, labels)
    
    # 2. Cross-Modal Regularization (L_reg)
    # Distance between Shared Projection of *Available* Modalities and Global Prototype
    l_reg = 0.0
    valid_terms = 0
    
    # Normalize shared feats for cosine distance logic usually, or just MSE
    # Paper typically uses MSE or Cosine. We use MSE against normalized Prototypes.
    
    for m_idx in range(3): # Iterate modalities
        # Get active samples for this modality
        active_mask = mask[:, m_idx].bool()
        if active_mask.sum() > 0:
            active_preds = shared_feats[m_idx][active_mask] # (N_active, D)
            active_labels = labels[active_mask]
            
            # Get corresponding prototypes
            target_protos = prototypes[active_labels] # (N_active, D)
            
            # MSE between projected feature and prototype
            l_reg += F.mse_loss(active_preds, target_protos)
            valid_terms += 1
            
    if valid_terms > 0:
        l_reg /= valid_terms

    # 3. Cross-Modal Alignment (L_align)
    # Alignment between shared representations of different modalities for same sample
    l_align = 0.0
    pairs = [(0, 1), (0, 2), (1, 2)] # (CSV, Img1), (CSV, Img2), (Img1, Img2)
    align_count = 0
    
    for (m1, m2) in pairs:
        # Both modalities must be present
        both_active = (mask[:, m1] == 1) & (mask[:, m2] == 1)
        if both_active.sum() > 0:
            feat1 = shared_feats[m1][both_active]
            feat2 = shared_feats[m2][both_active]
            l_align += F.mse_loss(feat1, feat2)
            align_count += 1
            
    if align_count > 0:
        l_align /= align_count

    # 4. Contrastive Loss (L_con)
    # Supervised Contrastive Loss on Specific Features
    # Pushes specific features of same class together, different classes apart
    l_con = 0.0
    con_count = 0
    
    # Collect all valid specific features into a batch
    # This is a simplified SupCon approach within the batch
    
    all_feats = []
    all_labels = []
    
    for m_idx in range(3):
        active_mask = mask[:, m_idx].bool()
        if active_mask.sum() > 0:
            all_feats.append(spec_feats[m_idx][active_mask])
            all_labels.append(labels[active_mask])
            
    if len(all_feats) > 0:
        cat_feats = torch.cat(all_feats, dim=0)
        cat_labels = torch.cat(all_labels, dim=0)
        
        # Normalize
        cat_feats = F.normalize(cat_feats, dim=1)
        
        # Similarity Matrix
        sim_matrix = torch.matmul(cat_feats, cat_feats.T) / Config.TEMP
        
        # Mask for same class (positive pairs)
        # Avoid self-contrast
        labels_matrix = cat_labels.unsqueeze(0) == cat_labels.unsqueeze(1)
        identity_mask = torch.eye(labels_matrix.shape[0], device=Config.DEVICE).bool()
        labels_matrix = labels_matrix & (~identity_mask) # Exclude self
        
        # SupCon Logic
        exp_sim = torch.exp(sim_matrix)
        # Denominator: Sum over all negatives and positives (except self)
        denom = exp_sim.sum(dim=1) 
        
        # Numerator: Sum over positives
        # Compute log prob for each positive pair, then average
        # Handle cases where a sample has no other positive in batch
        has_pos = labels_matrix.sum(dim=1) > 0
        
        if has_pos.sum() > 0:
            log_probs = (sim_matrix * labels_matrix.float()).sum(dim=1)[has_pos] / labels_matrix.sum(dim=1)[has_pos]
            # Actually standard SupCon formulation:
            # L = - sum(log(exp(pos) / sum(exp(all))))
            # Simplified:
            loss_partial = -torch.log( exp_sim[has_pos] / denom[has_pos].unsqueeze(1) + 1e-8 )
            # Apply mask again to only sum over positives
            l_con = (loss_partial * labels_matrix[has_pos].float()).sum(dim=1) / labels_matrix[has_pos].sum(dim=1)
            l_con = l_con.mean()

    # Total Loss
    total_loss = l_cls + (config.LAMBDA_REG * l_reg) + (config.LAMBDA_ALIGN * l_align) + (config.LAMBDA_CON * l_con)
    return total_loss, l_cls.item(), l_reg, l_align, l_con

# ==================== TRAINING HELPERS ====================

def train_client(model, client_data, client_mask, global_protos):
    """
    Local training with MFCPL losses and missing modality mask.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    
    X_csv, X1, X2, y = client_data['train']
    # Pass Mask to Dataset
    dataset = MFCPLDataset(X_csv, X1, X2, y, mask=client_mask)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    local_protos = PrototypeBank(Config.NUM_CLASSES, 128, Config.DEVICE)
    epoch_loss = 0.0
    
    for _ in range(Config.LOCAL_EPOCHS):
        for batch in loader:
            c, i1, i2, lbl, m = [x.to(Config.DEVICE) for x in batch]
            optimizer.zero_grad()
            
            # Forward
            out_logits, out_feats = model(c, i1, i2, m)
            
            # Loss Calculation (Needs Global Prototypes)
            loss, l_c, l_r, l_a, l_con = mfcpl_loss((out_logits, out_feats), lbl, m, global_protos, Config)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Accumulate Local Prototypes (If this client has full modalities)
            # Only full modality clients (Mask=[1,1,1]) contribute to global prototypes
            # We use the 'fused' feature for prototype generation
            if client_mask == [1, 1, 1]:
                local_protos.update(out_feats['fused'], lbl)
                
    # Return updated weights and the locally computed prototypes
    return model.state_dict(), epoch_loss / len(loader), local_protos

def evaluate(model, clients_data):
    """
    Evaluate on clean, full-modality test sets of ALL clients.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cid in clients_data:
            X_csv, X1, X2, y = clients_data[cid]['test']
            # Test is always full modality for valid evaluation
            loader = DataLoader(MFCPLDataset(X_csv, X1, X2, y, mask=[1,1,1]), batch_size=Config.BATCH_SIZE)
            for batch in loader:
                c, i1, i2, lbl, m = [x.to(Config.DEVICE) for x in batch]
                logits, _ = model(c, i1, i2, m)
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
    all_data = load_data()
    if len(all_data) == 0: exit()
    
    # 2. Init Model
    csv_dim = all_data[0]['train'][0].shape[1]
    global_model = MFCPLModel(csv_dim, Config.NUM_CLASSES).to(Config.DEVICE)
    
    # 3. Init Global Prototype Bank
    # We need an initial set of prototypes. 
    # Option: Random init or pre-train. 
    # For standalone script: Random init normalized.
    global_prototype_bank = torch.randn(Config.NUM_CLASSES, 128).to(Config.DEVICE)
    global_prototype_bank = F.normalize(global_prototype_bank, dim=1)
    
    best_acc = 0.0
    print("\n=== Starting MFCPL Federated Training ===")
    print(f"Missing Modality Setup:\n Clients 0-3: Full\n Clients 4-7: Sensor Only\n Clients 8-11: Images Only")
    
    for round_num in range(1, Config.ROUNDS + 1):
        print(f"\n--- Round {round_num} ---")
        
        selected_clients = random.sample(range(Config.TOTAL_CLIENTS), Config.NUM_SELECTED_CLIENTS)
        local_weights = []
        round_protos_list = [] # List of prototypes from full-modality clients
        
        for cid in selected_clients:
            mask = Config.CLIENT_MASKS[cid]
            # Train
            w, loss, c_protos = train_client(
                copy.deepcopy(global_model), 
                all_data[cid], 
                mask, 
                global_prototype_bank
            )
            local_weights.append(w)
            print(f"Client {cid} (Mask {mask}) Loss: {loss:.4f}")
            
            # If client generated prototypes (was full modality), collect them
            # We check if the prototype bank in c_protos has counts > 0
            if c_protos.counts.sum() > 0:
                round_protos_list.append(c_protos.average_and_normalize())
        
        # Aggregation
        global_w = fed_avg(local_weights)
        global_model.load_state_dict(global_w)
        
        # Update Global Prototypes
        # Average the prototypes received from full-modality clients this round
        if len(round_protos_list) > 0:
            stack_protos = torch.stack(round_protos_list) # (N_contributors, Classes, D)
            new_global_protos = stack_protos.mean(dim=0)
            # Update with momentum or replacement. Paper implies replacement/update.
            # We use moving average for stability
            global_prototype_bank = 0.5 * global_prototype_bank + 0.5 * new_global_protos
            global_prototype_bank = F.normalize(global_prototype_bank, dim=1)
        
        # Evaluate
        acc = evaluate(global_model, all_data)
        print(f"Global Test Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), "best_mfcpl_model.pth")
            
    print(f"\nTraining Finished. Best Accuracy: {best_acc:.2f}%")