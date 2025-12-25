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
from collections import defaultdict

# ==================== CONFIGURATION ====================
class Config:
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset
    SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    TOTAL_CLIENTS = 12
    NUM_CLASSES = 12  # Labels 0-11
    
    # Training
    ROUNDS = 30
    NUM_SELECTED_CLIENTS = 6
    LOCAL_EPOCHS = 3
    BATCH_SIZE = 64
    LR = 0.001
    
    # FedProto Hyperparameters
    LAMBDA_PROTO = 1.0  # Weight for prototype regularization loss
    EMBED_DIM = 256     # Dimension of the feature embedding used for prototypes

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

# ==================== DATASET ====================
class UPFallDataset(Dataset):
    def __init__(self, csv_data, img1_data, img2_data, labels):
        self.csv = torch.tensor(csv_data, dtype=torch.float32)
        # Images loaded as (N, 32, 32, 1), model expects (N, C, H, W)
        # We will handle permutation in the model or dataset. Doing it here is cleaner.
        self.img1 = torch.tensor(img1_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.img2 = torch.tensor(img2_data, dtype=torch.float32).permute(0, 3, 1, 2)
        self.labels = torch.tensor(labels, dtype=torch.long).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.csv[idx], self.img1[idx], self.img2[idx], self.labels[idx]

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

# ==================== FEDPROTO MODEL ====================
class MultimodalFedProtoModel(nn.Module):
    """
    Standard backbone modified to return features (embeddings) 
    alongside logits for Prototype Learning.
    """
    def __init__(self, num_csv_features, num_classes=12, embed_dim=256):
        super().__init__()
        
        # CSV Branch
        self.csv_net = nn.Sequential(
            nn.Linear(num_csv_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        # Image Branch (Shared Encoder for both images)
        self.img_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8x8
            
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Fusion & Embedding Head
        # We project the concatenated features to the prototype dimension (EMBED_DIM)
        self.fusion_fc = nn.Sequential(
            nn.Linear(256 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim) # This is the Feature Embedding (Z)
        )
        
        # Classifier
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x_csv, x_img1, x_img2):
        # Encoders
        h_csv = self.csv_net(x_csv)
        h_img1 = self.img_net(x_img1)
        h_img2 = self.img_net(x_img2)
        
        # Fusion
        concat = torch.cat((h_csv, h_img1, h_img2), dim=1)
        
        # Feature Embedding (for Prototypes)
        embedding = self.fusion_fc(concat)
        
        # Logits (for Classification)
        logits = self.classifier(embedding)
        
        return logits, embedding

# ==================== PROTOTYPE HELPERS ====================

def compute_local_prototypes(model, dataloader, device):
    """
    Computes local prototypes (mean feature vector per class) for the client.
    Returns: Dict {class_index: prototype_tensor}
    """
    model.eval()
    
    # Store features per class
    # sums: {class: sum_of_features}
    # counts: {class: count}
    sums = defaultdict(lambda: torch.zeros(Config.EMBED_DIM).to(device))
    counts = defaultdict(int)
    
    with torch.no_grad():
        for batch in dataloader:
            csv, i1, i2, y = [x.to(device) for x in batch]
            _, features = model(csv, i1, i2)
            
            for i in range(len(y)):
                label = y[i].item()
                sums[label] += features[i]
                counts[label] += 1
                
    # Calculate means
    prototypes = {}
    for label, total_feature in sums.items():
        if counts[label] > 0:
            prototypes[label] = (total_feature / counts[label]).cpu() # Move to CPU for upload
            
    return prototypes

def aggregate_global_prototypes(client_protos_list):
    """
    Aggregates local prototypes into global prototypes.
    Handles missing classes (not all clients have all classes).
    
    Args:
        client_protos_list: List of dicts [{class: proto}, ...]
        
    Returns:
        global_protos: Dict {class: global_proto_tensor}
    """
    global_sums = defaultdict(lambda: torch.zeros(Config.EMBED_DIM))
    global_counts = defaultdict(int)
    
    for client_protos in client_protos_list:
        for label, proto in client_protos.items():
            global_sums[label] += proto
            global_counts[label] += 1
            
    global_protos = {}
    for label, total_feature in global_sums.items():
        global_protos[label] = total_feature / global_counts[label]
        
    return global_protos

# ==================== TRAINING HELPERS ====================

def train_client_fedproto(model, train_data, global_protos):
    """
    Local training with Prototype Regularization.
    L = L_CE + lambda * L_Proto
    L_Proto = MSE(feature, global_proto[label])
    """
    X_csv, img1, img2, y = train_data
    dataset = UPFallDataset(X_csv, img1, img2, y)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)
    criterion = nn.CrossEntropyLoss()
    
    epoch_loss = 0.0
    
    for _ in range(Config.LOCAL_EPOCHS):
        batch_losses = []
        for batch in loader:
            csv, i1, i2, labels = [x.to(Config.DEVICE) for x in batch]
            
            optimizer.zero_grad()
            
            logits, features = model(csv, i1, i2)
            
            # 1. Classification Loss
            loss_ce = criterion(logits, labels)
            
            # 2. Prototype Loss (Regularization)
            loss_proto = 0.0
            
            if global_protos: # If global prototypes exist (Round > 1)
                # Create a tensor of target prototypes for this batch
                # We need to map labels to their global prototypes
                # Note: Some classes might not have global prototypes yet (if never seen globally)
                
                proto_targets = []
                valid_indices = []
                
                for idx, label in enumerate(labels):
                    l = label.item()
                    if l in global_protos:
                        proto_targets.append(global_protos[l].to(Config.DEVICE))
                        valid_indices.append(idx)
                        
                if len(valid_indices) > 0:
                    # Stack valid prototypes
                    targets = torch.stack(proto_targets)
                    # Get corresponding features
                    valid_features = features[valid_indices]
                    
                    # MSE Loss between feature and its class prototype
                    loss_proto = F.mse_loss(valid_features, targets)
            
            # Total Loss
            total_loss = loss_ce + (Config.LAMBDA_PROTO * loss_proto)
            
            total_loss.backward()
            optimizer.step()
            batch_losses.append(total_loss.item())
            
        epoch_loss += sum(batch_losses) / len(batch_losses)
        
    # After training, re-compute local prototypes based on updated model
    # We use a non-shuffled loader for consistent computation if needed, but mean is invariant
    new_local_protos = compute_local_prototypes(model, loader, Config.DEVICE)
    
    return model.state_dict(), new_local_protos, epoch_loss / Config.LOCAL_EPOCHS

def evaluate(model, clients_data):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cid in clients_data:
            X_csv, img1, img2, y = clients_data[cid]['test']
            dataset = UPFallDataset(X_csv, img1, img2, y)
            loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE)
            
            for batch in loader:
                csv, i1, i2, labels = [x.to(Config.DEVICE) for x in batch]
                logits, _ = model(csv, i1, i2)
                preds = logits.argmax(dim=1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
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
    global_model = MultimodalFedProtoModel(csv_dim, Config.NUM_CLASSES, Config.EMBED_DIM).to(Config.DEVICE)
    
    # 3. Init Global Prototypes (Empty at start)
    # Dictionary mapping Class Index -> Tensor
    global_prototypes = {} 
    
    best_acc = 0.0
    print("\n=== Starting FedProto Training ===")
    
    for round_num in range(1, Config.ROUNDS + 1):
        print(f"\n--- Round {round_num}/{Config.ROUNDS} ---")
        
        # Client Selection
        selected_clients = random.sample(range(Config.TOTAL_CLIENTS), Config.NUM_SELECTED_CLIENTS)
        
        local_weights = []
        local_protos_list = []
        
        for cid in selected_clients:
            # Send Global Model and Global Prototypes to Client
            # Client trains and returns updated weights + New Local Prototypes
            w, protos, loss = train_client_fedproto(
                copy.deepcopy(global_model), 
                all_data[cid]['train'], 
                global_prototypes
            )
            
            local_weights.append(w)
            local_protos_list.append(protos)
            
            print(f"Client {cid} Loss: {loss:.4f} | Protos Generated: {list(protos.keys())}")
        
        # 1. Aggregation of Model Weights (FedAvg)
        global_w = fed_avg(local_weights)
        global_model.load_state_dict(global_w)
        
        # 2. Aggregation of Prototypes (FedProto)
        # Update global prototypes based on the new local prototypes
        # Note: In standard FedProto, global protos are replaced or moved by momentum. 
        # Here we recalculate the mean of uploaded protos for simplicity and stability.
        new_global_protos = aggregate_global_prototypes(local_protos_list)
        
        # Optional: Momentum update for stability (Global = 0.5*Old + 0.5*New)
        if round_num > 1:
            for k in new_global_protos:
                if k in global_prototypes:
                    global_prototypes[k] = 0.5 * global_prototypes[k] + 0.5 * new_global_protos[k]
                else:
                    global_prototypes[k] = new_global_protos[k]
        else:
            global_prototypes = new_global_protos
            
        print(f"Global Prototypes Updated. Classes covered: {sorted(global_prototypes.keys())}")
        
        # Evaluation
        acc = evaluate(global_model, all_data)
        print(f"Global Test Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), "best_fedproto_model.pth")
            
    print(f"\nTraining Finished. Best Accuracy: {best_acc:.2f}%")