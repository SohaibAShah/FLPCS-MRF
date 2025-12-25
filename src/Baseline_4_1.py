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

# ==================== SEED & DEVICE ====================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== CONFIG ====================
ROUNDS = 30
NUM_SELECTED_CLIENTS = 6
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
TOTAL_CLIENTS = 12
SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
FEDPROX_MU = 0.01  # Hyperparameter for FedProx

# Missing Modality Profiles
# 0-3: Full, 4-7: No Images (Sensor Only), 8-11: No Sensor (Image Only)
CLIENT_PROFILES = {
    0: 'Full', 1: 'Full', 2: 'Full', 3: 'Full',
    4: 'Sensor_Only', 5: 'Sensor_Only', 6: 'Sensor_Only', 7: 'Sensor_Only',
    8: 'Image_Only', 9: 'Image_Only', 10: 'Image_Only', 11: 'Image_Only'
}

# ==================== DATASET ====================
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

# ==================== MODEL ====================
class ModelCSVIMG(nn.Module):
    def __init__(self, num_csv_features):
        super(ModelCSVIMG, self).__init__()
        # CSV Branch
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)

        # Image Branches
        self.img_conv = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.img_bn = nn.BatchNorm2d(18)
        self.img_pool = nn.MaxPool2d(2)
        self.img_fc = nn.Linear(18 * 16 * 16, 100)
        self.img_dropout = nn.Dropout(0.2)

        # Fusion
        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2000, 12)

    def forward(self, x_csv, x_img1, x_img2):
        # CSV
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)

        # Img1
        x_img1 = x_img1.permute(0, 3, 1, 2)
        x_img1 = F.relu(self.img_bn(self.img_conv(x_img1)))
        x_img1 = self.img_pool(x_img1)
        x_img1 = x_img1.reshape(x_img1.size(0), -1)
        x_img1 = F.relu(self.img_fc(x_img1))
        x_img1 = self.img_dropout(x_img1)

        # Img2
        x_img2 = x_img2.permute(0, 3, 1, 2)
        x_img2 = F.relu(self.img_bn(self.img_conv(x_img2)))
        x_img2 = self.img_pool(x_img2)
        x_img2 = x_img2.reshape(x_img2.size(0), -1)
        x_img2 = F.relu(self.img_fc(x_img2))
        x_img2 = self.img_dropout(x_img2)

        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = torch.cat((residual, x), dim=1)
        return self.fc2(x)

# ==================== DATA LOADING WITH HETEROGENEITY ====================
def load_clients_data_heterogeneous():
    data = {}
    print("\n=== Loading Data & Simulating Modality Heterogeneity ===")
    print("Clients 0-3: Full | 4-7: No Images | 8-11: No Sensor")

    for idx, sub in enumerate(SUBJECTS):
        profile = CLIENT_PROFILES.get(idx, 'Full')
        
        try:
            # --- Load Raw Data ---
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

            X_train_csv = train_data[:, :-1]
            y_train = label_train.astype(np.int64)
            scaler = StandardScaler()
            X_train_csv = scaler.fit_transform(X_train_csv)

            img1_train = img1_train.reshape(-1, 32, 32, 1)
            img2_train = img2_train.reshape(-1, 32, 32, 1)

            # Test
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
                img1_train = np.zeros_like(img1_train)
                img2_train = np.zeros_like(img2_train)
                img1_test = np.zeros_like(img1_test)
                img2_test = np.zeros_like(img2_test)
            
            elif profile == 'Image_Only':
                # Zero out sensor CSV
                X_train_csv = np.zeros_like(X_train_csv)
                X_test_csv = np.zeros_like(X_test_csv)
            # ============================================================

            data[idx] = {
                'train': (X_train_csv, img1_train, img2_train, y_train),
                'test': (X_test_csv, img1_test, img2_test, y_test)
            }
        except Exception as e:
            print(f"Error loading Subject {sub}: {e}")
            continue

    return data

# ==================== HELPERS ====================
def evaluate_global_model(model, clients_data):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for cid in clients_data:
            X_csv, X1, X2, y = clients_data[cid]['test']
            loader = DataLoader(CustomDataset(X_csv, X1, X2, y), batch_size=BATCH_SIZE)
            for batch in loader:
                csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                out = model(csv_b, img1_b, img2_b)
                pred = out.argmax(dim=1)
                correct += pred.eq(y_b).sum().item()
                total += y_b.size(0)
    return 100.0 * correct / total if total > 0 else 0.0

def train_client_fedavg_prox(global_model, client_data, use_prox=False, mu=0.0):
    X_csv, X1, X2, y = client_data['train']
    train_loader = DataLoader(CustomDataset(X_csv, X1, X2, y), batch_size=BATCH_SIZE, shuffle=True)
    
    local_model = copy.deepcopy(global_model)
    local_model.train()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    global_params = list(global_model.parameters())

    for _ in range(LOCAL_EPOCHS):
        for batch in train_loader:
            csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
            optimizer.zero_grad()
            out = local_model(csv_b, img1_b, img2_b)
            loss = criterion(out, y_b)
            
            if use_prox:
                prox_loss = 0.0
                for p, g_p in zip(local_model.parameters(), global_params):
                    prox_loss += (mu / 2) * torch.norm(p - g_p)**2
                loss += prox_loss
            
            loss.backward()
            optimizer.step()
            
    local_update = {}
    for name, param in local_model.state_dict().items():
        local_update[name] = param - global_model.state_dict()[name]
        
    return local_update

# ==================== FEDERATED STRATEGIES ====================

def run_fedavg(clients_data, initial_model):
    print("\n=== Running FedAvg (Hetero) ===")
    global_model = copy.deepcopy(initial_model)
    acc_history = []
    
    for round_num in range(1, ROUNDS + 1):
        selected_clients = random.sample(range(len(clients_data)), NUM_SELECTED_CLIENTS)
        client_updates = []
        
        for cid in selected_clients:
            update = train_client_fedavg_prox(global_model, clients_data[cid], use_prox=False)
            client_updates.append(update)
            
        global_dict = global_model.state_dict()
        for name in global_dict:
            updates_stack = torch.stack([up[name] for up in client_updates])
            if updates_stack.is_floating_point():
                global_dict[name].add_(updates_stack.mean(dim=0))
            else:
                global_dict[name].add_(updates_stack.float().mean(dim=0).to(updates_stack.dtype))
        global_model.load_state_dict(global_dict)
        
        acc = evaluate_global_model(global_model, clients_data)
        acc_history.append(acc)
        print(f"Round {round_num}: {acc:.2f}%")
        
    return acc_history

def run_fedprox(clients_data, initial_model, mu=0.01):
    print(f"\n=== Running FedProx (Hetero, mu={mu}) ===")
    global_model = copy.deepcopy(initial_model)
    acc_history = []
    
    for round_num in range(1, ROUNDS + 1):
        selected_clients = random.sample(range(len(clients_data)), NUM_SELECTED_CLIENTS)
        client_updates = []
        
        for cid in selected_clients:
            update = train_client_fedavg_prox(global_model, clients_data[cid], use_prox=True, mu=mu)
            client_updates.append(update)
            
        global_dict = global_model.state_dict()
        for name in global_dict:
            updates_stack = torch.stack([up[name] for up in client_updates])
            if updates_stack.is_floating_point():
                global_dict[name].add_(updates_stack.mean(dim=0))
            else:
                global_dict[name].add_(updates_stack.float().mean(dim=0).to(updates_stack.dtype))
        global_model.load_state_dict(global_dict)
        
        acc = evaluate_global_model(global_model, clients_data)
        acc_history.append(acc)
        print(f"Round {round_num}: {acc:.2f}%")
        
    return acc_history

def pareto_optimization(metrics, num_clients):
    data = np.array(metrics).T 
    def is_dominated(i):
        return any(np.all(data[:, j] >= data[:, i]) and np.any(data[:, j] > data[:, i]) 
                   for j in range(data.shape[1]) if j != i)
    pareto_front = [i for i in range(data.shape[1]) if not is_dominated(i)]
    
    if len(pareto_front) >= num_clients:
        return random.sample(pareto_front, num_clients)
    
    scores = (0.3 * data[0] + 0.3 * data[3] + 0.2 * data[1] + 0.2 * data[2])
    candidates = np.argsort(scores)[-num_clients:]
    selected = list(set(pareto_front) | set(candidates))
    return selected[:num_clients]

def run_pareto_fl(clients_data, initial_model):
    print("\n=== Running Pareto FL (Hetero) ===")
    global_model = copy.deepcopy(initial_model)
    acc_history = []
    
    for round_num in range(1, ROUNDS + 1):
        client_metrics = []
        all_client_updates = {}
        
        for cid in range(len(clients_data)):
            X_csv_tr, X1_tr, X2_tr, y_tr = clients_data[cid]['train']
            train_loader = DataLoader(CustomDataset(X_csv_tr, X1_tr, X2_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
            
            X_csv_te, X1_te, X2_te, y_te = clients_data[cid]['test']
            val_loader = DataLoader(CustomDataset(X_csv_te, X1_te, X2_te, y_te), batch_size=BATCH_SIZE)
            
            local_model = copy.deepcopy(global_model)
            local_model.train()
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            losses = []
            for _ in range(LOCAL_EPOCHS):
                epoch_loss = 0.0
                for batch in train_loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    optimizer.zero_grad()
                    out = local_model(csv_b, img1_b, img2_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))
                
            local_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    out = local_model(csv_b, img1_b, img2_b)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(y_b).sum().item()
                    total += y_b.size(0)
            local_acc = 100.0 * correct / total
            
            global_acc = evaluate_global_model(local_model, clients_data)
            
            rf_loss_red = losses[0] - losses[-1] if len(losses) > 1 else 0.0
            p_loss = losses[-1]
            p_bias = abs(local_acc - global_acc) / 100.0
            
            client_metrics.append([rf_loss_red, local_acc/100.0, local_acc/100.0, global_acc/100.0, p_loss, p_bias])
            
            local_update = {}
            for name, param in local_model.state_dict().items():
                local_update[name] = param - global_model.state_dict()[name]
            all_client_updates[cid] = local_update
            
        selected_clients = pareto_optimization(client_metrics, NUM_SELECTED_CLIENTS)
        
        global_dict = global_model.state_dict()
        for name in global_dict:
            updates_stack = torch.stack([all_client_updates[cid][name] for cid in selected_clients])
            if updates_stack.is_floating_point():
                global_dict[name].add_(updates_stack.mean(dim=0))
            else:
                global_dict[name].add_(updates_stack.float().mean(dim=0).to(updates_stack.dtype))
        global_model.load_state_dict(global_dict)
        
        acc = evaluate_global_model(global_model, clients_data)
        acc_history.append(acc)
        print(f"Round {round_num}: {acc:.2f}% (Selected: {selected_clients})")
        
    return acc_history

# ==================== MAIN ====================
if __name__ == "__main__":
    # Use Heterogeneous Loader
    clients_data = load_clients_data_heterogeneous()
    
    if len(clients_data) == 0:
        print("CRITICAL ERROR: No data loaded. Check paths and files.")
    else:
        # Determine CSV dim from first client
        csv_dim = clients_data[0]['train'][0].shape[1]
        
        # Initialize one common start point
        base_model = ModelCSVIMG(csv_dim).to(device)
        
        # 1. Pareto (Hetero)
        pareto_results = run_pareto_fl(clients_data, base_model)
        
        # 2. FedAvg (Hetero)
        fedavg_results = run_fedavg(clients_data, base_model)
        
        # 3. FedProx (Hetero)
        fedprox_results = run_fedprox(clients_data, base_model, mu=FEDPROX_MU)
        
        # Print Final Comparison Table
        print("\n" + "="*50)
        print(f"{'Round':<5} | {'Pareto':<10} | {'FedAvg':<10} | {'FedProx':<10}")
        print("-" * 50)
        for r in range(ROUNDS):
            print(f"{r+1:<5} | {pareto_results[r]:.2f}%{' '*4} | {fedavg_results[r]:.2f}%{' '*4} | {fedprox_results[r]:.2f}%")
        print("-" * 50)
        print(f"Max   | {max(pareto_results):.2f}%{' '*4} | {max(fedavg_results):.2f}%{' '*4} | {max(fedprox_results):.2f}%")