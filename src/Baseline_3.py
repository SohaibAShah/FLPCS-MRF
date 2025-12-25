import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

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
ROUNDS = 100
NUM_SELECTED_CLIENTS = 6
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
TOTAL_CLIENTS = 12
SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]

# Training-time modality dropout probability (robustness)
TRAIN_DROP_PROB = 0.3

# Define client modality types (heterogeneity)
# Clients 0-5: Full multimodal
# Clients 6-8: Missing sensor data (only images)
# Clients 9-11: Missing one camera (only sensor + one image)
CLIENT_MODALITY_TYPE = {
    0: "full", 1: "full", 2: "full", 3: "full", 4: "full", 5: "full",
    6: "no_sensor", 7: "no_sensor", 8: "no_sensor",
    9: "partial_img", 10: "partial_img", 11: "partial_img"
}

# ==================== DATASET ====================
class CustomDataset(Dataset):
    def __init__(self, csv_features, img1_features, img2_features, labels):
        self.csv = torch.tensor(csv_features, dtype=torch.float32)
        self.img1 = torch.tensor(img1_features, dtype=torch.float32)
        self.img2 = torch.tensor(img2_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).flatten()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.csv[idx], self.img1[idx], self.img2[idx], self.labels[idx]

# ==================== MODALITY DROPOUT (Training Robustness) ====================
def apply_training_dropout(csv_b, img1_b, img2_b, prob=TRAIN_DROP_PROB):
    if prob <= 0:
        return csv_b, img1_b, img2_b

    batch_size = csv_b.size(0)
    drop_csv = (torch.rand(batch_size) < prob).to(device)
    drop_img1 = (torch.rand(batch_size) < prob).to(device)
    drop_img2 = (torch.rand(batch_size) < prob).to(device)

    # Prevent dropping all modalities
    all_drop = drop_csv & drop_img1 & drop_img2
    if all_drop.any():
        keep_mask = torch.randint(0, 3, (all_drop.sum(),)).to(device)
        drop_csv[all_drop] = (keep_mask == 0)
        drop_img1[all_drop] = (keep_mask == 1)
        drop_img2[all_drop] = (keep_mask == 2)

    csv_b = csv_b.clone()
    img1_b = img1_b.clone()
    img2_b = img2_b.clone()

    csv_b[drop_csv] = 0
    img1_b[drop_img1] = 0
    img2_b[drop_img2] = 0

    return csv_b, img1_b, img2_b

# ==================== MODEL ====================
class ModelCSVIMG(nn.Module):
    def __init__(self, num_csv_features):
        super(ModelCSVIMG, self).__init__()
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_dropout = nn.Dropout(0.2)

        self.img_conv = nn.Conv2d(1, 18, kernel_size=3, padding=1)
        self.img_bn = nn.BatchNorm2d(18)
        self.img_pool = nn.MaxPool2d(2)
        self.img_fc = nn.Linear(18 * 16 * 16, 100)
        self.img_dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(800, 1200)
        self.dr1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(2000, 12)

    def forward(self, x_csv, x_img1, x_img2):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = self.csv_dropout(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        x_img1 = F.relu(self.img_bn(self.img_conv(x_img1)))
        x_img1 = self.img_pool(x_img1)
        x_img1 = x_img1.reshape(x_img1.size(0), -1)
        x_img1 = F.relu(self.img_fc(x_img1))
        x_img1 = self.img_dropout(x_img1)

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
        return self.fc2(x)  # Logits

# ==================== PARETO SELECTION ====================
def pareto_optimization(metrics, num_clients):
    data = np.array(metrics).T  # (6 objectives, n_clients)
    n = data.shape[1]
    pareto_front = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if np.all(data[:, j] >= data[:, i]) and np.any(data[:, j] > data[:, i]):
                dominated = True
                break
        if not dominated:
            pareto_front.append(i)
    
    if len(pareto_front) >= num_clients:
        return random.sample(pareto_front, num_clients)
    
    # Supplement with weighted score
    scores = (0.3 * data[0] + 0.3 * data[3] + 0.2 * data[1] + 0.2 * data[2])
    candidates = np.argsort(scores)[-num_clients:]
    selected = list(set(pareto_front) | set(candidates.tolist()))
    return selected[:num_clients]

# ==================== LOAD DATA WITH MODALITY HETEROGENEITY ====================
def load_clients_data():
    data = {}
    for idx, sub in enumerate(SUBJECTS):
        # Load and clean sensor data
        train_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_train.csv', skiprows=1)
        test_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_test.csv', skiprows=1)
        
        for df in [train_csv, test_csv]:
            df.dropna(inplace=True)
            df.drop_duplicates(inplace=True)
            df.drop(['Infrared 1','Infrared 2','Infrared 3','Infrared 4','Infrared 5','Infrared 6'], axis=1, inplace=True)
            na_cols = df.columns[df.isnull().any()]
            df.drop(na_cols, axis=1, inplace=True)
            df.set_index('Time', inplace=True)

        # Load images and labels
        img1_train = np.load(f'./dataset/Sensor + Image/{sub}_image_1_train.npy') / 255.0
        img2_train = np.load(f'./dataset/Sensor + Image/{sub}_image_2_train.npy') / 255.0
        label_train = np.load(f'./dataset/Sensor + Image/{sub}_label_1_train.npy')
        name_train = np.load(f'./dataset/Sensor + Image/{sub}_name_1_train.npy')

        img1_test = np.load(f'./dataset/Sensor + Image/{sub}_image_1_test.npy') / 255.0
        img2_test = np.load(f'./dataset/Sensor + Image/{sub}_image_2_test.npy') / 255.0
        label_test = np.load(f'./dataset/Sensor + Image/{sub}_label_1_test.npy')
        name_test = np.load(f'./dataset/Sensor + Image/{sub}_name_1_test.npy')

        # Align train
        valid_train = np.isin(name_train, train_csv.index)
        img1_train = img1_train[valid_train]
        img2_train = img2_train[valid_train]
        label_train = label_train[valid_train]
        train_data = train_csv.loc[name_train[valid_train]].values

        # Align test
        valid_test = np.isin(name_test, test_csv.index)
        img1_test = img1_test[valid_test]
        img2_test = img2_test[valid_test]
        label_test = label_test[valid_test]
        test_data = test_csv.loc[name_test[valid_test]].values

        label_train[label_train == 20] = 0
        label_test[label_test == 20] = 0

        X_train_csv = train_data[:, :-1]
        X_test_csv = test_data[:, :-1]
        y_train = label_train.astype(np.int64)
        y_test = label_test.astype(np.int64)

        scaler = StandardScaler()
        X_train_csv = scaler.fit_transform(X_train_csv)
        X_test_csv = scaler.transform(X_test_csv)

        img1_train = img1_train.reshape(-1, 32, 32, 1)
        img2_train = img2_train.reshape(-1, 32, 32, 1)
        img1_test = img1_test.reshape(-1, 32, 32, 1)
        img2_test = img2_test.reshape(-1, 32, 32, 1)

        data[idx] = {
            'train': (X_train_csv, img1_train, img2_train, y_train),
            'test': (X_test_csv, img1_test, img2_test, y_test),
            'modality': CLIENT_MODALITY_TYPE[idx]
        }

    return data

# ==================== MAIN TRAINING (Heterogeneous Modalities) ====================
if __name__ == "__main__":
    clients_data = load_clients_data()
    csv_dim = clients_data[0]['train'][0].shape[1]
    global_model = ModelCSVIMG(csv_dim).to(device)

    print("=== Federated Learning with Modality Heterogeneity & Missing Sensors ===")
    print("Clients 0-5: Full | 6-8: No Sensor | 9-11: Partial Image")
    best_acc = 0.0

    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")

        client_metrics = []
        client_updates = {}

        for cid in range(TOTAL_CLIENTS):
            modality = clients_data[cid]['modality']
            X_csv_tr, X1_tr, X2_tr, y_tr = clients_data[cid]['train']
            train_loader = DataLoader(CustomDataset(X_csv_tr, X1_tr, X2_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

            X_csv_te, X1_te, X2_te, y_te = clients_data[cid]['test']
            val_loader = DataLoader(CustomDataset(X_csv_te, X1_te, X2_te, y_te), batch_size=BATCH_SIZE)

            local_model = ModelCSVIMG(csv_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            local_model.train()
            losses = []
            for _ in range(LOCAL_EPOCHS):
                epoch_loss = 0.0
                for batch in train_loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]

                    # Apply modality constraints + random dropout
                    if modality == "no_sensor":
                        csv_b = torch.zeros_like(csv_b)
                    elif modality == "partial_img":
                        img2_b = torch.zeros_like(img2_b)  # Always missing cam2

                    # Additional random dropout for robustness
                    csv_b, img1_b, img2_b = apply_training_dropout(csv_b, img1_b, img2_b, prob=TRAIN_DROP_PROB)

                    optimizer.zero_grad()
                    out = local_model(csv_b, img1_b, img2_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))

            # Local validation (on clean data)
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

            # Global test (clean data)
            global_correct, global_total = 0, 0
            with torch.no_grad():
                for gid in range(TOTAL_CLIENTS):
                    X_csv_g, X1_g, X2_g, y_g = clients_data[gid]['test']
                    g_loader = DataLoader(CustomDataset(X_csv_g, X1_g, X2_g, y_g), batch_size=BATCH_SIZE)
                    for batch in g_loader:
                        csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                        out = local_model(csv_b, img1_b, img2_b)
                        pred = out.argmax(dim=1)
                        global_correct += pred.eq(y_b).sum().item()
                        global_total += y_b.size(0)
            global_acc = 100.0 * global_correct / global_total

            # Pareto metrics
            rf_loss_red = max(0.0, losses[0] - losses[-1])
            rf_train = local_acc / 100.0
            rf_val = local_acc / 100.0
            rf_global = global_acc / 100.0
            p_loss = losses[-1]
            p_bias = abs(local_acc - global_acc) / 100.0

            client_metrics.append([rf_loss_red, rf_train, rf_val, rf_global, p_loss, p_bias])

            # Store update
            update = {}
            for name, param in local_model.state_dict().items():
                update[name] = param - global_model.state_dict()[name]
            client_updates[cid] = update

        selected = pareto_optimization(client_metrics, NUM_SELECTED_CLIENTS)
        print(f"Selected clients: {selected} | Modalities: {[clients_data[c]['modality'] for c in selected]}")

        # FedAvg aggregation — ONLY on trainable parameters
        for name, param in global_model.named_parameters():
            updates = [client_updates[cid][name] for cid in selected]
            avg_update = torch.stack(updates).mean(dim=0)
            param.data.add_(avg_update)  # Use .data to avoid gradient issues

        # Optional: Zero out accumulated gradients (good practice)
        global_model.zero_grad(set_to_none=True)

        # Global evaluation on clean data
        global_model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for cid in range(TOTAL_CLIENTS):
                X_csv, X1, X2, y = clients_data[cid]['test']
                loader = DataLoader(CustomDataset(X_csv, X1, X2, y), batch_size=BATCH_SIZE)
                for batch in loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    out = global_model(csv_b, img1_b, img2_b)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(y_b).sum().item()
                    total += y_b.size(0)
        acc = 100.0 * correct / total
        print(f"Global Test Accuracy (Clean): {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(global_model.state_dict(), "best_pareto_heterogeneous_modality.pth")

    print(f"\nTraining completed! Best global accuracy: {best_acc:.2f}%")