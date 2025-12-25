import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import csv

# ==================== CONFIGURATION ====================
def set_seed(seed=0):
    # 设置环境变量以确保可重复性
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置 NumPy 随机种子
    np.random.seed(seed)
    # 设置 Python 随机种子
    random.seed(seed)
    # 设置 PyTorch 随机种子
    torch.manual_seed(seed)
    # 如果使用 GPU，设置 GPU 随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 对所有 GPU 进行设置
    # 确保所有操作都在单线程中进行（可选）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROUNDS = 20
NUM_SELECTED_CLIENTS = 6
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
TOTAL_CLIENTS = 12
SUBJECTS = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]

# ==================== DATASET ====================
class CustomDataset(Dataset):
    def __init__(self, csv_features, img1_features, img2_features, labels):
        self.csv = torch.tensor(csv_features, dtype=torch.float32)
        self.img1 = torch.tensor(img1_features, dtype=torch.float32)
        self.img2 = torch.tensor(img2_features, dtype=torch.float32)
        # Add .reshape(-1) or .squeeze() to ensure 1D shape
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
        x_img1 = x_img1.permute(0, 3, 1, 2)                    # (B, 32, 32, 1) -> (B, 1, 32, 32)
        x_img1 = F.relu(self.img_bn(self.img_conv(x_img1)))
        x_img1 = self.img_pool(x_img1)                         # -> (B, 18, 16, 16)
        x_img1 = x_img1.reshape(x_img1.size(0), -1)            # ← FIXED: Use reshape instead of view
        x_img1 = F.relu(self.img_fc(x_img1))
        x_img1 = self.img_dropout(x_img1)

        # Img2
        x_img2 = x_img2.permute(0, 3, 1, 2)
        x_img2 = F.relu(self.img_bn(self.img_conv(x_img2)))
        x_img2 = self.img_pool(x_img2)
        x_img2 = x_img2.reshape(x_img2.size(0), -1)            # ← FIXED here too
        x_img2 = F.relu(self.img_fc(x_img2))
        x_img2 = self.img_dropout(x_img2)

        # Fusion
        x = torch.cat((x_csv, x_img1, x_img2), dim=1)
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dr1(x)
        x = torch.cat((residual, x), dim=1)
        return self.fc2(x)   # Raw logits
    
# ==================== METRIC CALCULATIONS ====================
def calculate_relative_loss_reduction(client_losses):
    reductions = [losses[0] - losses[-1] if len(losses) > 1 else 0 for losses in client_losses.values()]
    max_red = max(reductions) if max(reductions) > 0 else 1
    return [r / max_red for r in reductions]

def calculate_relative_train_accuracy(client_acc):
    max_acc = max(client_acc.values())
    return [acc / max_acc if max_acc > 0 else 0 for acc in client_acc.values()]

def calculate_global_validation_accuracy(train_acc, global_acc):
    diffs = [global_acc[c] - train_acc[c] for c in train_acc]
    max_diff = max(diffs) if max(diffs) > 0 else 1
    max_global = max(global_acc.values())
    return [(global_acc[c] / max_global if max_global > 0 else 0) - (diffs[i] / max_diff) for i, c in enumerate(train_acc)]

def calculate_loss_outliers(client_losses, lambda_loss=1.5):
    final_losses = [losses[-1] for losses in client_losses.values()]
    mean_l = np.mean(final_losses)
    std_l = np.std(final_losses)
    threshold = mean_l + lambda_loss * std_l
    max_l = max(final_losses)
    return [l / max_l if l > threshold else 0 for l in final_losses] if max_l > 0 else [0]*len(final_losses)

def calculate_performance_bias(local_val_acc, global_val_acc):
    return [abs(local_val_acc[c] - global_val_acc[c]) / max(local_val_acc[c], global_val_acc[c])
            if max(local_val_acc[c], global_val_acc[c]) > 0 else 0
            for c in local_val_acc]

def pareto_optimization(metrics, num_clients):
    data = np.array(metrics).T  # 6 objectives
    def is_dominated(i):
        return any(np.all(data[j] >= data[i]) and np.any(data[j] > data[i]) for j in range(len(data)) if j != i)
    pareto = [i for i in range(len(data)) if not is_dominated(i)]
    if len(pareto) >= num_clients:
        return random.sample(pareto, num_clients)
    # Supplement
    scores = 0.3*data[:,0] + 0.3*data[:,3] + 0.2*data[:,1] + 0.2*data[:,2]
    top = np.argsort(scores)[- (num_clients - len(pareto)):]
    return list(set(pareto + list(top)))[:num_clients]

# ==================== DATA LOADING ====================
def load_clients_data():
    data = {}
    for idx, sub in enumerate(SUBJECTS):
        # Train
        train_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_train.csv', skiprows=1)
        train_csv.dropna(inplace=True)
        train_csv.drop_duplicates(inplace=True)
        drop_cols = ['Infrared 1','Infrared 2','Infrared 3','Infrared 4','Infrared 5','Infrared 6']
        na_cols = train_csv.columns[train_csv.isnull().any()]
        train_csv.drop(drop_cols + list(na_cols), axis=1, inplace=True)
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
        name_train = name_train[valid_idx]

        train_data = train_csv.loc[name_train].values
        label_train[label_train == 20] = 0

        X_train_csv = train_data[:, :-1]
        y_train = label_train
        scaler = StandardScaler()
        X_train_csv = scaler.fit_transform(X_train_csv)

        img1_train = img1_train.reshape(-1, 32, 32, 1)
        img2_train = img2_train.reshape(-1, 32, 32, 1)
        y_train = label_train.astype(np.int64)   # Keep as class indices
        
        # Test (same process)
        test_csv = pd.read_csv(f'./dataset/Sensor + Image/{sub}_sensor_test.csv', skiprows=1)
        test_csv.dropna(inplace=True)
        test_csv.drop_duplicates(inplace=True)
        test_csv.drop(drop_cols + list(na_cols), axis=1, inplace=True)
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
        y_test = label_test
        X_test_csv = scaler.transform(X_test_csv)

        img1_test = img1_test.reshape(-1, 32, 32, 1)
        img2_test = img2_test.reshape(-1, 32, 32, 1)
        y_test = label_test.astype(np.int64)

        data[idx] = {
            'train': (X_train_csv, img1_train, img2_train, y_train),
            'test': (X_test_csv, img1_test, img2_test, y_test),
            'scaler': scaler
        }

    return data

# ==================== FEDERATED TRAINING ====================
def run_experiment(scenario: str, selection_method: str):
    data = load_clients_data()
    csv_features_dim = data[0]['train'][0].shape[1]
    global_model = ModelCSVIMG(csv_features_dim).to(device)

    round_accuracies = []

    for round_num in range(ROUNDS):
        print(f"\n=== Round {round_num+1}/{ROUNDS} | Scenario: {scenario} | Selection: {selection_method} ===")

        client_local_val_acc = {}
        client_global_val_acc = {}
        client_losses = {}
        client_train_acc = {}

        selected_clients = random.sample(range(TOTAL_CLIENTS), NUM_SELECTED_CLIENTS) if selection_method == "random" else None

        if selection_method == "pareto":
            client_metrics = []

        weight_accumulator = {name: torch.zeros_like(param) for name, param in global_model.state_dict().items()}

        for cid in range(TOTAL_CLIENTS):
            X_csv_tr, X1_tr, X2_tr, y_tr = data[cid]['train']
            X_csv_te, X1_te, X2_te, y_te = data[cid]['test']

            # === MODALITY ABLATION ===
            if scenario == "only_imu":
                X1_tr = np.random.rand(*X1_tr.shape)
                X2_tr = np.random.rand(*X2_tr.shape)
                X1_te = np.random.rand(*X1_te.shape)
                X2_te = np.random.rand(*X2_te.shape)
            elif scenario == "only_camera":
                X_csv_tr = np.random.rand(*X_csv_tr.shape)
                X_csv_te = np.random.rand(*X_csv_te.shape)
            elif scenario == "missing_modalities" and cid >= 6:
                X_csv_tr = np.random.rand(*X_csv_tr.shape)
                X_csv_te = np.random.rand(*X_csv_te.shape)
            elif scenario == "partial_modalities" and cid >= 6:
                X2_tr = np.random.rand(*X2_tr.shape)
                X2_te = np.random.rand(*X2_te.shape)

            train_dataset = CustomDataset(X_csv_tr, X1_tr, X2_tr, y_tr)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_dataset = CustomDataset(X_csv_te, X1_te, X2_te, y_te)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

            local_model = ModelCSVIMG(csv_features_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # Local Training
            local_model.train()
            losses = []
            for _ in range(LOCAL_EPOCHS):
                epoch_loss = 0
                for batch in train_loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    optimizer.zero_grad()
                    out = local_model(csv_b, img1_b, img2_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                losses.append(epoch_loss / len(train_loader))

            # Local Validation Accuracy
            local_model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    out = local_model(csv_b, img1_b, img2_b)
                    pred = out.argmax(dim=1)
                    correct += pred.eq(y_b).sum().item()  # y_b is now (B,) long tensor → works!
                    total += y_b.size(0)
            local_val_acc = 100.0 * correct / total

            # Global Validation Accuracy (on all test data)
            global_correct = global_total = 0
            with torch.no_grad():
                for gid in range(TOTAL_CLIENTS):
                    X_csv_g, X1_g, X2_g, y_g = data[gid]['test']
                    if scenario == "only_imu":
                        X1_g = np.random.rand(*X1_g.shape); X2_g = np.random.rand(*X2_g.shape)
                    elif scenario == "only_camera":
                        X_csv_g = np.random.rand(*X_csv_g.shape)
                    elif scenario == "missing_modalities" and gid >= 6:
                        X_csv_g = np.random.rand(*X_csv_g.shape)
                    elif scenario == "partial_modalities" and gid >= 6:
                        X2_g = np.random.rand(*X2_g.shape)
                    g_dataset = CustomDataset(X_csv_g, X1_g, X2_g, y_g)
                    g_loader = DataLoader(g_dataset, batch_size=BATCH_SIZE)
                    for batch in g_loader:
                        csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                        out = local_model(csv_b, img1_b, img2_b)
                        pred = out.argmax(1)
                        global_correct += pred.eq(y_b.argmax(1)).sum().item()
                        global_total += y_b.size(0)
            global_val_acc = 100.0 * global_correct / global_total

            client_local_val_acc[cid] = local_val_acc
            client_global_val_acc[cid] = global_val_acc
            client_losses[cid] = losses
            client_train_acc[cid] = local_val_acc  # approximate

            if selection_method == "pareto":
                rf_loss = calculate_relative_loss_reduction({0: losses})[0]
                rf_acc_train = local_val_acc / 100.0
                rf_acc_val = local_val_acc / 100.0
                rf_acc_global = global_val_acc / 100.0
                p_loss = calculate_loss_outliers({0: losses})[0]
                p_bias = calculate_performance_bias({0: local_val_acc}, {0: global_val_acc})[0]
                client_metrics.append([rf_loss, rf_acc_train, rf_acc_val, rf_acc_global, p_loss, p_bias])

            # FedAvg update
            if selection_method == "random" and cid in selected_clients or selection_method == "pareto":
                for name, param in local_model.state_dict().items():
                    weight_accumulator[name] += (param - global_model.state_dict()[name])

        if selection_method == "pareto":
            selected_clients = pareto_optimization(client_metrics, NUM_SELECTED_CLIENTS)

        # Aggregate
        for name in weight_accumulator:
            weight_accumulator[name] /= len(selected_clients)
            global_model.state_dict()[name].add_(weight_accumulator[name])

        # Global Test
        global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for cid in range(TOTAL_CLIENTS):
                X_csv, X1, X2, y = data[cid]['test']
                if scenario == "only_imu":
                    X1 = np.random.rand(*X1.shape); X2 = np.random.rand(*X2.shape)
                elif scenario == "only_camera":
                    X_csv = np.random.rand(*X_csv.shape)
                elif scenario == "missing_modalities" and cid >= 6:
                    X_csv = np.random.rand(*X_csv.shape)
                elif scenario == "partial_modalities" and cid >= 6:
                    X2 = np.random.rand(*X2.shape)
                dataset = CustomDataset(X_csv, X1, X2, y)
                loader = DataLoader(dataset, batch_size=BATCH_SIZE)
                for batch in loader:
                    csv_b, img1_b, img2_b, y_b = [x.to(device) for x in batch]
                    out = global_model(csv_b, img1_b, img2_b)
                    pred = out.argmax(1)
                    correct += pred.eq(y_b.argmax(1)).sum().item()
                    total += y_b.size(0)
        acc = 100.0 * correct / total
        round_accuracies.append(acc)
        print(f"Global Test Accuracy: {acc:.2f}%")

    final_acc = round_accuracies[-1]
    peak_acc = max(round_accuracies)
    peak_round = round_accuracies.index(peak_acc) + 1
    return final_acc, peak_acc, peak_round

# ==================== RUN ABLATION & PRINT TABLE ====================
if __name__ == "__main__":
    scenarios = [
        "full_multimodal",
        "missing_modalities",
        "partial_modalities",
        "only_imu",
        "only_camera"
    ]
    methods = ["random", "pareto"]

    results = []

    for scenario in scenarios:
        for method in methods:
            print(f"\n{'='*20} STARTING {scenario.upper()} + {method.upper()} {'='*20}")
            final, peak, peak_round = run_experiment(scenario, method)
            results.append({
                "Modality Scenario": scenario.replace("_", " ").title(),
                "Client Selection": method.capitalize(),
                "Final Accuracy (%)": f"{final:.2f}",
                "Peak Accuracy (%)": f"{peak:.2f}",
                "Rounds to Peak": peak_round
            })

    # Print Table
    df = pd.DataFrame(results)
    print("\n" + "="*100)
    print("FINAL MODALITY ABLATION STUDY RESULTS")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100)