"""
E1: Baseline FedAvg (No Augmentation, No KD)
=============================================
Self-contained script — paste into a single Kaggle cell and run.
NO DDPM/DDIM needed. NO GPU-heavy generation.
This is the pure FedAvg baseline for comparison.

Kaggle Account #1
"""

import os, copy, random, math, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision.datasets import CIFAR10
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASS_NAMES  = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

FL_CFG = {
    'num_clients':       100,
    'clients_per_round':  10,
    'alpha':             0.6,     # Matched base papers
    'global_rounds':    1000,     # Matched base papers
    'local_epochs':        5,     # Matched base papers
    'batch_size':         50,     # Matched base papers
    'learning_rate':     0.1,     # Matched base papers
    'lr_decay_per_round': 0.998,  # Matched base papers
    'momentum':          0.0,     # Matched base papers
    'weight_decay':      1e-3,    # Matched base papers
    'num_classes':        10,
    'grad_clip_norm':    10.0,    # Matched base papers
}

BASE_DIR    = Path('/kaggle/working')
CKPT_DIR    = BASE_DIR / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results'
for d in [CKPT_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def create_resnet18(num_classes=10):
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PARTITIONING
# ═══════════════════════════════════════════════════════════════════════════════

def load_cifar10(data_dir='./data'):
    raw_train = CIFAR10(root=data_dir, train=True, download=True)
    raw_test  = CIFAR10(root=data_dir, train=False, download=True)

    mean_t = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    train_data   = (torch.from_numpy(raw_train.data.astype(np.float32)/255).permute(0,3,1,2) - mean_t) / std_t
    train_labels = np.array(raw_train.targets)
    test_data    = (torch.from_numpy(raw_test.data.astype(np.float32)/255).permute(0,3,1,2) - mean_t) / std_t
    test_labels  = np.array(raw_test.targets)
    return train_data, train_labels, test_data, test_labels


def dirichlet_split(labels, n_clients, alpha, seed=42):
    np.random.seed(seed)
    n_classes = len(np.unique(labels))
    class_idx = defaultdict(list)
    for i, l in enumerate(labels):
        class_idx[l].append(i)
    client_idx = defaultdict(list)
    for c in range(n_classes):
        ci = list(class_idx[c])
        props = np.random.dirichlet([alpha] * n_clients)
        spc = (props * len(ci)).astype(int)
        diff = len(ci) - spc.sum()
        for j in range(abs(diff)):
            spc[j % n_clients] += 1 if diff > 0 else -1
        np.random.shuffle(ci)
        s = 0
        for cid, n in enumerate(spc):
            n = max(0, n)
            client_idx[cid].extend(ci[s:s+n])
            s += n
    return dict(client_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def local_train(model, data_x, data_y, lr, cfg):
    """Local SGD training on real client data only."""
    if isinstance(data_y, np.ndarray):
        data_y = torch.from_numpy(data_y)
    ds = TensorDataset(data_x.to(DEVICE), data_y.long().to(DEVICE))
    loader = DataLoader(ds, batch_size=cfg['batch_size'], shuffle=True, drop_last=False)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=cfg['momentum'],
                    weight_decay=cfg['weight_decay'])
    crit = nn.CrossEntropyLoss()
    model.to(DEVICE).train()

    for _ in range(cfg['local_epochs']):
        for imgs, labs in loader:
            opt.zero_grad()
            loss = crit(model(imgs), labs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg['grad_clip_norm'])
            opt.step()
    return model


def get_model_params(model):
    params = []
    for p in model.state_dict().values():
        params.append(p.cpu().numpy().reshape(-1))
    return np.concatenate(params)


def set_model_from_params(model_template, params, device):
    model = copy.deepcopy(model_template)
    state = model.state_dict()
    idx = 0
    for name, p in state.items():
        length = p.numel()
        state[name] = torch.from_numpy(
            params[idx:idx+length].reshape(p.shape)
        ).float()
        idx += length
    model.load_state_dict(state)
    return model.to(device)


# ═══════════════════════════════════════════════════════════════════════════════
# FEDAVG AGGREGATION
# ═══════════════════════════════════════════════════════════════════════════════

def fedavg_aggregate(model_template, client_params, selected_clients,
                     weight_list, device):
    sel_params = np.array([client_params[cid] for cid in selected_clients])
    sel_weights = np.array([weight_list[cid] for cid in selected_clients]).reshape(-1, 1)
    sel_weights = sel_weights / sel_weights.sum()
    avg_params = np.sum(sel_params * sel_weights, axis=0)
    return set_model_from_params(model_template, avg_params, device), avg_params


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate(model, test_data, test_labels):
    model.eval()
    loader = DataLoader(
        TensorDataset(test_data.to(DEVICE), torch.from_numpy(test_labels).long().to(DEVICE)),
        batch_size=256, shuffle=False
    )
    crit = nn.CrossEntropyLoss(reduction='sum')
    total_loss = 0.0
    all_preds = []
    all_labs = []
    
    with torch.no_grad():
        for imgs, labs in loader:
            logits = model(imgs)
            loss = crit(logits, labs)
            total_loss += loss.item()
            
            preds = logits.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labs.extend(labs.cpu().numpy())
            
    acc = np.mean(np.array(all_preds) == np.array(all_labs))
    avg_loss = total_loss / len(all_labs)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labs, all_preds, average='macro', zero_division=0
    )
    
    return acc, avg_loss, precision, recall, f1


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FL TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    EXP_NAME = 'E1_Baseline'

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {EXP_NAME}")
    print(f"  KD: OFF")
    print(f"  FID filtering: OFF")
    print(f"{'='*60}")

    # ── Load Data ──
    print("Loading CIFAR-10...")
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # ── Dirichlet Split ──
    print(f"Partitioning into {FL_CFG['num_clients']} clients (α={FL_CFG['alpha']})...")
    indices = dirichlet_split(train_labels, FL_CFG['num_clients'], FL_CFG['alpha'])
    client_data   = {cid: train_data[indices[cid]] for cid in range(FL_CFG['num_clients'])}
    client_labels = {cid: train_labels[indices[cid]] for cid in range(FL_CFG['num_clients'])}

    # ── Initialize ──
    model_template = create_resnet18()
    global_model = create_resnet18().to(DEVICE)
    
    weight_list = {cid: max(len(client_labels[cid]), 1)
                   for cid in range(FL_CFG['num_clients'])}

    latest_ckpt_path = CKPT_DIR / f'{EXP_NAME}_latest.pt'
    start_round = 0
    history = []
    best_acc = 0.0

    # ── Checkpoint Resuming Logic ──
    if latest_ckpt_path.exists():
        print(f"  [!] Found existing checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        start_round = ckpt['round'] + 1
        global_model.load_state_dict(ckpt['model_state'])
        
        # Reconstruct client params from the loaded global model
        init_params = get_model_params(global_model)
        client_params = {cid: init_params.copy() for cid in range(FL_CFG['num_clients'])}
        
        history = ckpt['history']
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"  [!] Resuming from Round {start_round}. Best Acc so far: {best_acc:.4f}")
    else:
        init_params = get_model_params(global_model)
        client_params = {cid: init_params.copy() for cid in range(FL_CFG['num_clients'])}

    base_lr = FL_CFG['learning_rate']

    # ── Training Loop ──
    for rnd in range(start_round, FL_CFG['global_rounds']):
        t0 = time.time()

        lr_round = base_lr * (FL_CFG['lr_decay_per_round'] ** rnd)

        selected = np.random.choice(FL_CFG['num_clients'],
                                    FL_CFG['clients_per_round'], replace=False).tolist()

        for cid in selected:
            local_model = set_model_from_params(model_template, client_params[cid], DEVICE)
            for p in local_model.parameters():
                p.requires_grad = True
            local_model = local_train(local_model, client_data[cid],
                                      client_labels[cid], lr_round, FL_CFG)
            client_params[cid] = get_model_params(local_model)
            del local_model

        avg_model, _ = fedavg_aggregate(model_template, client_params,
                                         selected, weight_list, DEVICE)

        global_params = get_model_params(avg_model)
        for cid in range(FL_CFG['num_clients']):
            client_params[cid] = global_params.copy()

        # Evaluate & Checkpoint
        if rnd % 5 == 0 or rnd == FL_CFG['global_rounds'] - 1:
            acc, test_loss, prec, rec, f1 = evaluate(avg_model, test_data, test_labels)
            elapsed = time.time() - t0
            history.append({
                'round': rnd, 'lr': lr_round, 'accuracy': acc, 
                'test_loss': test_loss, 'precision': prec, 
                'recall': rec, 'f1': f1, 'time': elapsed
            })

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    'round': rnd, 'model_state': avg_model.state_dict(),
                    'accuracy': acc, 'experiment': EXP_NAME
                }, CKPT_DIR / f'{EXP_NAME}_best.pt')

            print(f"  Round {rnd:3d}/{FL_CFG['global_rounds']} | "
                  f"LR={lr_round:.5f} | Loss={test_loss:.4f} | "
                  f"Acc={acc:.4f} (Best={best_acc:.4f}) | "
                  f"F1={f1:.4f} | Time={elapsed:.1f}s")

        # Save checkpoint every round
        torch.save({
            'round': rnd,
            'model_state': avg_model.state_dict(),
            'history': history,
            'best_acc': best_acc,
            'experiment': EXP_NAME
        }, latest_ckpt_path)

        torch.cuda.empty_cache()

    # Save final model
    torch.save({
        'round': FL_CFG['global_rounds'],
        'model_state': avg_model.state_dict(),
        'experiment': EXP_NAME
    }, CKPT_DIR / f'{EXP_NAME}_final.pt')

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / f'{EXP_NAME}_history.csv', index=False)

    # Save config
    with open(RESULTS_DIR / 'config_fl.json', 'w') as f:
        json.dump(FL_CFG, f, indent=2)

    print(f"\n✓ {EXP_NAME} complete. Best accuracy: {best_acc:.4f}")
    print(f"✓ Results saved to {RESULTS_DIR}")
    print(f"✓ Model checkpoints saved to {CKPT_DIR}")
