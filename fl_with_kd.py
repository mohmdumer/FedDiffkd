"""
Complete FL Pipeline with Server-Side Knowledge Distillation
=============================================================
This is the MAIN notebook script. Paste into Kaggle after the DDIM sampler cell.

Experiments:
  E1: Baseline FedAvg (no augmentation, no KD)
  E2: FedAvg + Server-side KD with DDIM (main method)
  E3: FedAvg + Server-side KD with per-class FID filtering (enhanced method)
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
    'clients_per_round':  10,     # Base papers default to 100, but 10 is standard and fits in Kaggle limit
    'alpha':             0.6,     # Matched base papers
    'global_rounds':    1000,     # Matched base papers for fair comparison
    'local_epochs':        5,     # Matched base papers
    'batch_size':         50,     # Matched base papers
    'learning_rate':     0.1,     # Matched base papers
    'lr_decay_per_round': 0.998,  # Matched base papers
    'momentum':          0.0,     # Matched base papers
    'weight_decay':      1e-3,    # Matched base papers
    'num_classes':        10,
    'grad_clip_norm':    10.0,    # Matched base papers
}

KD_CFG = {
    'kd_iterations':      10,    # Distillation steps per round
    'kd_batch_size':     256,    # Synthetic batch size
    'kd_lr':            0.01,    # Student LR during distillation
    'temperature':       3.0,    # KD temperature
    'inner_student_steps': 5,    # Student steps per generator step
    'ddim_steps':        100,    # DDIM inference steps (fast)
}

BASE_DIR    = Path('/kaggle/working')
CKPT_DIR    = BASE_DIR / 'checkpoints'
RESULTS_DIR = BASE_DIR / 'results'
PLOTS_DIR   = BASE_DIR / 'plots'
for d in [CKPT_DIR, RESULTS_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DDPM_CKPT = '/kaggle/input/datasets/mtalikhan08/checkpoint/ddpm_final.pt'

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
# LOCAL TRAINING (Clients train on REAL data ONLY)
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
    """Flatten model params to numpy array."""
    params = []
    for p in model.state_dict().values():
        params.append(p.cpu().numpy().reshape(-1))
    return np.concatenate(params)


def set_model_from_params(model_template, params, device):
    """Load flattened params into a model."""
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
    """Weighted average of client parameters."""
    sel_params = np.array([client_params[cid] for cid in selected_clients])
    sel_weights = np.array([weight_list[cid] for cid in selected_clients]).reshape(-1, 1)
    sel_weights = sel_weights / sel_weights.sum()
    avg_params = np.sum(sel_params * sel_weights, axis=0)
    return set_model_from_params(model_template, avg_params, device), avg_params


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER-SIDE KNOWLEDGE DISTILLATION (Core Contribution)
# ═══════════════════════════════════════════════════════════════════════════════

def server_knowledge_distillation(
    global_model, client_params, selected_clients, client_class_counts,
    model_template, ddpm_model, ddpm_scheduler, ddim_gen_fn,
    kd_cfg, fid_weights=None, device='cuda'
):
    """
    After FedAvg, fine-tune global model via knowledge distillation
    using DDIM synthetic images and client models as teachers.
    """
    num_classes = FL_CFG['num_classes']
    student = copy.deepcopy(global_model).to(device)
    student.train()

    opt = optim.SGD(student.parameters(), lr=kd_cfg['kd_lr'],
                    momentum=0.9, weight_decay=1e-4)
    kl_fn = nn.KLDivLoss(reduction='batchmean')

    # Load teacher models
    teachers = []
    for cid in selected_clients:
        t = set_model_from_params(model_template, client_params[cid], device)
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
        teachers.append(t)

    # Per-class, per-client weights
    counts = np.zeros((len(selected_clients), num_classes))
    for i, cid in enumerate(selected_clients):
        for c in range(num_classes):
            counts[i, c] = client_class_counts[cid].get(c, 0)
    total = counts.sum(axis=0) + 1e-8
    cls_wt = (counts / total[np.newaxis, :]).T  # (classes, clients)

    # Class generation weights
    gen_w = total.copy()
    if fid_weights:
        for c in range(num_classes):
            gen_w[c] *= fid_weights.get(c, 1.0)

    T = kd_cfg['temperature']
    kd_losses = []

    for step in range(kd_cfg['kd_iterations']):
        # Generate labels proportional to class distribution
        probs = gen_w / gen_w.sum()
        labels = np.random.choice(num_classes, size=kd_cfg['kd_batch_size'], p=probs)
        labels_t = torch.from_numpy(labels).long().to(device)

        # Generate synthetic images per class
        fake_batch = torch.zeros(kd_cfg['kd_batch_size'], 3, 32, 32, device=device)
        for c in range(num_classes):
            mask = (labels_t == c)
            n_c = mask.sum().item()
            if n_c > 0:
                imgs = ddim_gen_fn(ddpm_model, ddpm_scheduler, class_id=c,
                                   n_samples=n_c, num_steps=kd_cfg['ddim_steps'],
                                   eta=0.0, device=device, batch_size=min(n_c, 64))
                # Normalize
                mean = torch.tensor(CIFAR10_MEAN).view(1,3,1,1).to(device)
                std  = torch.tensor(CIFAR10_STD).view(1,3,1,1).to(device)
                imgs = (imgs.to(device) - mean) / std
                idxs = mask.nonzero(as_tuple=True)[0][:n_c]
                fake_batch[idxs] = imgs[:len(idxs)]

        # Teacher ensemble prediction
        batch_wt = torch.from_numpy(cls_wt[labels]).float().to(device)  # (B, clients)
        with torch.no_grad():
            ensemble = torch.zeros(kd_cfg['kd_batch_size'], num_classes, device=device)
            for i, t_model in enumerate(teachers):
                t_logit = t_model(fake_batch)
                t_soft = F.softmax(t_logit / T, dim=1)
                ensemble += t_soft * batch_wt[:, i:i+1]

        # Apply FID quality weighting
        if fid_weights:
            q = torch.ones(kd_cfg['kd_batch_size'], device=device)
            for c in range(num_classes):
                q[labels_t == c] = fid_weights.get(c, 1.0)
            ensemble = ensemble * q.unsqueeze(1)
            ensemble = ensemble / (ensemble.sum(dim=1, keepdim=True) + 1e-8)

        # Train student (inner loop)
        for _ in range(kd_cfg['inner_student_steps']):
            opt.zero_grad()
            s_logit = student(fake_batch)
            s_log = F.log_softmax(s_logit / T, dim=1)
            loss = kl_fn(s_log, ensemble) * (T ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            opt.step()

        kd_losses.append(loss.item())

    del teachers
    torch.cuda.empty_cache()
    student.eval()
    return student, kd_losses


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

def run_experiment(name, global_model, client_data, client_labels,
                   client_class_counts, test_data, test_labels,
                   ddpm_model=None, ddpm_scheduler=None, ddim_gen_fn=None,
                   use_kd=False, fid_weights=None):
    """
    Full FL experiment.
    If use_kd=True, applies server-side KD after FedAvg each round.
    """
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"  KD: {'ON' if use_kd else 'OFF'}")
    print(f"  FID filtering: {'ON' if fid_weights else 'OFF'}")
    print(f"{'='*60}")

    model_template = create_resnet18()
    
    weight_list = {cid: max(len(client_labels[cid]), 1)
                   for cid in range(FL_CFG['num_clients'])}

    latest_ckpt_path = CKPT_DIR / f'{name}_latest.pt'
    start_round = 0
    history = []
    best_acc = 0.0
    
    # ── Checkpoint Resuming Logic ──
    if latest_ckpt_path.exists():
        print(f"  [!] Found existing checkpoint: {latest_ckpt_path}")
        ckpt = torch.load(latest_ckpt_path, map_location=DEVICE)
        start_round = ckpt['round'] + 1
        avg_model = model_template.to(DEVICE)
        avg_model.load_state_dict(ckpt['model_state'])
        
        # Reconstruct client params from the loaded global model
        init_params = get_model_params(avg_model)
        client_params = {cid: init_params.copy() for cid in range(FL_CFG['num_clients'])}
        
        history = ckpt['history']
        best_acc = ckpt.get('best_acc', 0.0)
        print(f"  [!] Resuming from Round {start_round}. Best Acc so far: {best_acc:.4f}")
    else:
        # Initialize from scratch
        init_params = get_model_params(global_model)
        client_params = {cid: init_params.copy() for cid in range(FL_CFG['num_clients'])}

    base_lr = FL_CFG['learning_rate']

    for rnd in range(start_round, FL_CFG['global_rounds']):
        t0 = time.time()

        # LR decay per round (matches base papers)
        lr_round = base_lr * (FL_CFG['lr_decay_per_round'] ** rnd)

        # Select clients
        selected = np.random.choice(FL_CFG['num_clients'],
                                    FL_CFG['clients_per_round'], replace=False).tolist()

        # Local training (REAL data only)
        for cid in selected:
            local_model = set_model_from_params(model_template, client_params[cid], DEVICE)
            for p in local_model.parameters():
                p.requires_grad = True
            local_model = local_train(local_model, client_data[cid],
                                      client_labels[cid], lr_round, FL_CFG)
            client_params[cid] = get_model_params(local_model)
            del local_model

        # FedAvg aggregation
        avg_model, _ = fedavg_aggregate(model_template, client_params,
                                         selected, weight_list, DEVICE)

        # Server-side Knowledge Distillation (if enabled)
        if use_kd and ddpm_model is not None:
            avg_model, kd_losses = server_knowledge_distillation(
                avg_model, client_params, selected, client_class_counts,
                model_template, ddpm_model, ddpm_scheduler, ddim_gen_fn,
                KD_CFG, fid_weights=fid_weights, device=DEVICE
            )

        # Update global params for next round
        global_params = get_model_params(avg_model)
        # Broadcast: all clients get the new global model (not just selected)
        # This is standard FedAvg behavior
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
                    'accuracy': acc, 'experiment': name
                }, CKPT_DIR / f'{name}_best.pt')

            print(f"  Round {rnd:3d}/{FL_CFG['global_rounds']} | "
                  f"LR={lr_round:.5f} | Loss={test_loss:.4f} | "
                  f"Acc={acc:.4f} (Best={best_acc:.4f}) | "
                  f"F1={f1:.4f} | Time={elapsed:.1f}s")
                  
        # Save LATEST checkpoint every round so you can resume on Kaggle crash
        torch.save({
            'round': rnd,
            'model_state': avg_model.state_dict(),
            'history': history,              # Save metrics history
            'best_acc': best_acc,
            'experiment': name
        }, latest_ckpt_path)

        torch.cuda.empty_cache()

    # Save final model
    torch.save({
        'round': FL_CFG['global_rounds'],
        'model_state': avg_model.state_dict(),
        'experiment': name
    }, CKPT_DIR / f'{name}_final.pt')

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(RESULTS_DIR / f'{name}_history.csv', index=False)
    print(f"\n✓ {name} complete. Best accuracy: {best_acc:.4f}")
    return avg_model, hist_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Load Data ──
    print("Loading CIFAR-10...")
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # ── Dirichlet Split ──
    print(f"Partitioning into {FL_CFG['num_clients']} clients (α={FL_CFG['alpha']})...")
    indices = dirichlet_split(train_labels, FL_CFG['num_clients'], FL_CFG['alpha'])
    client_data   = {cid: train_data[indices[cid]] for cid in range(FL_CFG['num_clients'])}
    client_labels = {cid: train_labels[indices[cid]] for cid in range(FL_CFG['num_clients'])}
    client_class_counts = {
        cid: {c: int((client_labels[cid] == c).sum()) for c in range(10)}
        for cid in range(FL_CFG['num_clients'])
    }

    # ── Load DDPM for DDIM ──
    print("Loading DDPM model for DDIM sampling...")
    ddpm_model, ddpm_scheduler = load_ddpm_model(DDPM_CKPT, DEVICE)

    # ── FID Quality Weights (from your per_class_fid.py results) ──
    # Update these values after running per_class_fid.py with 250 steps!
    # Higher weight = better quality = more synthetic data used
    # Formula: weight = max(0, 1 - (fid - 25) / 50)
    fid_scores = {
        0: 40.6128,  # airplane
        1: 34.2055,  # automobile
        2: 41.3769,  # bird
        3: 38.1637,  # cat
        4: 29.3205,  # deer
        5: 40.3650,  # dog
        6: 30.7910,  # frog
        7: 37.3693,  # horse
        8: 31.7575,  # ship
        9: 37.0094,  # truck
    }
    fid_weights = {c: max(0.1, 1.0 - (fid - 25) / 50) for c, fid in fid_scores.items()}
    print("FID quality weights:", {CLASS_NAMES[c]: f"{w:.2f}" for c, w in fid_weights.items()})

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Baseline FedAvg (No augmentation, No KD)
    # ══════════════════════════════════════════════════════════════════════
    baseline_model = create_resnet18().to(DEVICE)
    baseline_model, baseline_hist = run_experiment(
        'E1_Baseline', baseline_model,
        client_data, client_labels, client_class_counts,
        test_data, test_labels,
        use_kd=False
    )

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: FedAvg + Server-Side KD with DDIM (Main Method)
    # ══════════════════════════════════════════════════════════════════════
    kd_model = create_resnet18().to(DEVICE)
    kd_model, kd_hist = run_experiment(
        'E2_FedAvg_KD', kd_model,
        client_data, client_labels, client_class_counts,
        test_data, test_labels,
        ddpm_model=ddpm_model, ddpm_scheduler=ddpm_scheduler,
        ddim_gen_fn=ddim_generate,
        use_kd=True, fid_weights=None  # No filtering
    )

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: FedAvg + Server-Side KD + FID Filtering (Enhanced)
    # ══════════════════════════════════════════════════════════════════════
    kd_fid_model = create_resnet18().to(DEVICE)
    kd_fid_model, kd_fid_hist = run_experiment(
        'E3_FedAvg_KD_FID', kd_fid_model,
        client_data, client_labels, client_class_counts,
        test_data, test_labels,
        ddpm_model=ddpm_model, ddpm_scheduler=ddpm_scheduler,
        ddim_gen_fn=ddim_generate,
        use_kd=True, fid_weights=fid_weights  # Quality filtering ON
    )

    # ══════════════════════════════════════════════════════════════════════
    # FINAL COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("FINAL RESULTS COMPARISON")
    print("=" * 60)
    for name, hist in [('E1_Baseline', baseline_hist),
                       ('E2_FedAvg_KD', kd_hist),
                       ('E3_FedAvg_KD_FID', kd_fid_hist)]:
        best = hist['accuracy'].max()
        final = hist['accuracy'].iloc[-1]
        print(f"  {name:20s}: best={best:.4f}, final={final:.4f}")

    # Save comparison
    comparison = {
        'E1_Baseline': {'best': float(baseline_hist['accuracy'].max()),
                        'final': float(baseline_hist['accuracy'].iloc[-1])},
        'E2_FedAvg_KD': {'best': float(kd_hist['accuracy'].max()),
                         'final': float(kd_hist['accuracy'].iloc[-1])},
        'E3_FedAvg_KD_FID': {'best': float(kd_fid_hist['accuracy'].max()),
                             'final': float(kd_fid_hist['accuracy'].iloc[-1])},
    }
    with open(RESULTS_DIR / 'final_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # Save all configs
    with open(RESULTS_DIR / 'config_fl.json', 'w') as f:
        json.dump(FL_CFG, f, indent=2)
    with open(RESULTS_DIR / 'config_kd.json', 'w') as f:
        json.dump(KD_CFG, f, indent=2)

    print(f"\n✓ All experiments complete. Results saved to {RESULTS_DIR}")
    print(f"✓ Model checkpoints saved to {CKPT_DIR}")
