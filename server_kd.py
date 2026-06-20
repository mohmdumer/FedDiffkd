"""
Server-Side Knowledge Distillation Module
==========================================
Core contribution: After FedAvg aggregation each round, the server uses
DDIM-generated synthetic images to distill knowledge from client models
(teachers) into the global model (student).

This replaces naive client-side data injection with principled KD.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DiversityLoss(nn.Module):
    """Encourages the generator to produce diverse outputs (anti-mode-collapse)."""
    def __init__(self, metric='l1'):
        super().__init__()
        self.metric = metric

    def forward(self, noises, images):
        if len(images.shape) > 2:
            images = images.view(images.size(0), -1)
        # Pairwise distances
        n = images.size(0)
        img1 = images.unsqueeze(1).expand(n, n, -1)
        img2 = images.unsqueeze(0).expand(n, n, -1)
        img_dist = torch.abs(img1 - img2).mean(dim=2)

        if len(noises.shape) > 2:
            noises = noises.view(noises.size(0), -1)
        n1 = noises.unsqueeze(1).expand(n, n, -1)
        n2 = noises.unsqueeze(0).expand(n, n, -1)
        noise_dist = torch.pow(n1 - n2, 2).mean(dim=2)

        return torch.exp(torch.mean(-noise_dist * img_dist))


def generate_balanced_labels(batch_size, num_classes, class_weights=None):
    """
    Generate class labels for synthetic batch, optionally weighted by
    actual class distribution across clients.
    """
    if class_weights is not None:
        probs = class_weights / class_weights.sum()
        labels = np.random.choice(num_classes, size=batch_size, p=probs)
    else:
        labels = np.arange(batch_size) % num_classes
        np.random.shuffle(labels)
    return torch.from_numpy(labels).long()


def compute_class_client_weights(client_class_counts, selected_clients):
    """
    Compute per-class, per-client weights based on how much data each
    client has for each class. Clients with more data for a class have
    more authority as teachers for that class.

    Returns: (num_classes, num_selected_clients) weight matrix
    """
    counts = np.array([client_class_counts[cid] for cid in selected_clients])
    # counts shape: (num_selected, num_classes)
    total_per_class = counts.sum(axis=0) + 1e-8
    weights = counts / total_per_class[np.newaxis, :]
    return weights.T  # (num_classes, num_selected)


def get_batch_weights(labels, cls_client_weights):
    """
    For each sample in the batch, get the per-client weight based on
    the sample's class label.

    Args:
        labels: (batch_size,) int tensor
        cls_client_weights: (num_classes, num_clients) array

    Returns:
        (batch_size, num_clients) tensor
    """
    labels_np = labels.cpu().numpy()
    batch_weights = cls_client_weights[labels_np]  # (batch_size, num_clients)
    return torch.from_numpy(batch_weights).float()


def load_model_from_params(model_template, params_array, device):
    """Load a model's parameters from a flattened numpy array."""
    model = copy.deepcopy(model_template)
    state_dict = model.state_dict()
    idx = 0
    for name, param in state_dict.items():
        length = param.numel()
        new_vals = torch.from_numpy(
            params_array[idx:idx + length].reshape(param.shape)
        ).float().to(device)
        state_dict[name] = new_vals
        idx += length
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def get_model_params(model):
    """Flatten all model parameters into a 1D numpy array."""
    params = []
    for param in model.state_dict().values():
        params.append(param.cpu().numpy().reshape(-1))
    return np.concatenate(params)


@torch.no_grad()
def get_teacher_ensemble_prediction(fake_images, client_models, batch_weights,
                                     num_classes, temperature=3.0):
    """
    Get weighted ensemble soft prediction from all teacher (client) models.

    Args:
        fake_images: (B, C, H, W) synthetic images
        client_models: list of client nn.Module models (in eval mode)
        batch_weights: (B, num_clients) per-sample, per-client weights
        num_classes: int
        temperature: softmax temperature for soft labels

    Returns:
        (B, num_classes) soft probability distribution
    """
    device = fake_images.device
    batch_weights = batch_weights.to(device)
    ensemble_logits = torch.zeros(fake_images.size(0), num_classes, device=device)

    for i, t_model in enumerate(client_models):
        t_logit = t_model(fake_images)
        t_soft = F.softmax(t_logit / temperature, dim=1)
        # Weight by this client's authority per sample
        w = batch_weights[:, i].unsqueeze(1)  # (B, 1)
        ensemble_logits += t_soft * w

    return ensemble_logits


def server_knowledge_distillation(
    global_model,
    client_params_list,
    selected_clients,
    client_class_counts,
    model_template,
    ddpm_model,
    ddpm_scheduler,
    ddim_generate_fn,
    num_classes=10,
    kd_iterations=10,
    kd_batch_size=256,
    kd_lr=0.01,
    temperature=3.0,
    fid_weights=None,
    device='cuda',
):
    """
    Server-side knowledge distillation using DDIM-generated synthetic data.

    This is the CORE of the methodology:
    1. Generate synthetic images with DDIM (fast)
    2. Feed through each client model (teachers)
    3. Get weighted ensemble prediction
    4. Train global model (student) to match ensemble

    Args:
        global_model:       The FedAvg-aggregated global model (will be fine-tuned)
        client_params_list: Dict {client_id: np.array of flattened params}
        selected_clients:   List of client IDs selected this round
        client_class_counts: Dict {client_id: {class_id: count}}
        model_template:     A fresh model instance (for loading client params)
        ddpm_model:         Pretrained DDPM model (for DDIM sampling)
        ddpm_scheduler:     DDIM scheduler
        ddim_generate_fn:   The ddim_generate function
        num_classes:        Number of classes (10 for CIFAR-10)
        kd_iterations:      Number of distillation steps per round
        kd_batch_size:      Batch size for synthetic data
        kd_lr:              Learning rate for student training
        temperature:        KD temperature (higher = softer labels)
        fid_weights:        Optional per-class quality weights from FID scores
                            e.g., {0: 1.0, 1: 1.0, 2: 0.5, ...}
                            Classes with high FID get lower weight
        device:             'cuda' or 'cpu'

    Returns:
        fine_tuned_model:   The improved global model
        kd_losses:          List of KD losses per iteration
    """
    # 1. Prepare student model
    student = copy.deepcopy(global_model).to(device)
    student.train()

    optimizer = optim.SGD(student.parameters(), lr=kd_lr, momentum=0.9, weight_decay=1e-4)
    kl_loss_fn = nn.KLDivLoss(reduction='batchmean')

    # 2. Load teacher models (selected clients only)
    teachers = []
    for cid in selected_clients:
        t_model = load_model_from_params(model_template, client_params_list[cid], device)
        for p in t_model.parameters():
            p.requires_grad = False
        teachers.append(t_model)

    # 3. Compute class-client weights
    cls_client_weights = compute_class_client_weights(
        client_class_counts, selected_clients
    )

    # 4. Compute class generation weights (combine distribution + FID quality)
    total_class_counts = np.zeros(num_classes)
    for cid in selected_clients:
        for c in range(num_classes):
            total_class_counts[c] += client_class_counts[cid].get(c, 0)

    gen_weights = total_class_counts.copy()
    if fid_weights is not None:
        for c in range(num_classes):
            gen_weights[c] *= fid_weights.get(c, 1.0)

    # 5. Distillation loop
    kd_losses = []

    # Inner loop structure matching base papers:
    # Generator side: 1 step, Student side: 5 steps
    inner_student_steps = 5

    for step in range(kd_iterations):
        # Generate class-balanced labels
        labels = generate_balanced_labels(kd_batch_size, num_classes, gen_weights)
        labels = labels.to(device)

        # Generate synthetic images using DDIM (fast, 100 steps)
        fake_images_list = []
        for c in range(num_classes):
            mask = (labels == c)
            n_c = mask.sum().item()
            if n_c > 0:
                imgs = ddim_generate_fn(
                    ddpm_model, ddpm_scheduler,
                    class_id=c, n_samples=n_c,
                    num_steps=100, eta=0.0,
                    device=device, batch_size=min(n_c, 64)
                )
                fake_images_list.append((imgs, c, n_c))

        # Reconstruct batch in label order
        fake_batch = torch.zeros(kd_batch_size, 3, 32, 32, device=device)
        for imgs, c, n_c in fake_images_list:
            mask = (labels == c)
            idxs = mask.nonzero(as_tuple=True)[0][:n_c]
            fake_batch[idxs] = imgs[:len(idxs)].to(device)

        # Normalize to match CIFAR-10 preprocessing
        mean_t = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
        std_t = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
        fake_batch_norm = (fake_batch - mean_t) / std_t

        # Get teacher ensemble prediction
        batch_weights = get_batch_weights(labels, cls_client_weights)
        with torch.no_grad():
            teacher_soft = get_teacher_ensemble_prediction(
                fake_batch_norm, teachers, batch_weights,
                num_classes, temperature
            )

        # Apply FID quality weighting per sample
        if fid_weights is not None:
            sample_quality = torch.ones(kd_batch_size, device=device)
            for c in range(num_classes):
                mask = (labels == c)
                sample_quality[mask] = fid_weights.get(c, 1.0)
            # Scale teacher confidence by quality
            teacher_soft = teacher_soft * sample_quality.unsqueeze(1)
            teacher_soft = teacher_soft / teacher_soft.sum(dim=1, keepdim=True)

        # Train student (inner loop: 5 steps on same batch)
        for _ in range(inner_student_steps):
            optimizer.zero_grad()
            student_logits = student(fake_batch_norm)
            student_log_soft = F.log_softmax(student_logits / temperature, dim=1)

            # KL divergence loss (scaled by T^2 as per Hinton et al.)
            loss = kl_loss_fn(student_log_soft, teacher_soft) * (temperature ** 2)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            optimizer.step()

        kd_losses.append(loss.item())

        if (step + 1) % 5 == 0:
            print(f"    KD step {step+1}/{kd_iterations}, loss={loss.item():.4f}")

    # Cleanup teacher models
    del teachers
    torch.cuda.empty_cache()

    student.eval()
    return student, kd_losses
