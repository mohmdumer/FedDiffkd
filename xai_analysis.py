"""
Explainability Analysis: Grad-CAM & SHAP
==========================================
Run this AFTER experiments E1/E2/E3 are complete.
Uses the _best.pt checkpoints to generate visual explanations.

Requirements (install in first Kaggle cell):
    !pip install grad-cam shap

Setup:
    - Attach your checkpoint dataset containing E1_Baseline_best.pt,
      E2_FedAvg_KD_best.pt, E3_FedAvg_KD_FID_best.ptip
    - Update CKPT_PATHS below with the correct paths
"""

import os, copy, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# LIME
from lime import lime_image
from skimage.segmentation import mark_boundaries

from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
CLASS_NAMES  = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']

# ── UPDATE THESE PATHS to match your Kaggle input dataset ──
CKPT_PATHS = {
    'E1_Baseline':      r'H:\MS\Masters Thesis\Implementation\thesis_fl_package\apni_methodology_Results\E1_baseline_checkpoints\E2_FedAvg_KD_best.pt',
    'E2_FedAvg_KD':     r'H:\MS\Masters Thesis\Implementation\thesis_fl_package\apni_methodology_Results\E2\checkpoints\E2_FedAvg_KD_best.pt',
    'E3_FedAvg_KD_FID': r'H:\MS\Masters Thesis\Implementation\thesis_fl_package\apni_methodology_Results\E3\checkpoints\E3_FedAvg_KD_FID_best.pt',
}

OUTPUT_DIR = Path('xai_results')
GRADCAM_DIR = OUTPUT_DIR / 'gradcam'
LIME_DIR    = OUTPUT_DIR / 'lime'
for d in [GRADCAM_DIR, LIME_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# How many test images per class to analyze
SAMPLES_PER_CLASS = 3


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL (must match training exactly)
# ═══════════════════════════════════════════════════════════════════════════════

def create_resnet18(num_classes=10):
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_model(ckpt_path):
    """Load a trained model from checkpoint."""
    model = create_resnet18()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(DEVICE).eval()
    print(f"  Loaded: {ckpt_path} (Round {ckpt.get('round','?')}, "
          f"Acc={ckpt.get('accuracy','?')})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_test_data():
    """Load CIFAR-10 test set. Returns normalized tensors AND raw [0,1] images."""
    raw_test = CIFAR10(root=r'H:\MS\Masters Thesis\Implementation\thesis_fl_package\apni_methodology_Results\WGAN-GP\data', train=False, download=False)

    mean_t = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std_t  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)

    # Raw images in [0,1] for visualization
    raw_imgs = torch.from_numpy(raw_test.data.astype(np.float32) / 255.0)  # (N,32,32,3)
    
    # Normalized images for model input
    norm_imgs = (raw_imgs.permute(0, 3, 1, 2) - mean_t) / std_t  # (N,3,32,32)
    labels = np.array(raw_test.targets)

    return norm_imgs, raw_imgs.numpy(), labels


def select_samples(labels, samples_per_class=3):
    """Select a few correctly diverse samples per class for analysis."""
    selected = []
    for c in range(10):
        idxs = np.where(labels == c)[0]
        chosen = np.random.choice(idxs, size=min(samples_per_class, len(idxs)), replace=False)
        selected.extend(chosen.tolist())
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_gradcam(models_dict, norm_imgs, raw_imgs, labels, sample_indices):
    """
    Generate Grad-CAM heatmaps for selected images across all models.
    Creates side-by-side comparison plots.
    """
    print("\n" + "="*60)
    print("GRAD-CAM ANALYSIS")
    print("="*60)

    model_names = list(models_dict.keys())
    n_models = len(model_names)

    # Build GradCAM objects — target last conv layer of ResNet-18
    cams = {}
    for name, model in models_dict.items():
        target_layer = [model.layer4[-1]]  # Last residual block
        cams[name] = GradCAM(model=model, target_layers=target_layer)

    # Group samples by class
    for class_id in range(10):
        class_samples = [i for i in sample_indices if labels[i] == class_id]
        if not class_samples:
            continue

        for idx in class_samples:
            fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

            # Original image
            raw_img = raw_imgs[idx]  # (32,32,3) in [0,1]
            axes[0].imshow(raw_img)
            axes[0].set_title(f"Original\n{CLASS_NAMES[class_id]}", fontsize=12, fontweight='bold')
            axes[0].axis('off')

            # Grad-CAM for each model
            input_tensor = norm_imgs[idx].unsqueeze(0).to(DEVICE)
            targets = [ClassifierOutputTarget(class_id)]

            for j, name in enumerate(model_names):
                grayscale_cam = cams[name](input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]  # (32, 32)

                # Overlay heatmap on original image
                visualization = show_cam_on_image(raw_img, grayscale_cam, use_rgb=True)

                # Get model's prediction
                with torch.no_grad():
                    pred = models_dict[name](input_tensor).argmax(1).item()
                pred_str = CLASS_NAMES[pred]
                correct = "✓" if pred == class_id else "✗"

                axes[j + 1].imshow(visualization)
                axes[j + 1].set_title(f"{name}\nPred: {pred_str} {correct}", fontsize=10)
                axes[j + 1].axis('off')

            plt.suptitle(f"Grad-CAM: {CLASS_NAMES[class_id]} (Test #{idx})",
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            save_path = GRADCAM_DIR / f"gradcam_{CLASS_NAMES[class_id]}_{idx}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    # Clean up
    for cam in cams.values():
        del cam
    torch.cuda.empty_cache()

    print(f"  OK: Grad-CAM heatmaps saved to {GRADCAM_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# LIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_lime(models_dict, norm_imgs, raw_imgs, labels, sample_indices):
    """
    Generate LIME explanations for selected images across all models.
    """
    print("\n" + "="*60)
    print("LIME ANALYSIS")
    print("="*60)

    model_names = list(models_dict.keys())
    n_models = len(model_names)

    explainer = lime_image.LimeImageExplainer()

    for class_id in range(10):
        class_samples = [i for i in sample_indices if labels[i] == class_id]
        if not class_samples:
            continue

        # Take first sample per class for LIME (it can be expensive)
        idx = class_samples[0]
        # LIME expects images in (H, W, C) format with values [0, 1] or [0, 255]. 
        # Our raw_imgs are (32, 32, 3) in [0, 1]. LIME works well with double.
        test_image = raw_imgs[idx].astype(np.double)

        fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 4))

        # Original image
        axes[0].imshow(raw_imgs[idx])
        axes[0].set_title(f"Original\n{CLASS_NAMES[class_id]}", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        for j, name in enumerate(model_names):
            model = models_dict[name]
            
            # Define prediction function for LIME
            def predict_fn(images):
                # LIME passes numpy array (N, 32, 32, 3)
                batch = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2) # (N, 3, 32, 32)
                # Normalize
                mean_t = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
                std_t  = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
                batch = (batch - mean_t) / std_t
                batch = batch.to(DEVICE)
                
                with torch.no_grad():
                    logits = model(batch)
                    probs = F.softmax(logits, dim=1)
                return probs.cpu().numpy()

            # Generate explanation
            explanation = explainer.explain_instance(
                test_image, 
                predict_fn, 
                top_labels=1, 
                hide_color=0, 
                num_samples=500,
                random_seed=42
            )

            # Get model's prediction
            input_tensor = norm_imgs[idx].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(input_tensor).argmax(1).item()
            
            # Show explanation for the predicted class
            temp, mask = explanation.get_image_and_mask(
                pred, positive_only=True, num_features=5, hide_rest=False
            )
            
            # Overlay boundaries
            img_boundry = mark_boundaries(temp, mask)
            
            pred_str = CLASS_NAMES[pred]
            correct = "(OK)" if pred == class_id else "(FAIL)"
            
            axes[j + 1].imshow(img_boundry)
            axes[j + 1].set_title(f"{name}\nPred: {pred_str} {correct}", fontsize=10)
            axes[j + 1].axis('off')

        plt.suptitle(f"LIME: {CLASS_NAMES[class_id]} (Test #{idx})",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = LIME_DIR / f"lime_{CLASS_NAMES[class_id]}_{idx}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  OK: LIME explanations saved to {LIME_DIR}")


# ═══════════════════════════════════════════════════════════════════════════════
# QUANTITATIVE XAI METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_cam_confidence(models_dict, norm_imgs, labels, sample_indices):
    """
    Compute average Grad-CAM activation intensity and prediction confidence
    per model. Higher CAM focus + higher confidence = better model.
    """
    print("\n" + "="*60)
    print("QUANTITATIVE XAI METRICS")
    print("="*60)

    results = {}

    for name, model in models_dict.items():
        target_layer = [model.layer4[-1]]
        cam = GradCAM(model=model, target_layers=target_layer)

        cam_scores = []
        confidences = []
        correct = 0

        for idx in sample_indices:
            true_label = labels[idx]
            input_tensor = norm_imgs[idx].unsqueeze(0).to(DEVICE)

            # Grad-CAM intensity
            targets = [ClassifierOutputTarget(true_label)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
            cam_scores.append(grayscale_cam.mean())

            # Prediction confidence
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred = probs.argmax(1).item()
                conf = probs[0, true_label].item()
                confidences.append(conf)
                if pred == true_label:
                    correct += 1

        acc = correct / len(sample_indices)
        results[name] = {
            'accuracy_on_samples': round(acc, 4),
            'avg_cam_intensity': round(float(np.mean(cam_scores)), 4),
            'avg_confidence': round(float(np.mean(confidences)), 4),
        }

        print(f"  {name}:")
        print(f"    Sample Accuracy:    {acc:.4f}")
        print(f"    Avg CAM Intensity:  {np.mean(cam_scores):.4f}")
        print(f"    Avg Confidence:     {np.mean(confidences):.4f}")

        del cam

    # Save metrics
    with open(OUTPUT_DIR / 'xai_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  OK: Quantitative XAI metrics saved to {OUTPUT_DIR / 'xai_metrics.json'}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    np.random.seed(42)
    torch.manual_seed(42)

    # ── Load models ──
    print("Loading trained models...")
    models_dict = {}
    for name, path in CKPT_PATHS.items():
        if os.path.exists(path):
            models_dict[name] = load_model(path)
        else:
            print(f"  [!] Skipping {name} — file not found: {path}")

    if not models_dict:
        raise FileNotFoundError(
            "No checkpoint files found! Update CKPT_PATHS at the top of this script."
        )

    # ── Load test data ──
    print("\nLoading CIFAR-10 test set...")
    norm_imgs, raw_imgs, labels = load_test_data()

    # ── Select samples ──
    sample_indices = select_samples(labels, samples_per_class=SAMPLES_PER_CLASS)
    print(f"Selected {len(sample_indices)} test images for analysis "
          f"({SAMPLES_PER_CLASS} per class)")

    # ── Run Grad-CAM ──
    run_gradcam(models_dict, norm_imgs, raw_imgs, labels, sample_indices)

    # ── Run LIME ──
    run_lime(models_dict, norm_imgs, raw_imgs, labels, sample_indices)

    # ── Quantitative Metrics ──
    compute_cam_confidence(models_dict, norm_imgs, labels, sample_indices)

    print("\n" + "="*60)
    print("ALL XAI ANALYSIS COMPLETE!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)
