# src/eval_mvtec.py
"""
Evaluate MVTec anomaly detection model: image-level ROC + example montages.
Usage:
  python src/eval_mvtec.py --root data/mvtec_anomaly_detection --category bottle --model results/tiny_ae.pth --img_size 64
"""
import os, argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
import torch
from torch.utils.data import DataLoader
from PIL import Image
import cv2

from mvtec_dataset import MVTecDataset

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Simple TinyAE architecture (must match training)
import torch.nn as nn
class TinyAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def tensor_to_numpy(img_tensor):
    t = img_tensor.detach().cpu()
    if t.dim() == 3 and t.shape[0] == 1:
        arr = t.squeeze(0).numpy()
    elif t.dim() == 2:
        arr = t.numpy()
    else:
        arr = t.squeeze().numpy()
    return arr

def save_roc_curve(y_true, scores, out_path):
    fpr, tpr, th = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5,5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Image-level ROC"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.savefig(out_path, dpi=150); plt.close()
    return roc_auc

def save_montage(examples, out_path, ncols=2):
    imgs = []
    for e in examples:
        im = e.copy()
        if im.ndim == 2:
            im = np.stack([im,im,im], axis=-1)
        im = np.clip(im, 0.0, 1.0)
        im = (im*255).astype(np.uint8)
        imgs.append(im)
    h,w,_ = imgs[0].shape
    n = len(imgs)
    nrows = int(np.ceil(n / ncols))
    canvas = np.ones((h*nrows, w*ncols, 3), dtype=np.uint8) * 255
    for i,img in enumerate(imgs):
        r = i // ncols; c = i % ncols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
    Image.fromarray(canvas).save(out_path)

def evaluate_and_save(model, test_loader, results_dir, img_size=64):
    model.eval()
    y_true = []
    scores = []
    originals = []
    reconstructions = []
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x)
            err = torch.mean((out - x) ** 2, dim=(1,2,3)).cpu().numpy()
            scores.extend(err.tolist()); y_true.extend(y.numpy().tolist())
            for i in range(x.shape[0]):
                originals.append(tensor_to_numpy(x[i]))
                reconstructions.append(tensor_to_numpy(out[i]))
    roc_path = os.path.join(results_dir, "roc.png")
    img_auc = save_roc_curve(y_true, scores, roc_path)
    print("Saved ROC ->", roc_path, " AUC=", img_auc)
    # montages
    idxs = np.argsort(scores)[::-1]
    top = idxs[:8]; low = idxs[-8:]
    top_examples = []
    low_examples = []
    def stack_triplet(orig, recon):
        a = orig if orig.ndim==2 else orig[:,:,0]
        b = recon if recon.ndim==2 else recon[:,:,0]
        err = np.clip(np.abs(recon - orig), 0, 1)
        c = err if err.ndim==2 else err[:,:,0]
        def norm01(x):
            mn, mx = x.min(), x.max()
            if mx - mn < 1e-6: return np.zeros_like(x)
            return (x - mn) / (mx - mn)
        a,b,c = norm01(a), norm01(b), norm01(c)
        return np.concatenate([a,b,c], axis=1)
    for i in top:
        top_examples.append(stack_triplet(originals[i], reconstructions[i]))
    for i in low:
        low_examples.append(stack_triplet(originals[i], reconstructions[i]))
    top_path = os.path.join(results_dir, "top_anomalies.png")
    low_path = os.path.join(results_dir, "top_normals.png")
    save_montage(top_examples, top_path, ncols=2)
    save_montage(low_examples, low_path, ncols=2)
    print("Saved montages ->", top_path, low_path)
    # summary
    with open(os.path.join(results_dir,"summary.txt"), "w") as f:
        f.write(f"Image AUC: {img_auc:.6f}\n")
    return img_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="path to mvtec root folder")
    parser.add_argument("--category", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    test_ds = MVTecDataset(root=args.root, category=args.category, mode="test", img_size=args.img_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    model = TinyAE()
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    evaluate_and_save(model, test_loader, RESULTS_DIR, args.img_size)
    print("Evaluation complete. Results in:", RESULTS_DIR)

if __name__ == "__main__":
    import argparse
    main()
