# src/mvtec_dataset.py
import os
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MVTecDataset(Dataset):
    """
    PyTorch Dataset for MVTec Anomaly Detection dataset.
    root: path to mvtec_anomaly_detection folder
    category: e.g., 'bottle'
    mode: 'train' (only normal images) or 'test' (normal + anomalies)
    img_size: resize size (height = width = img_size)
    """
    def __init__(self, root, category="bottle", mode="train", img_size=64):
        self.root = root
        self.category = category
        self.mode = mode
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])

        self.samples = []
        self.labels = []

        if mode == "train":
            train_path = os.path.join(root, category, "train", "good")
            files = sorted(glob(os.path.join(train_path, "*.png")))
            for f in files:
                self.samples.append(f)
                self.labels.append(0)
        elif mode == "test":
            test_path = os.path.join(root, category, "test")
            # test folders contain 'good' and defect directories
            for defect_type in sorted(os.listdir(test_path)):
                defect_dir = os.path.join(test_path, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                files = sorted(glob(os.path.join(defect_dir, "*.png")))
                label = 0 if defect_type == "good" else 1
                for f in files:
                    self.samples.append(f)
                    self.labels.append(label)
        else:
            raise ValueError("mode must be 'train' or 'test'")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path)
        img = self.transform(img)
        return img, label
