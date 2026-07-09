"""Custom dataset loaders."""
import os
import csv
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


def download_and_extract_aic4(dataset_root):
    root = Path(dataset_root) / "AIC4"
    csv_path = root / "JPEG_AIC_reconstructed_jnd_scores.csv"
    
    if root.exists() and csv_path.exists():
        return root

    root.mkdir(parents=True, exist_ok=True)
    zip_url = "https://aicdb.jpeg.org/JPEG_AIC-4_Sample_Dataset.zip"
    zip_path = root / "JPEG_AIC-4_Sample_Dataset.zip"
    
    print(f"Downloading AIC-4 dataset from {zip_url}...")
    urllib.request.urlretrieve(zip_url, zip_path)
    
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(root)
        
    csv_url = "https://raw.githubusercontent.com/jpeg-aic/JPEG-AIC-4-datasets/main/JPEG_AIC_reconstructed_jnd_scores.csv"
    print(f"Downloading scores from {csv_url}...")
    urllib.request.urlretrieve(csv_url, csv_path)
    
    return root


class AIC4Dataset(Dataset):
    """
    JPEG AIC-4 Dataset.
    Loads subjective JND scores (distortion column) and dynamically resolves nested image paths.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = download_and_extract_aic4(root_dir)
        self.transform = transform
        self.samples = []
        
        # Build an index of all images in the extracted folder to allow fast lookup by filename
        # This handles the nested directory structure dynamically.
        self.image_paths = {p.name: p for p in self.root_dir.rglob("*.png")}
        
        csv_path = self.root_dir / "JPEG_AIC_reconstructed_jnd_scores.csv"
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref_name = row['img_source']
                dis_name = row['img_distorted']
                
                # Check if images exist in our mapping
                if ref_name in self.image_paths and dis_name in self.image_paths:
                    self.samples.append({
                        "ref_img_path": str(self.image_paths[ref_name]),
                        "dis_img_path": str(self.image_paths[dis_name]),
                        "score": float(row['distortion'])
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ref_img = Image.open(sample["ref_img_path"]).convert('RGB')
        dis_img = Image.open(sample["dis_img_path"]).convert('RGB')
        
        if self.transform:
            ref_img = self.transform(ref_img)
            dis_img = self.transform(dis_img)
            
        return {
            "ref_img": ref_img,
            "dis_img": dis_img,
            "score": sample["score"],
            "ref_img_path": sample["ref_img_path"],
            "dis_img_path": sample["dis_img_path"]
        }


def get_aic4_dataset(dataset_root, transform=None):
    return AIC4Dataset(dataset_root, transform=transform)
