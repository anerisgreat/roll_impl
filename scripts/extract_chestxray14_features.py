"""
One-time feature extraction for NIH ChestX-ray14 using a TorchXRayVision
DenseNet-121 pretrained on the NIH dataset.

Expects:
  ~/datasets/chestxray14/images/          -- 224×224 images from Kaggle download
  ~/datasets/chestxray14/Data_Entry_2017.csv  -- label metadata

Produces:
  ~/datasets/chestxray14/features.npz     -- {features: (N,1024), labels: (N,), filenames: (N,)}

Usage (run after food101n finishes so GPU is free):
  nix develop --command python scripts/extract_chestxray14_features.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image
import concurrent.futures

DATA_DIR   = os.path.expanduser('~/datasets/chestxray14')
IMAGES_DIR = os.path.join(DATA_DIR, 'images-224', 'images-224')
CSV_PATH   = os.path.join(DATA_DIR, 'Data_Entry_2017.csv')
OUT_PATH   = os.path.join(DATA_DIR, 'features.npz')
BATCH_SIZE = 64

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load pretrained DenseNet-121 (NIH weights), strip classifier head
    print('Loading TorchXRayVision DenseNet-121 (NIH weights)...')
    model = xrv.models.DenseNet(weights='densenet121-res224-nih')
    model.eval()
    model.to(device)

    # Load labels
    df = pd.read_csv(CSV_PATH)
    filenames = df['Image Index'].tolist()
    print(f'Found {len(filenames)} images in CSV')

    # Filter to images that actually exist
    filenames = [f for f in filenames if os.path.exists(os.path.join(IMAGES_DIR, f))]
    print(f'{len(filenames)} images found on disk')

    def load_image(fname):
        path = os.path.join(IMAGES_DIR, fname)
        img = Image.open(path).convert('L')  # grayscale
        img = img.resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)  # (1, 224, 224)
        return arr

    n = len(filenames)
    FEAT_DIM = 1024
    tmp_path = OUT_PATH + '.tmp.npy'
    features = np.memmap(tmp_path, dtype='float32', mode='w+', shape=(n, FEAT_DIM))

    with torch.no_grad(), concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for start in range(0, n, BATCH_SIZE):
            batch_fnames = filenames[start:start + BATCH_SIZE]
            imgs = list(ex.map(load_image, batch_fnames))
            batch = torch.tensor(np.stack(imgs)).to(device)
            fmap = model.features(batch)              # (B, 1024, 7, 7)
            feats = F.adaptive_avg_pool2d(F.relu(fmap, inplace=True), 1)
            feats = feats.view(feats.size(0), -1)     # (B, 1024)
            end = start + len(batch_fnames)
            features[start:end] = feats.cpu().numpy()
            del imgs, batch, feats

            if (start // BATCH_SIZE) % 10 == 0:
                print(f'  {start}/{n} images processed', flush=True)

    # Free model and GPU memory before the save step to avoid OOM
    del model
    torch.cuda.empty_cache()

    # Save raw labels (pipe-separated) and filenames
    fname_to_label = dict(zip(df['Image Index'], df['Finding Labels']))
    labels_str = np.array([fname_to_label.get(f, 'No Finding') for f in filenames])

    # Pass memmap directly — avoids loading all 438MB into RAM as a copy
    np.savez_compressed(OUT_PATH,
                        features=features,
                        labels=labels_str,
                        filenames=np.array(filenames))
    del features
    os.remove(tmp_path)
    print(f'\nSaved {n} × {FEAT_DIM} features to {OUT_PATH}')
    print(f'File size: {os.path.getsize(OUT_PATH) / 1e6:.1f} MB')

if __name__ == '__main__':
    main()
