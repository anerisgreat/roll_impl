import sys, time, torch
sys.path.insert(0, ".")
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.datasets import Food101NDataset
from src.networks import ConvNet
from src.utils import get_device

print("Loading dataset...")
dataset = Food101NDataset()
device = get_device()
print(f"Device: {device}, dataset size: {len(dataset)}")

# Balanced sampler matching what basic_data_splitter does
n_pos = int(dataset.y.sum().item())
n_neg = len(dataset) - n_pos
weights = torch.where(dataset.y == 1, 1.0 / n_pos, 1.0 / n_neg)

model = ConvNet(image_size=64).to(device)
loss_fn = torch.nn.BCEWithLogitsLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"{'bs':>6}  {'ms/batch':>10}  {'img/s':>8}  {'batches/ep':>12}  {'est epoch':>10}  {'mem GB':>8}")
for bs in [256, 512, 1024, 2048, 4096]:
    try:
        sampler = WeightedRandomSampler(weights, num_samples=2 * n_pos, replacement=True)
        loader = DataLoader(dataset, batch_size=bs, sampler=sampler, num_workers=4, pin_memory=True)
        n_batches = len(loader)

        # warmup 3 batches
        for i, (x, y) in enumerate(loader):
            if i >= 3: break
            x, y = x.to(device), y.to(device)
            optim.zero_grad(); loss_fn(model(x), y).backward(); optim.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # time 10 batches
        t0 = time.perf_counter()
        for i, (x, y) in enumerate(loader):
            if i >= 10: break
            x, y = x.to(device), y.to(device)
            optim.zero_grad(); loss_fn(model(x), y).backward(); optim.step()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        ms_batch = elapsed / 10 * 1000
        imgs_s = bs * 10 / elapsed
        est_epoch = n_batches * elapsed / 10
        mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
        print(f"{bs:6d}  {ms_batch:10.1f}  {imgs_s:8.0f}  {n_batches:12d}  {est_epoch:8.0f}s  {mem:8.2f}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"{bs:6d}  OOM")
        else:
            print(f"{bs:6d}  ERROR: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
