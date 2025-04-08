import os
import torch
import csv
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dataset import BrainTumorDataset
from model import UNet3D
from metrics import dice_coefficient, iou_score
import nibabel as nib

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]



def evaluate_model(model, dataloader, device):
    model.eval()
    dice_scores, iou_scores = [], []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            dice_scores.append(dice)
            iou_scores.append(iou)

    avg_dice = sum(dice_scores) / len(dice_scores)
    avg_iou = sum(iou_scores) / len(iou_scores)
    return avg_dice, avg_iou


def extract_metrics(logdir):
    acc = EventAccumulator(logdir)
    acc.Reload()
    data = {}
    for tag in acc.Tags()['scalars']:
        data[tag] = [(x.step, x.value) for x in acc.Scalars(tag)]
    return data


def plot_metrics(metrics, save_path=None):
    plt.figure(figsize=(10, 6))
    for tag, values in metrics.items():
        steps, vals = zip(*values)
        plt.plot(steps, vals, label=tag)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training & Validation Metrics")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"📊 Plot saved to: {save_path}")
    else:
        plt.show()


def save_results_csv(dice, iou, model_path, log_dir, save_path= PROJECT_ROOT/"results/metrics.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    header = ["timestamp", "model_path", "log_dir", "val_dice", "val_iou"]
    row = [
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        model_path,
        log_dir,
        round(dice, 4),
        round(iou, 4)
    ]
    write_header = not os.path.exists(save_path)
    with open(save_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"✅ Results saved to: {save_path}")


if __name__ == "__main__":
    # === Ewaluacja modelu ===
    data_dir = PROJECT_ROOT /"data/processed"
    model_path = PROJECT_ROOT /"runs/experiment_1/best_model.pth"
    log_dir = PROJECT_ROOT /"runs/experiment_1"
    results_csv = PROJECT_ROOT /"results/metrics.csv"
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = BrainTumorDataset(
        images_dir=os.path.join(data_dir, "val", "images"),
        masks_dir=os.path.join(data_dir, "val", "masks")
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Ustalenie liczby kanałów wejściowych
    sample_image_path = os.path.join(data_dir, "val/images", os.listdir(os.path.join(data_dir, "val/images"))[0])
    in_channels = nib.load(sample_image_path).get_fdata().shape[3] if nib.load(sample_image_path).get_fdata().ndim == 4 else 1

    model = UNet3D(in_channels=in_channels, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Ocena
    avg_dice, avg_iou = evaluate_model(model, val_loader, device)
    print(f"\n🧪 Validation Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

    # Zapis wyników
    save_results_csv(avg_dice, avg_iou, model_path, log_dir, save_path=results_csv)

    # === Wizualizacja metryk z TensorBoard ===
    if os.path.exists(log_dir):
        metrics = extract_metrics(log_dir)
        plot_metrics(metrics, save_path= PROJECT_ROOT /"results/metrics_plot.png")
    else:
        print("⚠️ No TensorBoard logs found.")
