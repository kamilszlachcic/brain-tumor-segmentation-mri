import os
import sys
import nibabel as nib
import numpy as np
from pathlib import Path
import random
import shutil
from tqdm import tqdm
from skimage.morphology import remove_small_objects, binary_closing
from skimage.measure import label, regionprops
from scipy.ndimage import center_of_mass

# === Parametry ===
OUTPUT_SHAPE = (128, 128, 16)
NUM_CLASSES = 4
MAX_TRIES = 1000
BG_THRESHOLD = 0.95
VAL_RATIO = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))


# === Standaryzacja ===
def standardize(image):
    standardized_image = np.zeros(image.shape)
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            slice_ = image[c, :, :, z]
            centered = slice_ - np.mean(slice_)
            if np.std(centered) != 0:
                centered /= np.std(centered)
            standardized_image[c, :, :, z] = centered
    return standardized_image


# === ROI maska (zgodna z artykułem Nature) ===
def compute_roi_mask(image):
    t1ce, t2, flair = image[1], image[2], image[3]
    flair_bin = flair > 0.7
    t2_bin = t2 > 0.7
    t1ce_bin = t1ce > 0.9
    flair_t2_mask = np.logical_and(flair_bin, t2_bin)

    roi_mask = np.zeros_like(t1ce_bin, dtype=bool)
    for z in range(t1ce_bin.shape[2]):
        labeled = label(t1ce_bin[:, :, z])
        for region in regionprops(labeled):
            if region.solidity > 0.7 and region.area > 500 and region.major_axis_length > 35:
                tumor_candidate = labeled == region.label
                overlap = np.logical_and(tumor_candidate, flair_t2_mask[:, :, z])
                if np.sum(overlap) > 20:
                    roi_mask[:, :, z] = np.logical_or(roi_mask[:, :, z], tumor_candidate)

    return roi_mask


# === Ekstrakcja patchy ===
def get_sub_volume(image, label, roi_mask,
                   output_x=128, output_y=128, output_z=16,
                   num_classes=4, max_tries=2000, background_threshold=0.95):
    orig_x, orig_y, orig_z, _ = image.shape
    tries = 0

    while tries < max_tries:
        start_x = np.random.randint(0, orig_x - output_x + 1)
        start_y = np.random.randint(0, orig_y - output_y + 1)
        start_z = np.random.randint(0, orig_z - output_z + 1)

        roi_patch = roi_mask[start_x:start_x + output_x,
                             start_y:start_y + output_y,
                             start_z:start_z + output_z]

        if not np.any(roi_patch):
            tries += 1
            continue

        y = label[start_x:start_x + output_x,
                  start_y:start_y + output_y,
                  start_z:start_z + output_z]
        y = np.eye(num_classes)[y]
        bgrd_ratio = np.sum(y[:, :, :, 0]) / (output_x * output_y * output_z)
        tries += 1

        if bgrd_ratio < background_threshold:
            X = image[start_x:start_x + output_x,
                      start_y:start_y + output_y,
                      start_z:start_z + output_z, :]
            X = np.moveaxis(X, 3, 0)
            y = np.moveaxis(y, 3, 0)[1:]  # remove background
            return standardize(X), y

    return None, None


# === Fallback z centroidu ===
def extract_patch_from_centroid(image, label, output_shape):
    tumor_mask = label > 0
    if not np.any(tumor_mask):
        return None, None

    center = np.round(center_of_mass(tumor_mask)).astype(int)
    start_x = max(center[0] - output_shape[0] // 2, 0)
    start_y = max(center[1] - output_shape[1] // 2, 0)
    start_z = max(center[2] - output_shape[2] // 2, 0)

    start_x = min(start_x, image.shape[0] - output_shape[0])
    start_y = min(start_y, image.shape[1] - output_shape[1])
    start_z = min(start_z, image.shape[2] - output_shape[2])

    patch_img = image[start_x:start_x + output_shape[0],
                      start_y:start_y + output_shape[1],
                      start_z:start_z + output_shape[2], :]
    patch_lbl = label[start_x:start_x + output_shape[0],
                      start_y:start_y + output_shape[1],
                      start_z:start_z + output_shape[2]]
    patch_img = np.moveaxis(patch_img, 3, 0)
    patch_lbl = np.eye(NUM_CLASSES)[patch_lbl]
    patch_lbl = np.moveaxis(patch_lbl, 3, 0)[1:]
    return standardize(patch_img), patch_lbl


# === Główna funkcja ===
def preprocess_all(images_dir, labels_dir, output_dir, patches_per_subject=5):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir_X = Path(output_dir) / "X"
    output_dir_y = Path(output_dir) / "y"
    output_dir_X.mkdir(parents=True, exist_ok=True)
    output_dir_y.mkdir(parents=True, exist_ok=True)

    existing_ids = {f.name.split('_patch')[0].split('_centroid')[0] for f in output_dir_X.glob("*.npz")}
    print(f"Found {len(existing_ids)} existing subjects.")

    for filename in sorted(os.listdir(images_dir)):
        if not filename.endswith('.nii.gz') or filename.startswith("._"):
            continue

        subject_id = filename.replace(".nii.gz", "")
        if subject_id in existing_ids:
            continue

        print(f"Processing {subject_id}...")
        img = nib.load(images_dir / filename).get_fdata()
        label = nib.load(labels_dir / filename).get_fdata().astype(np.uint8)
        roi_mask = compute_roi_mask(np.moveaxis(img, 3, 0))

        saved = False
        for i in range(patches_per_subject):
            X, y = get_sub_volume(img, label, roi_mask)
            if X is not None:
                np.savez_compressed(output_dir_X / f"{subject_id}_patch{i}.npz", X=X)
                np.savez_compressed(output_dir_y / f"{subject_id}_patch{i}.npz", y=y)
                saved = True

        if not saved:
            print(f" → fallback: centroid patch for {subject_id}")
            Xc, yc = extract_patch_from_centroid(img, label, OUTPUT_SHAPE)
            if Xc is not None:
                np.savez_compressed(output_dir_X / f"{subject_id}_centroid_patch.npz", X=Xc)
                np.savez_compressed(output_dir_y / f"{subject_id}_centroid_patch.npz", y=yc)
            else:
                print(f" ✘ Failed to process {subject_id} (no tumor found)")


# === Podział na train/val ===
def split_dataset(images_dir, masks_dir, output_dir, val_ratio=VAL_RATIO, seed=42):
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    for subset in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, subset, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, subset, 'masks'), exist_ok=True)

    filenames = sorted([f for f in os.listdir(images_dir) if f.endswith('.npz')])
    random.shuffle(filenames)
    val_count = int(len(filenames) * val_ratio)
    val_set = set(filenames[:val_count])

    for fname in filenames:
        subset = 'val' if fname in val_set else 'train'
        shutil.copy2(os.path.join(images_dir, fname), os.path.join(output_dir, subset, 'images', fname))
        shutil.copy2(os.path.join(masks_dir, fname), os.path.join(output_dir, subset, 'masks', fname))

    print(f"Split complete: {len(filenames) - val_count} train / {val_count} val")


if __name__ == "__main__":
    preprocess_all(
        images_dir=PROJECT_ROOT / "data/raw/imagesTr",
        labels_dir=PROJECT_ROOT / "data/raw/labelsTr",
        output_dir=PROJECT_ROOT / "data/patches",
        patches_per_subject=5
    )

    split_dataset(
        images_dir=PROJECT_ROOT / "data/patches/X",
        masks_dir=PROJECT_ROOT / "data/patches/y",
        output_dir=PROJECT_ROOT / "data/processed"
    )
