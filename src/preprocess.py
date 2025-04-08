import os
import nibabel as nib
import numpy as np

def load_decathlon_image(image_path):
    """Load multi-channel MRI from .nii.gz (shape: H, W, D, C or C, H, W, D)."""
    img = nib.load(image_path)
    data = img.get_fdata()

    # Transpose if channels are last
    if data.ndim == 4 and data.shape[-1] <= 4:
        data = np.transpose(data, (3, 0, 1, 2))  # -> (C, H, W, D)

    return data.astype(np.float32)

def load_decathlon_mask(mask_path):
    """Load ground truth mask (shape: H, W, D)."""
    mask = nib.load(mask_path).get_fdata()
    return mask.astype(np.int64)

def normalize_per_channel(volume):
    """Normalize each channel to zero mean and unit variance."""
    norm_volume = np.zeros_like(volume)
    for c in range(volume.shape[0]):
        channel = volume[c]
        mean = channel.mean()
        std = channel.std()
        norm_volume[c] = (channel - mean) / (std + 1e-8)
    return norm_volume

def preprocess_and_save_all(images_dir, labels_dir, output_dir):
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    for filename in sorted(os.listdir(images_dir)):
        if not filename.endswith('.nii.gz') or filename.startswith("._"):
            continue

        name = filename.replace('.nii.gz', '')
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, f"{name}.nii.gz")

        image = load_decathlon_image(image_path)
        mask = load_decathlon_mask(label_path)

        image = normalize_per_channel(image)
        mask = mask.astype(np.uint8)

        nib.save(nib.Nifti1Image(image, np.eye(4)), os.path.join(output_dir, 'images', f"{name}.nii.gz"))
        nib.save(nib.Nifti1Image(mask, np.eye(4)), os.path.join(output_dir, 'masks', f"{name}.nii.gz"))

        print(f"Saved preprocessed sample: {name}")
