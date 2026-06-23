"""
data_generation.py
-------------------
Objective
    Build a labelled character-image dataset for the 35 target classes
    (uppercase letters A-Z and digits 1-9) entirely on this machine, with
    no external dataset download. We render each character using many
    different installed system fonts and then apply random geometric and
    noise augmentations (rotation, scaling, translation, Gaussian noise,
    slight blur) to imitate the natural variability that real handwritten /
    scanned characters would have.

Why synthetic generation?
    The assignment requires >= 50 samples per class. Collecting real
    handwritten samples for 35 classes is impractical inside this
    environment, and no internet dataset download is available. Rendering
    characters with many fonts + randomised distortions is a standard,
    defensible way to manufacture a from-scratch dataset for an
    educational character-recognition pipeline, and it keeps the whole
    pipeline 100% reproducible from this single script.

Output
    A single compressed .npz file (data/raw_dataset.npz) containing:
        images : float32 array of shape (N, IMG_SIZE*IMG_SIZE), values in [0, 1]
        labels : int64 array of shape (N,)  -> class index 0..34
        classes: list[str] of length 35     -> index -> character mapping
"""

import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
IMG_SIZE = 28                       # final square image size (28x28, MNIST-style)
RENDER_SIZE = 64                    # render large then downscale -> smoother anti-aliasing
SAMPLES_PER_CLASS = 300             # comfortably above the 50-sample minimum
RANDOM_SEED = 42

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + list("123456789")
assert len(CLASSES) == 35, "There must be exactly 35 classes (26 letters + 9 digits)."

# A pool of visually different fonts already installed on this machine.
# Mixing serif / sans / mono / bold / italic fonts gives the model real
# style variation instead of 300 near-identical copies of one glyph.
WIN_FONTS = "C:/Windows/Fonts"
FONT_CANDIDATES = [
    # Windows fonts
    f"{WIN_FONTS}/arial.ttf",
    f"{WIN_FONTS}/arialbd.ttf",
    f"{WIN_FONTS}/ariali.ttf",
    f"{WIN_FONTS}/arialbi.ttf",
    f"{WIN_FONTS}/times.ttf",
    f"{WIN_FONTS}/timesbd.ttf",
    f"{WIN_FONTS}/timesi.ttf",
    f"{WIN_FONTS}/cour.ttf",
    f"{WIN_FONTS}/courbd.ttf",
    f"{WIN_FONTS}/verdana.ttf",
    f"{WIN_FONTS}/verdanab.ttf",
    f"{WIN_FONTS}/georgia.ttf",
    f"{WIN_FONTS}/georgiab.ttf",
    f"{WIN_FONTS}/calibri.ttf",
    f"{WIN_FONTS}/calibrib.ttf",
    f"{WIN_FONTS}/trebuc.ttf",
    f"{WIN_FONTS}/trebucbd.ttf",
    f"{WIN_FONTS}/comic.ttf",
    f"{WIN_FONTS}/comicbd.ttf",
    f"{WIN_FONTS}/tahoma.ttf",
    # Linux fonts (kept for cross-platform compatibility)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    "/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf",
]
FONTS = [f for f in FONT_CANDIDATES if os.path.exists(f)]
if len(FONTS) < 3:
    # Fallback: grab anything renderable from Windows or Linux font dirs.
    FONTS = (glob.glob(f"{WIN_FONTS}/*.ttf")[:10] or
             glob.glob("/usr/share/fonts/truetype/dejavu/*.ttf")[:5])
print(f"[data_generation] Using {len(FONTS)} fonts: {[os.path.basename(f) for f in FONTS]}")


def render_glyph(char: str, font_path: str, font_size: int) -> Image.Image:
    """
    Render a single character onto a blank canvas using the given font.

    Parameters
    ----------
    char : str
        The character to render (one of CLASSES).
    font_path : str
        Path to a .ttf font file.
    font_size : int
        Font size in pixels to render at (on the RENDER_SIZE canvas).

    Returns
    -------
    PIL.Image.Image
        A grayscale (mode 'L') RENDER_SIZE x RENDER_SIZE image with the
        character centered, white glyph on black background.
    """
    canvas = Image.new("L", (RENDER_SIZE, RENDER_SIZE), color=0)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(font_path, font_size)

    # Center the glyph using its actual rendered bounding box.
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (RENDER_SIZE - w) / 2 - bbox[0]
    y = (RENDER_SIZE - h) / 2 - bbox[1]
    draw.text((x, y), char, fill=255, font=font)
    return canvas


def augment(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """
    Apply a random combination of realistic distortions to one rendered
    glyph image, so that repeated samples of the same character/font are
    not pixel-identical.

    Distortions applied (each with randomised magnitude):
        - small rotation                (-15 deg to +15 deg)
        - scale jitter                  (85% - 115%)
        - translation jitter            (+/- 3 px)
        - slight Gaussian blur          (simulates scan/print softness)
        - additive Gaussian pixel noise (simulates sensor/scan noise)

    Parameters
    ----------
    img : PIL.Image.Image
        Source glyph image (grayscale).
    rng : numpy.random.Generator
        Seeded random generator, passed in so the whole script stays
        reproducible from one global seed.

    Returns
    -------
    PIL.Image.Image
        The augmented image, still RENDER_SIZE x RENDER_SIZE.
    """
    # Rotation
    angle = rng.uniform(-15, 15)
    img = img.rotate(angle, resample=Image.BICUBIC, fillcolor=0)

    # Scale jitter (resize then paste back onto a RENDER_SIZE canvas, centered)
    scale = rng.uniform(0.85, 1.15)
    new_size = max(8, int(RENDER_SIZE * scale))
    img = img.resize((new_size, new_size), resample=Image.BICUBIC)
    canvas = Image.new("L", (RENDER_SIZE, RENDER_SIZE), color=0)
    off = ((RENDER_SIZE - new_size) // 2, (RENDER_SIZE - new_size) // 2)
    canvas.paste(img, off)
    img = canvas

    # Translation jitter
    dx, dy = rng.integers(-3, 4), rng.integers(-3, 4)
    img = img.transform(
        img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0
    )

    # Slight blur (simulate scan softness) - applied ~50% of the time
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    return img


def add_pixel_noise(arr: np.ndarray, rng: np.random.Generator, sigma: float = 0.04) -> np.ndarray:
    """
    Add small additive Gaussian noise to a normalized [0, 1] image array
    and clip back into range. Mimics sensor/scan noise.

    Parameters
    ----------
    arr : np.ndarray
        Normalized image array, values in [0, 1].
    rng : numpy.random.Generator
        Seeded random generator.
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    np.ndarray
        Noisy array, clipped to [0, 1].
    """
    noisy = arr + rng.normal(0, sigma, size=arr.shape)
    return np.clip(noisy, 0.0, 1.0)


def build_dataset():
    """
    Generate the full synthetic dataset for all 35 classes and save it to
    disk as data/raw_dataset.npz.

    Returns
    -------
    images : np.ndarray, shape (N, IMG_SIZE*IMG_SIZE), dtype float32
    labels : np.ndarray, shape (N,), dtype int64
    classes: list[str]
    """
    rng = np.random.default_rng(RANDOM_SEED)
    images, labels = [], []

    for class_idx, char in enumerate(CLASSES):
        for _ in range(SAMPLES_PER_CLASS):
            font_path = FONTS[rng.integers(0, len(FONTS))]
            font_size = int(rng.integers(34, 46))  # vary glyph size too
            glyph = render_glyph(char, font_path, font_size)
            glyph = augment(glyph, rng)

            # Downscale to final IMG_SIZE with anti-aliasing, then normalize.
            small = glyph.resize((IMG_SIZE, IMG_SIZE), resample=Image.LANCZOS)
            arr = np.asarray(small, dtype=np.float32) / 255.0
            arr = add_pixel_noise(arr, rng)

            images.append(arr.flatten())
            labels.append(class_idx)

        print(f"[data_generation] class '{char}' ({class_idx+1}/35): "
              f"{SAMPLES_PER_CLASS} samples generated")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "raw_dataset.npz")
    np.savez_compressed(out_path, images=images, labels=labels, classes=np.array(CLASSES))
    print(f"[data_generation] Saved dataset: {images.shape[0]} samples, "
          f"{images.shape[1]} features each -> {out_path}")
    return images, labels, CLASSES


if __name__ == "__main__":
    build_dataset()
