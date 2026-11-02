"""Central configuration for VisionVoice.

Everything tunable lives here: directory layout, model hyper-parameters,
vocabulary boundaries, inference defaults, text-to-speech defaults and
live-stream defaults. Other modules import this file and never hard-code
paths or magic numbers.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

# Quiet TensorFlow's C++ logging before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")

# --------------------------------------------------------------------------- #
# Directory layout
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FLICKR_DIR = DATA_DIR / "flickr8k"          # put the extracted Flickr8k files here
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
CACHE_DIR = OUTPUT_DIR / "cache"

MODEL_PATH = MODEL_DIR / "caption_model.keras"
VOCAB_PATH = MODEL_DIR / "vocab.json"
FEATURES_CACHE_PATH = CACHE_DIR / "resnet50_features.npz"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"

# Dataset file names are searched recursively under FLICKR_DIR, so both the
# classic release (Flickr8k_Dataset + Flickr8k_text) and the common Kaggle
# layout (Images + captions.txt) work without renaming anything.
TOKEN_FILE_NAMES: tuple[str, ...] = ("Flickr8k.token.txt", "captions.txt")
TRAIN_SPLIT_FILE = "Flickr_8k.trainImages.txt"
VAL_SPLIT_FILE = "Flickr_8k.devImages.txt"
TEST_SPLIT_FILE = "Flickr_8k.testImages.txt"
FALLBACK_SPLIT_SIZES = (6000, 1000, 1000)   # used when split files are absent
SEED = 42

# --------------------------------------------------------------------------- #
# Image encoder (ResNet-50)
# --------------------------------------------------------------------------- #
IMAGE_SIZE = (224, 224)
FEATURE_DIM = 2048                           # ResNet-50 global-average-pool width

# --------------------------------------------------------------------------- #
# Decoder hyper-parameters
# --------------------------------------------------------------------------- #
EMBEDDING_DIM = 256
LSTM_UNITS = 256
DENSE_UNITS = 256
DROPOUT_RATE = 0.5
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
CLIP_NORM = 1.0
EPOCHS = 30
EARLY_STOP_PATIENCE = 4
LR_REDUCE_PATIENCE = 2
LR_REDUCE_FACTOR = 0.5
SHUFFLE_BUFFER = 50_000

# --------------------------------------------------------------------------- #
# Vocabulary boundaries
# --------------------------------------------------------------------------- #
PAD_TOKEN = "<pad>"        # index 0 (masked by the Embedding layer)
UNK_TOKEN = "<unk>"        # index 1
START_TOKEN = "startseq"   # index 2
END_TOKEN = "endseq"       # index 3
MIN_WORD_FREQUENCY = 5     # words rarer than this become <unk>
MAX_VOCAB_SIZE: Optional[int] = None   # optional hard cap (None = no cap)
MAX_CAPTION_LENGTH_CAP = 40            # truncate pathological captions

# --------------------------------------------------------------------------- #
# Inference defaults
# --------------------------------------------------------------------------- #
DEFAULT_BEAM_WIDTH = 3
LENGTH_PENALTY = 0.7       # beam score = log-prob / length**LENGTH_PENALTY

# --------------------------------------------------------------------------- #
# Text-to-speech defaults
# --------------------------------------------------------------------------- #
TTS_RATE = 170             # words per minute
TTS_VOLUME = 1.0           # 0.0 - 1.0

# --------------------------------------------------------------------------- #
# Live stream defaults (dashboard)
# --------------------------------------------------------------------------- #
LIVE_DISPLAY_FPS = 15
LIVE_CAPTION_INTERVAL_S = 4.0
LIVE_SCENE_CHANGE_THRESHOLD = 6.0     # mean abs grey-level diff (0-255)
LIVE_JPEG_QUALITY = 70
LIVE_MAX_FRAME_WIDTH = 960
HISTORY_LIMIT = 300


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create every output directory the project writes to."""
    for directory in (DATA_DIR, FLICKR_DIR, OUTPUT_DIR, MODEL_DIR, CACHE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def find_dataset_file(names: Iterable[str]) -> Optional[Path]:
    """Return the first file under FLICKR_DIR whose name matches any of `names`."""
    if not FLICKR_DIR.exists():
        return None
    for name in names:
        match = next(FLICKR_DIR.rglob(name), None)
        if match is not None:
            return match
    return None


def find_images_dir() -> Optional[Path]:
    """Locate the folder that holds the Flickr8k .jpg files."""
    if not FLICKR_DIR.exists():
        return None
    first_jpg = next(FLICKR_DIR.rglob("*.jpg"), None)
    return first_jpg.parent if first_jpg is not None else None
