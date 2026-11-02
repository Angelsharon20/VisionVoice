"""Train the caption decoder on Flickr8k, evaluate BLEU + latency, write dashboard metrics.

    python train.py                       # full training (cached ResNet-50 features, then LSTM decoder)
    python train.py --epochs 10 --beam 3  # shorter run
    python train.py --limit 300           # tiny smoke run on 300 images

Artefacts (all under ./outputs):
    models/caption_model.keras   best decoder weights (lowest validation loss)
    models/vocab.json            vocabulary + max sequence length
    training_history.json        loss / val_loss / lr per epoch   -> dashboard "Training" chart
    metrics.json                 BLEU-1..4 + latency statistics   -> dashboard "Accuracy" chart
"""
from __future__ import annotations

import argparse
import random
import time
from datetime import datetime, timezone

import numpy as np

import config
from utils import data_pipeline as dp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--beam", type=int, default=config.DEFAULT_BEAM_WIDTH, help="beam width for evaluation")
    p.add_argument("--limit", type=int, default=0, help="use only N images (smoke tests)")
    p.add_argument("--eval-images", type=int, default=0, help="BLEU on N test images (0 = whole test split)")
    p.add_argument("--bench-images", type=int, default=20, help="images used for the end-to-end latency benchmark")
    p.add_argument("--refresh-features", action="store_true", help="recompute the cached ResNet-50 features")
    return p.parse_args()


def evaluate_bleu(generator, features, captions, names, beam: int) -> dict:
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    references, hypotheses = [], []
    start = time.perf_counter()
    for i, name in enumerate(names, 1):
        hypotheses.append(generator.generate_from_features(features[name], beam).caption.split())
        references.append(captions[name])
        if i % 100 == 0:
            print(f"  decoded {i}/{len(names)}")
    elapsed = time.perf_counter() - start
    smooth = SmoothingFunction().method1
    weights = {"bleu1": (1, 0, 0, 0), "bleu2": (.5, .5, 0, 0), "bleu3": (1 / 3,) * 3 + (0,), "bleu4": (.25,) * 4}
    scores = {k: float(corpus_bleu(references, hypotheses, weights=w, smoothing_function=smooth))
              for k, w in weights.items()}
    scores["decode_ms_per_image"] = 1000 * elapsed / max(1, len(names))
    return scores


def benchmark_latency(generator, images_dir, names) -> dict:
    """End-to-end (ResNet-50 + decoder) latency on real images, after warm-up."""
    import tensorflow as tf

    generator.warmup()
    feat, dec = [], []
    for name in names:
        rgb = tf.io.decode_jpeg(tf.io.read_file(str(images_dir / name)), channels=3).numpy()
        result = generator.generate(rgb, beam_width=config.DEFAULT_BEAM_WIDTH)
        feat.append(result.feature_ms)
        dec.append(result.decode_ms)
    total = np.array(feat) + np.array(dec)
    return {"images": len(names), "feature_ms_mean": float(np.mean(feat)), "decode_ms_mean": float(np.mean(dec)),
            "total_ms_mean": float(total.mean()), "total_ms_p95": float(np.percentile(total, 95))}


def main() -> None:
    args = parse_args()
    import tensorflow as tf
    from tensorflow import keras

    from model import CaptionGenerator, build_caption_model, count_parameters, dump_json

    config.ensure_dirs()
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.keras.utils.set_random_seed(config.SEED)
    print(f"[env] TensorFlow {tf.__version__} | GPUs: {len(tf.config.list_physical_devices('GPU'))}")

    # ---- data ------------------------------------------------------------------ #
    captions, images_dir = dp.load_dataset_captions()
    train_names, val_names, test_names = dp.make_splits(captions)
    if args.limit:
        k = args.limit
        train_names, val_names, test_names = train_names[:k], val_names[:max(2, k // 6)], test_names[:max(2, k // 6)]
    print(f"[data] images: train {len(train_names)} | val {len(val_names)} | test {len(test_names)}")

    vocab = dp.Vocabulary.build(t for n in train_names for t in captions[n])
    vocab.save()
    print(f"[vocab] {vocab.size} words | max sequence length {vocab.max_length}")

    all_names = train_names + val_names + test_names
    features = dp.extract_features(images_dir, all_names, cache_path=config.FEATURES_CACHE_PATH,
                                   force=args.refresh_features)

    train_ds, train_steps = dp.make_dataset(train_names, captions, vocab, features, args.batch_size, shuffle=True)
    val_ds, val_steps = dp.make_dataset(val_names, captions, vocab, features, args.batch_size, shuffle=False)
    print(f"[data] {train_steps} train steps / {val_steps} val steps per epoch (batch {args.batch_size})")

    # ---- model --------------------------------------------------------------------- #
    config.LEARNING_RATE = args.lr
    model = build_caption_model(vocab.size, vocab.max_length)
    model.summary(line_length=100)
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(config.MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=config.LR_REDUCE_FACTOR,
                                          patience=config.LR_REDUCE_PATIENCE, min_lr=1e-5, verbose=1),
    ]
    started = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks, verbose=2)
    train_minutes = (time.time() - started) / 60
    hist = {k: [float(x) for x in v] for k, v in history.history.items()}
    dump_json(config.HISTORY_PATH, hist)

    # ---- evaluation ----------------------------------------------------------------- #
    print("[eval] loading best checkpoint for BLEU ...")
    generator = CaptionGenerator(load_cnn=False)
    eval_names = test_names[: args.eval_images] if args.eval_images else test_names
    bleu = evaluate_bleu(generator, features, captions, eval_names, args.beam)
    print("[eval] " + " | ".join(f"{k}={v:.3f}" for k, v in bleu.items() if k.startswith("bleu")))

    print("[bench] measuring end-to-end latency ...")
    latency = benchmark_latency(CaptionGenerator(load_cnn=True), images_dir, test_names[: args.bench_images])

    dump_json(config.METRICS_PATH, {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_minutes": round(train_minutes, 2),
        "epochs_run": len(hist.get("loss", [])),
        "best_val_loss": float(min(hist.get("val_loss", [float("nan")]))),
        "parameters": count_parameters(model),
        "vocab_size": vocab.size,
        "max_length": vocab.max_length,
        "beam_width": args.beam,
        "test_images": len(eval_names),
        "bleu": {k: v for k, v in bleu.items() if k.startswith("bleu")},
        "decode_ms_per_image": bleu["decode_ms_per_image"],
        "latency": latency,
    })
    print(f"[done] model -> {config.MODEL_PATH}\n       launch the dashboard with:  streamlit run app.py")


if __name__ == "__main__":
    main()
