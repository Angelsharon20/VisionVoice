"""Model definitions: ResNet-50 encoder, LSTM merge decoder and inference engine.

Architecture (the classic "merge" captioner):

    image  -> ResNet-50 (frozen, GAP) -> 2048-d -> Dropout -> Dense(256) ----\
                                                                              add -> Dense(256) -> softmax(vocab)
    tokens -> Embedding(mask_zero) -> Dropout -> LSTM(256) ------------------/

The CNN is frozen and its features are cached once, so training only updates
the light decoder. At inference time both stages run as compiled tf.functions
with fixed input signatures, so a live stream never triggers retracing.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
def preprocess_image_tensor(image: tf.Tensor) -> tf.Tensor:
    """uint8/float [H, W, 3] RGB -> ResNet-50 preprocessed float32 [224, 224, 3].

    The exact same op chain is used for feature caching (training) and for live
    inference, which keeps train/test features consistent.
    """
    image = tf.image.resize(tf.cast(image, tf.float32), config.IMAGE_SIZE)
    return keras.applications.resnet50.preprocess_input(image)


def build_feature_extractor() -> keras.Model:
    """ImageNet ResNet-50 truncated before the classifier: image -> 2048-d vector."""
    base = keras.applications.ResNet50(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*config.IMAGE_SIZE, 3),
    )
    base.trainable = False
    return keras.Model(base.input, base.output, name="resnet50_encoder")


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #
def build_caption_model(vocab_size: int, max_length: int) -> keras.Model:
    """Merge-architecture decoder: (image features, partial caption) -> next-word distribution."""
    image_in = keras.Input(shape=(config.FEATURE_DIM,), name="image_features")
    image_branch = layers.Dropout(config.DROPOUT_RATE, name="image_dropout")(image_in)
    image_branch = layers.Dense(config.DENSE_UNITS, activation="relu", name="image_dense")(image_branch)

    text_in = keras.Input(shape=(max_length,), dtype="int32", name="caption_input")
    text_branch = layers.Embedding(
        vocab_size, config.EMBEDDING_DIM, mask_zero=True, name="word_embedding"
    )(text_in)
    text_branch = layers.Dropout(config.DROPOUT_RATE, name="text_dropout")(text_branch)
    text_branch = layers.LSTM(config.LSTM_UNITS, name="caption_lstm")(text_branch)

    merged = layers.add([image_branch, text_branch], name="merge")
    hidden = layers.Dense(config.DENSE_UNITS, activation="relu", name="decoder_dense")(merged)
    output = layers.Dense(vocab_size, activation="softmax", name="next_word")(hidden)

    model = keras.Model([image_in, text_in], output, name="caption_decoder")
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE, clipnorm=config.CLIP_NORM
        ),
        loss="sparse_categorical_crossentropy",
    )
    return model


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
@dataclass
class CaptionResult:
    caption: str
    feature_ms: float
    decode_ms: float
    beam_width: int

    @property
    def latency_ms(self) -> float:
        return self.feature_ms + self.decode_ms


class CaptionGenerator:
    """Loads trained artefacts and turns RGB frames into sentences, quickly."""

    def __init__(self, load_cnn: bool = True, model_path: Optional[Path] = None,
                 vocab_path: Optional[Path] = None) -> None:
        from utils.data_pipeline import Vocabulary  # local import avoids a cycle

        self.vocab = Vocabulary.load(vocab_path or config.VOCAB_PATH)
        self.max_length = self.vocab.max_length
        self.decoder = keras.models.load_model(model_path or config.MODEL_PATH, compile=False)
        self.cnn = build_feature_extractor() if load_cnn else None

        self._decode_fn = tf.function(
            lambda feats, seqs: self.decoder([feats, seqs], training=False),
            input_signature=[
                tf.TensorSpec([None, config.FEATURE_DIM], tf.float32),
                tf.TensorSpec([None, self.max_length], tf.int32),
            ],
        )
        self._extract_fn = tf.function(
            lambda image: self.cnn(preprocess_image_tensor(image)[None], training=False),
            input_signature=[tf.TensorSpec([None, None, 3], tf.uint8)],
        ) if load_cnn else None

        # Token ids that must never be emitted as a word.
        self._blocked = [self.vocab.pad_id, self.vocab.start_id, self.vocab.unk_id]

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def artifacts_ready() -> bool:
        return config.MODEL_PATH.exists() and config.VOCAB_PATH.exists()

    def warmup(self) -> None:
        """Trace both graphs once so the first live frame is not slow."""
        if self._extract_fn is not None:
            dummy = np.zeros((224, 224, 3), np.uint8)
            self.generate(dummy, beam_width=1)

    def extract_features(self, rgb: np.ndarray) -> np.ndarray:
        if self._extract_fn is None:
            raise RuntimeError("CaptionGenerator was created with load_cnn=False")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        return self._extract_fn(tf.constant(rgb)).numpy()

    # -- public API --------------------------------------------------------- #
    def generate(self, rgb: np.ndarray, beam_width: int = config.DEFAULT_BEAM_WIDTH) -> CaptionResult:
        t0 = time.perf_counter()
        features = self.extract_features(rgb)
        feature_ms = (time.perf_counter() - t0) * 1000.0
        result = self.generate_from_features(features, beam_width)
        result.feature_ms = feature_ms
        return result

    def generate_from_features(self, features: np.ndarray, beam_width: int = 1) -> CaptionResult:
        features = np.asarray(features, np.float32).reshape(1, config.FEATURE_DIM)
        t0 = time.perf_counter()
        tokens = self._greedy(features) if beam_width <= 1 else self._beam(features, beam_width)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return CaptionResult(self.vocab.decode(tokens), 0.0, decode_ms, max(1, beam_width))

    # -- decoding strategies ------------------------------------------------- #
    def _next_log_probs(self, feats: np.ndarray, seqs: np.ndarray) -> np.ndarray:
        probs = self._decode_fn(tf.constant(feats), tf.constant(seqs)).numpy()
        logp = np.log(probs + 1e-9)
        logp[:, self._blocked] = -np.inf
        return logp

    def _greedy(self, features: np.ndarray) -> list[int]:
        seq = np.zeros((1, self.max_length), np.int32)
        seq[0, 0] = self.vocab.start_id
        out: list[int] = []
        for t in range(1, self.max_length):
            logp = self._next_log_probs(features, seq)[0]
            if out:                       # forbid immediate word repetition ("a a")
                logp[out[-1]] = -np.inf
            word = int(np.argmax(logp))
            if word == self.vocab.end_id:
                break
            out.append(word)
            seq[0, t] = word
        return out

    def _beam(self, features: np.ndarray, width: int) -> list[int]:
        start, end = self.vocab.start_id, self.vocab.end_id
        beams: list[tuple[list[int], float]] = [([start], 0.0)]
        finished: list[tuple[list[int], float]] = []

        for _ in range(1, self.max_length):
            seqs = np.zeros((len(beams), self.max_length), np.int32)
            for i, (toks, _) in enumerate(beams):
                seqs[i, : len(toks)] = toks
            feats = np.repeat(features, len(beams), axis=0)
            logp = self._next_log_probs(feats, seqs)

            candidates: list[tuple[list[int], float]] = []
            for i, (toks, score) in enumerate(beams):
                top = np.argpartition(logp[i], -width)[-width:]
                for w in top:
                    if np.isfinite(logp[i, w]):
                        candidates.append((toks + [int(w)], score + float(logp[i, w])))
            candidates.sort(key=lambda c: c[1], reverse=True)

            beams = []
            for toks, score in candidates:
                if toks[-1] == end:
                    finished.append((toks, score))
                else:
                    beams.append((toks, score))
                if len(beams) == width:
                    break
            if not beams or len(finished) >= width:
                break

        pool = finished or beams
        best = max(pool, key=lambda c: c[1] / (len(c[0]) ** config.LENGTH_PENALTY))
        return [t for t in best[0][1:] if t != end]


def count_parameters(model: keras.Model) -> int:
    return int(model.count_params())


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
