"""Fast unit tests for text pre-processing and the vocabulary (no TensorFlow needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from utils import data_pipeline as dp


def test_clean_caption_strips_punctuation_and_single_letters():
    assert dp.clean_caption("A dog's running, 2 times!") == ["a", "dog", "running", "times"]


def test_parse_both_file_layouts(tmp_path):
    token = tmp_path / "Flickr8k.token.txt"
    token.write_text("1.jpg#0\tA man rides a horse .\n1.jpg#1\tA person on a horse .\n")
    assert len(dp.parse_caption_file(token)["1.jpg"]) == 2

    kaggle = tmp_path / "captions.txt"
    kaggle.write_text("image,caption\n2.jpg,\"Two kids, playing outside\"\n")
    assert dp.parse_caption_file(kaggle)["2.jpg"] == [["two", "kids", "playing", "outside"]]


def test_vocabulary_roundtrip(tmp_path):
    caps = [["a", "dog", "runs"]] * 6 + [["a", "rare", "word"]]
    vocab = dp.Vocabulary.build(caps, min_freq=5)
    assert vocab.itos[:4] == [config.PAD_TOKEN, config.UNK_TOKEN, config.START_TOKEN, config.END_TOKEN]
    assert "rare" not in vocab.stoi and vocab.max_length == 5
    ids = vocab.encode(["a", "dog", "rare"])
    assert ids[0] == vocab.start_id and ids[-1] == vocab.end_id and vocab.unk_id in ids
    assert vocab.decode(ids) == "a dog"                    # <unk> is never spoken
    vocab.save(tmp_path / "v.json")
    assert dp.Vocabulary.load(tmp_path / "v.json").itos == vocab.itos


def test_fallback_split_is_deterministic():
    names = [f"{i}.jpg" for i in range(100)]
    a, b = dp.make_splits(names), dp.make_splits(names)
    assert a == b and sum(len(s) for s in a) == 100
