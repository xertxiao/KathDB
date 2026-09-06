"""call_model media handling: paths, URLs, data URIs and audio tuples."""

import pytest

from kathdb.common.model_call import audio_data, image_url, modality_of


def test_modality_inference(tmp_path):
    assert modality_of("a.png") == "image"
    assert modality_of("https://x/y.jpg") == "image"
    assert modality_of("data:image/png;base64,AAAA") == "image"
    assert modality_of("clip.wav") == "audio"
    assert modality_of(("QUJD", "mp3")) == "audio"


def test_image_url_passthrough_and_local_file(tmp_path):
    assert image_url("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    assert image_url("https://x/y.jpg") == "https://x/y.jpg"
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG")
    assert image_url(str(p)).startswith("data:image/png;base64,")
    with pytest.raises(FileNotFoundError):
        image_url(str(tmp_path / "missing.jpg"))


def test_audio_data_from_path_and_tuple(tmp_path):
    p = tmp_path / "c.mp3"
    p.write_bytes(b"ID3")
    b64, fmt = audio_data(str(p))
    assert fmt == "mp3" and b64
    assert audio_data(("QUJD", "wav")) == ("QUJD", "wav")
