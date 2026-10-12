import io

from PIL import Image

from fake_blot_detection.images import StandardizeConfig, build_dataset, read_manifest, standardize
from fake_blot_detection.synth import draw_blot


def test_standardize_makes_same_format():
    a = Image.new("RGB", (1000, 300), (200, 100, 50))
    b = Image.new("L", (200, 400), 128)
    for img in (a, b):
        out = Image.open(io.BytesIO(standardize(img, StandardizeConfig(long_side=256))))
        assert out.format == "JPEG"
        assert out.mode == "L"
        assert max(out.size) == 256


def test_draw_blot_is_deterministic():
    assert draw_blot(3).tobytes() == draw_blot(3).tobytes()
    assert draw_blot(3).tobytes() != draw_blot(4).tobytes()


def test_build_dataset(tmp_path):
    real, fake = tmp_path / "real", tmp_path / "fake"
    real.mkdir()
    fake.mkdir()
    for i in range(3):
        Image.new("L", (300, 100), 100 + i).save(real / f"r{i}.png")
        draw_blot(i).save(fake / f"blot_{i}.png")
    rows = build_dataset(
        {"real_set": (real, "real"), "synthetic": (fake, "ai")}, tmp_path / "out", max_per_source=2
    )
    assert len(rows) == 4
    manifest = read_manifest(tmp_path / "out" / "manifest.csv")
    assert sorted(r["label"] for r in manifest) == ["ai", "ai", "real", "real"]
    for r in manifest:
        assert (tmp_path / "out" / r["path"]).exists()
        assert "blot" not in r["path"] and "r0" not in r["path"]  # names don't leak origin


def test_match_histogram_copies_reference_levels():
    import numpy as np

    from fake_blot_detection.synth import match_histogram

    img = draw_blot(1, 120, 40)
    ref = Image.fromarray(np.linspace(0, 255, 120 * 40).reshape(40, 120).astype(np.uint8))
    out = np.asarray(match_histogram(img, ref))
    assert abs(out.mean() - np.asarray(ref).mean()) < 1
    assert abs(out.std() - np.asarray(ref).std()) < 1
    # Pixel order is kept: the darkest pixel stays the darkest.
    assert np.asarray(img).argmin() == out.argmin()
