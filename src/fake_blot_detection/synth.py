"""Procedurally drawn western blots: the "Claude writes code that draws a blot" generator.

Each blot has a few lanes; each lane has bands at shared molecular-weight positions with
random presence and intensity. Bands are elliptical Gaussian blobs with slight "smiling"
(curvature), horizontal smearing and vertical tails, on a background with an uneven
gradient, film grain and occasional specks. Everything is driven by one seed, so a given
seed always produces the same image.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BlotParams:
    width: int = 640
    height: int = 220
    min_lanes: int = 4
    max_lanes: int = 12
    min_band_rows: int = 1
    max_band_rows: int = 4


DEFAULT_PARAMS = BlotParams()


def draw_blot(seed: int, params: BlotParams = DEFAULT_PARAMS) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = params.height, params.width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Signal image: 0 = no signal, larger = darker band on the final image.
    signal = np.zeros((h, w), dtype=np.float32)

    n_lanes = int(rng.integers(params.min_lanes, params.max_lanes + 1))
    margin = w * rng.uniform(0.04, 0.1)
    pitch = (w - 2 * margin) / n_lanes
    lane_width = pitch * rng.uniform(0.55, 0.8)
    lane_centers = (
        margin + pitch * (np.arange(n_lanes) + 0.5) + rng.normal(0, pitch * 0.02, n_lanes)
    )

    n_rows = int(rng.integers(params.min_band_rows, params.max_band_rows + 1))
    row_ys = np.sort(rng.uniform(h * 0.15, h * 0.85, n_rows))
    smile = rng.uniform(-0.15, 0.35) * h / 100  # curvature of bands across the gel

    for row_y in row_ys:
        base_intensity = rng.uniform(0.4, 1.0)
        thickness = rng.uniform(3, 9)
        for cx in lane_centers:
            if rng.random() < 0.12:  # missing band in this lane
                continue
            intensity = base_intensity * rng.lognormal(0, 0.35)
            y = row_y + smile * ((cx - w / 2) / (w / 2)) ** 2 * 10 + rng.normal(0, 0.8)
            half_w = lane_width / 2 * rng.uniform(0.85, 1.05)
            sigma_y = thickness * rng.uniform(0.8, 1.2) / 2
            # Flat-topped horizontal profile with soft edges.
            dx = np.abs(xx - cx) / half_w
            horiz = 1 / (1 + dx**8)
            # Band edges curve slightly ("smile") within the lane.
            local_y = y + 0.6 * sigma_y * ((xx - cx) / half_w) ** 2
            vert = np.exp(-0.5 * ((yy - local_y) / sigma_y) ** 2)
            # Downward tail (trailing smear).
            tail = np.where(yy > local_y, np.exp(-(yy - local_y) / (sigma_y * 4)), 0) * 0.25
            signal += intensity * horiz * (vert + tail)

    # Lane-wide faint smear (protein trailing down the lane).
    for cx in lane_centers:
        dx = np.abs(xx - cx) / (lane_width / 2)
        signal += rng.uniform(0.0, 0.06) * (1 / (1 + dx**6)) * (yy / h)

    # Film-like saturation: dense bands flatten out.
    signal = 1 - np.exp(-1.6 * signal)

    # Background: uneven gradient plus low-frequency blotches.
    gx, gy = rng.normal(0, 0.04, 2)
    background = 0.06 + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    low = rng.normal(0, 1, (h // 20 + 2, w // 20 + 2)).astype(np.float32)
    low_img = Image.fromarray(low).resize((w, h), Image.BICUBIC)
    background += 0.025 * np.asarray(low_img, dtype=np.float32)

    darkness = np.clip(background + signal, 0, 1)
    darkness += rng.normal(0, rng.uniform(0.008, 0.025), (h, w))  # film grain

    # Occasional dust specks.
    for _ in range(int(rng.poisson(3))):
        sy, sx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.6, 2.0)
        darkness += rng.uniform(0.2, 0.6) * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * r**2))

    gray = 255 * (1 - np.clip(darkness, 0, 1))
    return Image.fromarray(gray.astype(np.uint8), mode="L")
