"""Procedurally drawn western blots: the "Claude writes code that draws a blot" generator.

Each blot has a few lanes; each lane has bands at shared molecular-weight positions with
random presence and intensity. Bands are Gaussian blobs with slight "smiling" (curvature),
horizontal smearing and vertical tails, on a background with an uneven gradient, film
grain and occasional specks. Like cropped figure panels, some blots get a dark frame.

Everything is driven by one seed, so a given seed always produces the same image. Pass
`width` and `height` to match the size of real images; band sizes and lane counts are
scaled to the canvas.
"""

import numpy as np
from PIL import Image, ImageFilter


def draw_blot(seed: int, width: int = 300, height: int = 70) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = height, width
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    # Signal image: 0 = no signal, larger = darker band on the final image.
    signal = np.zeros((h, w), dtype=np.float32)

    n_rows = int(rng.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05]))
    n_rows = max(n_rows, min(6, round(h / w * 2)))  # tall canvases hold more rows
    n_lanes = int(np.clip(round(w / (h / n_rows) * rng.uniform(0.25, 0.6)), 2, 20))
    margin = w * rng.uniform(0.02, 0.08)
    pitch = (w - 2 * margin) / n_lanes
    lane_width = pitch * rng.uniform(0.7, 0.92)
    lane_centers = (
        margin + pitch * (np.arange(n_lanes) + 0.5) + rng.normal(0, pitch * 0.02, n_lanes)
    )

    row_height = h / n_rows
    row_ys = row_height * (np.arange(n_rows) + 0.5) + rng.normal(0, row_height * 0.08, n_rows)
    smile = rng.uniform(-0.03, 0.08) * row_height  # curvature of bands across the blot

    for row_y in row_ys:
        base_intensity = rng.uniform(0.5, 1.6)
        sigma = row_height * rng.uniform(0.05, 0.13)
        curve = rng.uniform(-0.25, 0.25)
        tail_strength = rng.uniform(0.0, 0.12)
        for cx in lane_centers:
            if rng.random() < 0.1:  # missing band in this lane
                continue
            intensity = base_intensity * rng.lognormal(0, 0.35)
            y = row_y + smile * ((cx - w / 2) / (w / 2)) ** 2 + rng.normal(0, sigma * 0.1)
            half_w = lane_width / 2 * rng.uniform(0.85, 1.05)
            sigma_y = sigma * rng.uniform(0.85, 1.15)
            # Flat-topped horizontal profile with soft edges.
            dx = np.abs(xx - cx) / half_w
            horiz = 1 / (1 + dx**6)
            # Bands are a little thinner at their ends and curve slightly within the lane.
            local_sigma = sigma_y * (1 - 0.25 * np.minimum(dx, 1) ** 2)
            local_y = y + curve * sigma_y * ((xx - cx) / half_w) ** 2
            offset = yy - local_y
            vert = np.exp(-0.5 * (offset / local_sigma) ** 2)
            # Faint trailing smear below the band.
            tail = np.exp(-np.maximum(offset, 0) / (sigma_y * 2.5)) * (offset > 0) * tail_strength
            signal += intensity * horiz * (vert + tail)

    # Film-like saturation: dense bands flatten out.
    signal = 1 - np.exp(-1.6 * signal)

    # Background: overall level, uneven gradient, and low-frequency blotches.
    level = rng.choice([rng.uniform(0.0, 0.08), rng.uniform(0.08, 0.35)], p=[0.6, 0.4])
    gx, gy = rng.normal(0, 0.03, 2)
    background = level + gx * (xx / w - 0.5) + gy * (yy / h - 0.5)
    low = rng.normal(0, 1, (max(2, h // 15), max(2, w // 15))).astype(np.float32)
    low_img = Image.fromarray(low).resize((w, h), Image.BICUBIC)
    background += rng.uniform(0.002, 0.01) * np.asarray(low_img, dtype=np.float32)

    darkness = np.clip(background + signal * (1 - background), 0, 1)
    darkness += rng.normal(0, rng.uniform(0.002, 0.01), (h, w))  # film grain

    # Levels adjustment, as when a figure is prepared: often clips the background to white
    # and the densest bands to black.
    lo = rng.uniform(0.0, 0.12)
    hi = rng.uniform(0.6, 1.0)
    darkness = (darkness - lo) / (hi - lo)

    # Occasional dust specks.
    for _ in range(int(rng.poisson(1))):
        sy, sx = rng.uniform(0, h), rng.uniform(0, w)
        r = rng.uniform(0.5, 1.5)
        darkness += rng.uniform(0.2, 0.6) * np.exp(-((yy - sy) ** 2 + (xx - sx) ** 2) / (2 * r**2))

    gray = (255 * (1 - np.clip(darkness, 0, 1))).astype(np.uint8)
    img = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))

    if rng.random() < 0.3:  # dark frame, as on panels cropped from figures
        frame = np.asarray(img).copy()
        t = int(rng.integers(1, 3))
        frame[:t], frame[-t:], frame[:, :t], frame[:, -t:] = 0, 0, 0, 0
        img = Image.fromarray(frame)
    return img


def match_histogram(img: Image.Image, reference: Image.Image) -> Image.Image:
    """Remap the gray levels of `img` so its histogram matches `reference`'s.

    The darkest pixel of `img` gets the darkest value of `reference`, the median pixel
    the median value, and so on. Pixel order (the blot's structure) is unchanged.
    """
    src = np.asarray(img.convert("L"), dtype=np.float64)
    ref = np.sort(np.asarray(reference.convert("L"), dtype=np.float64).ravel())
    ranks = src.ravel().argsort(kind="stable").argsort(kind="stable")
    quantiles = ranks / max(1, src.size - 1)
    matched = np.interp(quantiles, np.linspace(0, 1, ref.size), ref)
    return Image.fromarray(matched.reshape(src.shape).round().astype(np.uint8))
