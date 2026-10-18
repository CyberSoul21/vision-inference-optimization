from pathlib import Path
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data/samples/coralscapes_train_0000.png"
RESULTS = ROOT / "results/first_inference"
OUTPUT = RESULTS / "visualization"


def load_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")

    with np.load(RESULTS / "predictions.npz", allow_pickle=False) as data:
        masks = data["masks"].astype(bool)
        scores = data["scores"].reshape(-1)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    width, height = image.size
    if masks.shape != (len(scores), height, width):
        raise ValueError(f"Dimensiones inesperadas: {masks.shape}")

    original = np.asarray(image)
    overlay = original.astype(np.float32).copy()
    rng = np.random.default_rng(42)
    colors = rng.integers(60, 256, size=(len(scores), 3))
    detections = []

    for index, (mask, score, color) in enumerate(
        zip(masks, scores, colors), start=1
    ):
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            print(f"Detección {index}: máscara vacía; omitida")
            continue

        # Caja calculada directamente desde la máscara.
        box = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color
        detections.append((index, float(score), color, box))

    annotated = Image.fromarray(overlay.astype(np.uint8))
    draw = ImageDraw.Draw(annotated)
    font = load_font(24)

    for index, score, color, box in detections:
        rgb = tuple(int(c) for c in color)
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=rgb, width=3)

        text = f"#{index} fish | score={score:.3f}"
        bounds = draw.textbbox((0, 0), text, font=font)
        text_width = bounds[2] - bounds[0]
        text_height = bounds[3] - bounds[1]
        tx = max(0, min(x0, width - text_width - 12))
        ty = max(0, y0 - text_height - 16)

        draw.rectangle(
            (tx, ty, tx + text_width + 12, ty + text_height + 12),
            fill="black",
        )
        draw.text(
            (tx + 6, ty + 6 - bounds[1]),
            text,
            font=font,
            fill=rgb,
        )

        # Recorte con contexto y máscara aislada con transparencia.
        padding = 30
        crop_box = (
            max(0, x0 - padding),
            max(0, y0 - padding),
            min(width, x1 + padding),
            min(height, y1 + padding),
        )
        prefix = f"detection_{index:02d}"

        image.crop(crop_box).save(OUTPUT / f"{prefix}_original.png")

        isolated_overlay = original.astype(np.float32).copy()
        mask = masks[index - 1]
        isolated_overlay[mask] = (
            0.55 * isolated_overlay[mask] + 0.45 * color
        )
        Image.fromarray(isolated_overlay.astype(np.uint8)).crop(
            crop_box
        ).save(OUTPUT / f"{prefix}_overlay.png")

        cutout = image.convert("RGBA")
        cutout.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
        cutout.crop(crop_box).save(OUTPUT / f"{prefix}_cutout.png")

        print(f"#{index}: score={score:.3f}, caja={box}")

    annotated.save(OUTPUT / "detections_numbered.png")
    print(f"\nResultados guardados en: {OUTPUT}")


if __name__ == "__main__":
    main()
