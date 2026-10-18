from pathlib import Path
import json

import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import sam3.model.vitdet as vitdet


def linear_activation_fp32(activation, linear, x):
    """Ruta convencional para evitar la conversión forzada a BF16."""
    return activation()(linear(x))


vitdet.addmm_act = linear_activation_fp32

def report(stage):
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(
        f"{stage} | VRAM libre: {free / 1024**3:.2f} GiB",
        flush=True,
    )


def main():
    root = Path(__file__).resolve().parents[1]
    image_path = root / "data/samples/coralscapes_train_0000.png"
    output_dir = root / "results/first_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = source.convert("RGB")

    report("1. Cargando modelo")
    model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model)

    with torch.inference_mode():
        report("2. Procesando imagen")
        state = processor.set_image(image)

        # Libera bloques de caché que ya no están en uso.
        torch.cuda.empty_cache()

        report("3. Aplicando prompt: fish")
        result = processor.set_text_prompt(
            state=state,
            prompt="fish",
        )

        report("4. Copiando resultados a CPU")
        masks = result["masks"].detach().cpu().numpy()
        scores = result["scores"].detach().float().cpu().numpy()
        boxes = result["boxes"].detach().float().cpu().numpy()

    # Normaliza las máscaras a [N, H, W].
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim != 3:
        raise ValueError(f"Forma de máscaras inesperada: {masks.shape}")

    masks = masks > 0.5
    count = len(masks)

    print(f"Objetos detectados: {count}", flush=True)
    print(f"Puntuaciones: {scores.tolist()}", flush=True)

    np.savez_compressed(
        output_dir / "predictions.npz",
        masks=masks,
        scores=scores,
        boxes=boxes,
    )

    # Colores reproducibles para distinguir las instancias.
    overlay = np.asarray(image).astype(np.float32).copy()
    rng = np.random.default_rng(42)

    for mask in masks:
        if mask.shape != overlay.shape[:2]:
            raise ValueError("La máscara no coincide con la imagen original")
        color = rng.integers(40, 256, size=3)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color

    overlay_path = output_dir / "overlay2.png"
    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(overlay_path)

    summary = {
        "image": str(image_path),
        "prompt": "fish",
        "num_detections": count,
        "scores": scores.tolist(),
        "gpu": torch.cuda.get_device_name(0),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "vitdet_mlp_implementation": "unfused_fp32",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Superposición guardada: {overlay_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except torch.OutOfMemoryError:
        print(
            "\nMemoria insuficiente en esta prueba FP32. "
            "Guarda la última etapa mostrada y el error completo.",
            flush=True,
        )
        raise
