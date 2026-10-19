"""Run SAM 3 on the images listed in a sample manifest."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sam3.model.vitdet as vitdet
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROOT = Path(__file__).resolve().parents[1]


def linear_activation_fp32(activation, linear, x):
    return activation()(linear(x))


# Same compatibility adjustment used in first_inference.py.
vitdet.addmm_act = linear_activation_fp32


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def save_predictions(image, result, output):
    masks = result["masks"].detach().cpu().numpy()
    scores = result["scores"].detach().float().cpu().numpy()
    boxes = result["boxes"].detach().float().cpu().numpy()

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    width, height = image.size
    if masks.shape != (len(scores), height, width):
        raise ValueError(f"Dimensiones de máscaras inesperadas: {masks.shape}")

    masks = masks > 0.5
    np.savez_compressed(
        output / "predictions.npz",
        masks=masks,
        scores=scores,
        boxes=boxes,
    )

    overlay = np.asarray(image).astype(np.float32).copy()
    rng = np.random.default_rng(42)

    for mask in masks:
        color = rng.integers(40, 256, size=3)
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * color

    Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8)
    ).save(output / "overlay.png")

    return {
        "num_detections": len(masks),
        "scores": scores.reshape(-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompt", default="fish")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples_root = manifest_path.parent

    if args.output.exists():
        raise FileExistsError(
            f"La salida ya existe: {args.output}. "
            "Elige otra carpeta para conservar los resultados anteriores."
        )

    # Check all input images before loading the model.
    for sample in manifest["samples"]:
        path = samples_root / sample["image"]
        if not path.is_file():
            raise FileNotFoundError(path)

    if not torch.cuda.is_available():
        raise RuntimeError("Esta prueba requiere CUDA.")

    args.output.mkdir(parents=True)
    save_json(args.output / "manifest.json", manifest)

    print("Cargando SAM 3 una sola vez...", flush=True)
    model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model)

    configuration = {
        "prompt": args.prompt,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "parameter_dtype": str(next(model.parameters()).dtype),
        "vitdet_mlp_implementation": "unfused_fp32",
    }
    save_json(args.output / "configuration.json", configuration)

    summaries = []

    with torch.inference_mode():
        for index, sample in enumerate(manifest["samples"], start=1):
            sample_id = sample["sample_id"]
            output = args.output / sample_id
            output.mkdir()

            print(
                f"[{index}/{len(manifest['samples'])}] {sample_id}",
                flush=True,
            )

            with Image.open(samples_root / sample["image"]) as source:
                image = source.convert("RGB")

            state = processor.set_image(image)
            result = processor.set_text_prompt(
                state=state,
                prompt=args.prompt,
            )

            summary = {
                "sample_id": sample_id,
                "prompt": args.prompt,
                **save_predictions(image, result, output),
            }
            save_json(output / "summary.json", summary)
            summaries.append(summary)

            print(
                f"  Detecciones: {summary['num_detections']}",
                flush=True,
            )

            # Avoid retaining the previous image's GPU tensors.
            del result, state

    save_json(args.output / "summary.json", summaries)
    print(f"\nResultados guardados en: {args.output.resolve()}")


if __name__ == "__main__":
    main()