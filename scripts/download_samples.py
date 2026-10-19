"""Download a small, reproducible sample of Coralscapes v2."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "josauder/coralscapesV2"


def save_json(path, content):
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--split", default="train")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count debe ser mayor que cero")

    # Resolve the requested revision to an immutable dataset commit.
    revision = HfApi().dataset_info(
        DATASET_ID, revision=args.revision
    ).sha

    # Separate this sample set from the earlier single-image download.
    output = ROOT / "data/samples" / (
        f"coralscapes_{args.split}_{revision[:12]}_{args.count:03d}"
    )

    if output.exists():
        raise FileExistsError(
            f"El directorio ya existe: {output}\n"
            "No se sobrescribirá. Revisa su manifest.json antes de repetir."
        )

    output.mkdir(parents=True)
    print(f"Dataset: {DATASET_ID}", flush=True)
    print(f"Revision: {revision}", flush=True)
    print(f"Salida: {output}", flush=True)

    dataset = load_dataset(
        DATASET_ID,
        split=args.split,
        revision=revision,
        streaming=True,
    )
    iterator = iter(dataset)
    records = []

    try:
        for index in range(args.count):
            try:
                sample = next(iterator)
            except StopIteration:
                raise RuntimeError(
                    f"Solo se encontraron {index} muestras; "
                    f"se solicitaron {args.count}."
                ) from None

            image = sample["image"].convert("RGB")
            label = np.asarray(sample["label"])
            width, height = image.size

            if label.shape != (height, width):
                raise ValueError(
                    f"Muestra {index}: etiqueta {label.shape} "
                    f"incompatible con imagen {image.size}"
                )

            sample_id = f"coralscapes_{args.split}_{index:04d}"
            image_name = f"{sample_id}.png"
            label_name = f"{sample_id}_label.npy"

            image.save(output / image_name)
            np.save(output / label_name, label, allow_pickle=False)

            record = {
                "sample_id": sample_id,
                "sample_index": index,
                "image": image_name,
                "label": label_name,
                "image_size_wh": [width, height],
                "label_dtype": str(label.dtype),
                "label_values": [
                    int(value) for value in np.unique(label)
                ],
            }
            records.append(record)

            print(
                f"[{index + 1}/{args.count}] {image_name} "
                f"({width} x {height})",
                flush=True,
            )

        manifest = {
            "dataset": DATASET_ID,
            "revision": revision,
            "split": args.split,
            "selection": "first_n_in_dataset_order",
            "count": len(records),
            "samples": records,
        }
        save_json(output / "manifest.json", manifest)

    finally:
        # Attempt to release the streaming iterator before Python shuts down.
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
        del iterator
        del dataset
        gc.collect()

    print(
        f"\nDescarga completada. Manifiesto: {output / 'manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
