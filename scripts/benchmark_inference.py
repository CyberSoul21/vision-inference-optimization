"""Benchmark end-to-end de SAM 3, con batch size 1."""

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import sam3
import sam3.model.vitdet as vitdet

from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


# Conservamos la adaptación usada en las pruebas anteriores.
# Evita que addmm_act convierta explícitamente las activaciones a BF16.
def linear_activation_fp32(activation, linear, x):
    return activation()(linear(x))


vitdet.addmm_act = linear_activation_fp32


def save_json(path, content):
    """Guardar informes se hace fuera del intervalo medido."""
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_info(directory):
    """Registrar el commit y si hay cambios locales, cuando existe Git."""
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"commit": commit, "dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_images(processor, image_paths, prompt):
    """Una pasada secuencial: cada llamada procesa una imagen."""
    detection_counts = []

    for image_path in image_paths:
        # La lectura, decodificación y conversión a RGB están incluidas.
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        # Incluye preparación de entrada y codificador visual.
        # Creamos un estado nuevo: no reutilizamos características
        # calculadas para la misma imagen en una repetición anterior.
        state = processor.set_image(image)

        # Incluye procesamiento del prompt, detección y máscaras.
        result = processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

        # El resultado final de este benchmark debe estar en CPU.
        # Estas copias forman parte del tiempo end-to-end.
        # No convertimos a NumPy ni guardamos imágenes o máscaras.
        cpu_result = {
            key: result[key].detach().cpu()
            for key in ("masks", "boxes", "scores")
        }

        detection_counts.append(int(cpu_result["scores"].numel()))

        # Liberamos las referencias de esta imagen antes de la siguiente.
        # No acumulamos predicciones de todo el conjunto en GPU o RAM.
        del cpu_result, result, state, image

    return detection_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default="fish")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats < 1:
        parser.error("--warmup debe ser >= 0 y --repeats >= 1")

    if not torch.cuda.is_available():
        raise RuntimeError("Este benchmark requiere CUDA.")

    # Validamos las rutas antes de cargar el modelo.
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    if not samples:
        raise ValueError("El manifiesto no contiene imágenes.")

    image_paths = [
        manifest_path.parent / sample["image"]
        for sample in samples
    ]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Evitamos sobrescribir una ejecución anterior.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", manifest)

    # Inicialización y carga: se registran por separado.
    torch.cuda.synchronize()
    start = time.perf_counter()
    model = build_sam3_image_model()
    model.eval()
    processor = Sam3Processor(model)
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start

    gpu = torch.cuda.get_device_properties(0)
    configuration = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu.name,
        "gpu_total_gib": gpu.total_memory / GIB,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cpu_threads": torch.get_num_threads(),
        "parameter_dtype": str(next(model.parameters()).dtype),
        "vitdet_mlp_implementation": "unfused_fp32",
        "autocast": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "sam3_git": git_info(Path(sam3.__file__).resolve().parent),
        "benchmark_git": git_info(ROOT),
        "checkpoint": "builder_default; not independently identified",
        "prompt": args.prompt,
        "batch_size": 1,
        "num_images": len(image_paths),
        "warmup_passes": args.warmup,
        "measured_passes": args.repeats,
        "model_and_processor_setup_seconds": setup_seconds,
        "measurement": "image_read_to_cpu_predictions",
        "output_policy": "consume_and_release_each_image",
    }
    save_json(output / "configuration.json", configuration)

    print(f"GPU: {gpu.name}", flush=True)
    print(f"Imágenes por pasada: {len(image_paths)}", flush=True)
    print(f"Carga y preparación: {setup_seconds:.2f} s", flush=True)

    rows = []

    # Desactiva autograd durante calentamiento y mediciones.
    with torch.inference_mode():
        # El calentamiento recorre el conjunto completo.
        # Inicializa rutas de ejecución y calienta las cachés.
        for index in range(args.warmup):
            run_images(processor, image_paths, args.prompt)
            torch.cuda.synchronize()
            print(
                f"Calentamiento {index + 1}/{args.warmup}",
                flush=True,
            )

        # CSV incremental: conserva las repeticiones ya terminadas
        # si una ejecución posterior falla.
        fields = [
            "repeat",
            "num_images",
            "total_seconds",
            "images_per_second",
            "amortized_ms_per_image",
            "peak_allocated_gib",
            "peak_reserved_gib",
            "detections_per_image",
        ]
        with (output / "measurements.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()

            for index in range(args.repeats):
                # Esperamos el trabajo anterior antes de iniciar el reloj.
                torch.cuda.synchronize()

                # El pico incluye pesos y tensores vivos del proceso.
                # Reiniciar estadísticas no libera memoria.
                torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                counts = run_images(
                    processor, image_paths, args.prompt
                )

                # CUDA es asíncrona: esperamos antes de parar el reloj.
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                row = {
                    "repeat": index + 1,
                    "num_images": len(image_paths),
                    "total_seconds": elapsed,
                    "images_per_second": len(image_paths) / elapsed,
                    "amortized_ms_per_image": (
                        elapsed * 1000 / len(image_paths)
                    ),
                    "peak_allocated_gib": (
                        torch.cuda.max_memory_allocated() / GIB
                    ),
                    "peak_reserved_gib": (
                        torch.cuda.max_memory_reserved() / GIB
                    ),
                    "detections_per_image": json.dumps(counts),
                }
                rows.append(row)
                writer.writerow(row)
                file.flush()

                # Imprimir queda fuera del intervalo medido.
                print(
                    f"Repetición {index + 1}/{args.repeats}: "
                    f"{elapsed:.3f} s | "
                    f"{row['images_per_second']:.2f} imágenes/s | "
                    f"pico: {row['peak_allocated_gib']:.2f} GiB",
                    flush=True,
                )

    times = np.array([row["total_seconds"] for row in rows])
    throughput = np.array([row["images_per_second"] for row in rows])

    summary = {
        "num_images_per_pass": len(image_paths),
        "measured_passes": args.repeats,
        "median_pass_seconds": float(np.median(times)),
        "p95_pass_seconds": float(np.percentile(times, 95)),
        "min_pass_seconds": float(times.min()),
        "max_pass_seconds": float(times.max()),
        "median_images_per_second": float(np.median(throughput)),
        "overall_images_per_second": float(
            len(image_paths) * args.repeats / times.sum()
        ),
        "max_peak_allocated_gib": max(
            row["peak_allocated_gib"] for row in rows
        ),
        "max_peak_reserved_gib": max(
            row["peak_reserved_gib"] for row in rows
        ),
    }
    save_json(output / "summary.json", summary)

    print("\nResumen:", flush=True)
    print(json.dumps(summary, indent=2))
    print(f"\nInformes: {output}")


if __name__ == "__main__":
    main()