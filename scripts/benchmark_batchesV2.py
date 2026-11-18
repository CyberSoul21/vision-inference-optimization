"""P5: batching oficial de SAM 3; tiempo end-to-end, VRAM y energia GPU.

Ejecutar cada batch size en un proceso separado. Requiere una GPU exclusiva.
No modifica SAM 3 ni reemplaza su MLP. Sigue el notebook oficial con autocast BF16.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pynvml as nvml
import torch
from PIL import Image

import sam3
from sam3.eval.postprocessors import PostProcessImage
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import (
    Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata,
)
from sam3.train.transforms.basic_for_api import (
    ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI,
)


# ---------------------------------------------------------------------------
# Parametros fijos del experimento
# ---------------------------------------------------------------------------
BATCH_SIZES = [1, 4, 8, 16, 32]
RESOLUTION = 1008   # Lado del cuadrado al que se redimensiona cada imagen
THRESHOLD = 0.5     # Puntuacion minima para conservar una deteccion


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------
def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, records):
    """Reescribe el CSV completo; las columnas salen de las claves del primer registro."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    """Hash SHA-256 leyendo por bloques, para no cargar en memoria ficheros de varios GB."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(path):
    """El diff conserva cambios locales que el commit por si solo no identifica."""
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    try:
        return {"commit": run("rev-parse", "HEAD"),
                "status": run("status", "--porcelain"), "diff": run("diff", "HEAD")}
    except (OSError, subprocess.CalledProcessError):
        return None


def check_exclusive(handle):
    """La energia es de toda la GPU: otro proceso invalidaria su atribucion."""
    others = [p.pid for p in nvml.nvmlDeviceGetComputeRunningProcesses(handle)
              if p.pid != os.getpid()]
    if others:
        raise RuntimeError(f"GPU compartida con procesos {others}; no medir energia")


# ---------------------------------------------------------------------------
# Entradas: argumentos y manifest
# ---------------------------------------------------------------------------
def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, choices=BATCH_SIZES, required=True)
    p.add_argument("--prompt", default="fish")
    p.add_argument("--warmup", type=int, default=5, help="Pasadas completas no medidas")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--checkpoint", type=Path, help="Checkpoint explicito, si se dispone de el")
    p.add_argument("--save-predictions", action="store_true",
                   help="Una pasada EXTRA sin cronometrar para inspeccionar predicciones")
    a = p.parse_args()
    if a.warmup < 0 or a.repeats < 1:
        p.error("Se requiere warmup >= 0 y repeats >= 1")
    return a


def load_manifest(path, batch_size):
    """Lee el manifest y comprueba que todas las imagenes existen y se reparten en batches completos.

    Devuelve la ruta absoluta (las imagenes son relativas a su carpeta) y el contenido.
    """
    manifest_path = path.resolve()
    manifest = json.loads(manifest_path.read_text())
    samples = manifest["samples"]
    if not samples or len(samples) % batch_size:
        raise ValueError("El numero de imagenes debe ser positivo y divisible por batch size")
    for s in samples:
        if not (manifest_path.parent / s["image"]).is_file():
            raise FileNotFoundError(manifest_path.parent / s["image"])
    if len({s["sample_id"] for s in samples}) != len(samples):
        raise ValueError("Los sample_id deben ser unicos")
    return manifest_path, manifest


# ---------------------------------------------------------------------------
# Metadatos para reproducir el experimento
# ---------------------------------------------------------------------------
def build_configuration(args, samples, gpu_uuid):
    """Todo lo necesario para saber, a posteriori, con que se obtuvo cada resultado."""
    return {
        # Entorno
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(), "python": platform.python_version(),
        "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "gpu_uuid": gpu_uuid,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver": str(nvml.nvmlSystemGetDriverVersion()),
        "torch_cpu_threads": torch.get_num_threads(),
        # Parametros del experimento
        "batch_size": args.batch_size, "num_images": len(samples),
        "warmup_passes": args.warmup, "measured_passes": args.repeats,
        "prompt": args.prompt, "resolution": [RESOLUTION, RESOLUTION], "threshold": THRESHOLD,
        "autocast_dtype": "bfloat16", "tf32_allowed": True, "mlp": "original_unmodified",
        # Version exacta del codigo y del modelo
        "sam3_git": git_state(Path(sam3.__file__).parent),
        "benchmark_git": git_state(Path(__file__).parent),
        "script_sha256": sha256_file(__file__),
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else "builder_default_unresolved",
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        # Que se mide exactamente (para interpretar los numeros)
        "measurement": "local_file_read_to_cpu_predictions",
        "execution": "serial_batches_no_explicit_prefetch_or_overlap",
        "energy_scope": "whole_GPU_gross_energy_including_idle_during_CPU_work",
        "disk_cache": "not_flushed; warmup_reads_same_files",
    }


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    args = arguments()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Se requiere exactamente una GPU CUDA visible")
    manifest_path, manifest = load_manifest(args.manifest, args.batch_size)
    samples = manifest["samples"]
    if args.checkpoint and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "manifest.json", manifest)

    # CUDA renumera la GPU seleccionada como cuda:0. Su UUID real permite
    # encontrar el dispositivo NVML correcto sin interpretar indices fisicos.
    gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not gpu_uuid.startswith("GPU-"):
        gpu_uuid = "GPU-" + gpu_uuid
    print(f"GPU seleccionada: {torch.cuda.get_device_name(0)} | {gpu_uuid}", flush=True)
    nvml.nvmlInit()
    try:
        try:
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
        except TypeError:
            handle = nvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
        check_exclusive(handle)
        nvml.nvmlDeviceGetTotalEnergyConsumption(handle)  # Fallar si no esta soportado.

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        config = build_configuration(args, samples, gpu_uuid)
        save_json(args.output / "configuration.json", config)

        print("Cargando modelo original...", flush=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        kwargs = {"checkpoint_path": str(args.checkpoint)} if args.checkpoint else {}
        model = build_sam3_image_model(**kwargs).eval()
        torch.cuda.synchronize()
        config["setup_seconds"] = time.perf_counter() - t
        config["parameter_dtype"] = str(next(model.parameters()).dtype)
        save_json(args.output / "configuration.json", config)

        transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=RESOLUTION, max_size=RESOLUTION, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5] * 3, std=[0.5] * 3),
        ])
        postprocessor = PostProcessImage(
            max_dets_per_img=-1, iou_type="segm",
            use_original_sizes_box=True, use_original_sizes_mask=True,
            convert_mask_to_rle=False, detection_threshold=THRESHOLD, to_cpu=False,
        )

        def one_batch(start, save=False):
            """Todas las etapas estan incluidas; variables GPU mueren al retornar."""
            points = []
            group = samples[start:start + args.batch_size]
            for offset, sample in enumerate(group):
                with Image.open(manifest_path.parent / sample["image"]) as src:
                    image = src.convert("RGB")
                width, height = image.size
                query_id = start + offset + 1
                point = Datapoint(find_queries=[], images=[
                    SAMImage(data=image, objects=[], size=[height, width])
                ])
                point.find_queries.append(FindQueryLoaded(
                    query_text=args.prompt, image_id=0, object_ids_output=[],
                    is_exhaustive=True, query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query_id, original_image_id=query_id,
                        original_category_id=1, original_size=[height, width],
                        object_id=0, frame_index=0,
                    ),
                ))
                points.append(transform(point))
            # Collate crea un batch real, no un bucle de forwards individuales.
            batch = collate_fn_api(points, dict_key="input")["input"]
            batch = copy_data_to_device(batch, torch.device("cuda"), non_blocking=True)
            output = model(batch)
            results = postprocessor.process_results(output, batch.find_metadatas)
            counts = []
            for offset, sample in enumerate(group):
                item = results[start + offset + 1]
                cpu = {k: item[k].detach().cpu() for k in ("masks", "boxes", "scores")}
                counts.append(int(cpu["scores"].numel()))
                if save:  # Esta ruta SOLO se ejecuta en una pasada extra no medida.
                    dest = args.output / "predictions"
                    dest.mkdir(exist_ok=True)
                    np.savez_compressed(dest / (sample["sample_id"] + ".npz"),
                                        **{k: v.float().numpy() if v.is_floating_point()
                                           else v.numpy() for k, v in cpu.items()})
                del cpu
            return counts

        def run_pass(save=False):
            counts = []
            batch_times = []
            for start in range(0, len(samples), args.batch_size):
                # Baseline serial: cada batch termina en CPU antes del siguiente.
                begin = time.perf_counter()
                counts.extend(one_batch(start, save=save))
                torch.cuda.synchronize()
                batch_times.append(time.perf_counter() - begin)
            return counts, batch_times

        rows = []
        latency_rows = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(args.warmup):
                run_pass()
                torch.cuda.synchronize()
                print(f"Warmup {i + 1}/{args.warmup}", flush=True)
            reference_counts = None
            for i in range(args.repeats):
                check_exclusive(handle)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                # La ventana energetica engloba ligeramente la ventana temporal.
                e0 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                start = time.perf_counter()
                counts, latencies = run_pass()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                e1 = nvml.nvmlDeviceGetTotalEnergyConsumption(handle)
                check_exclusive(handle)
                if e1 <= e0:
                    raise RuntimeError("El contador de energia no avanzo: medicion invalida")
                if reference_counts is None:
                    reference_counts = counts
                elif counts != reference_counts:
                    print("Aviso: cambiaron los recuentos de detecciones", flush=True)
                joules = (e1 - e0) / 1000
                rows.append({
                    "repeat": i + 1, "batch_size": args.batch_size,
                    "num_images": len(samples), "total_seconds": elapsed,
                    "images_per_second": len(samples) / elapsed,
                    "amortized_ms_per_image": elapsed * 1000 / len(samples),
                    "gpu_energy_j": joules, "gpu_j_per_image": joules / len(samples),
                    "gpu_average_power_w_approx": joules / elapsed,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    "detections_per_image": json.dumps(counts),
                })
                latency_rows.extend({"repeat": i + 1, "batch_index": j,
                                     "batch_size": args.batch_size, "latency_ms": s * 1000}
                                    for j, s in enumerate(latencies))
                # Escritura e impresion fuera de las ventanas de tiempo y energia.
                write_csv(args.output / "measurements.csv", rows)
                write_csv(args.output / "batch_latencies.csv", latency_rows)
                print(f"{i + 1}/{args.repeats}: {elapsed:.3f}s | "
                      f"{len(samples)/elapsed:.2f} img/s | {joules/len(samples):.3f} J/img", flush=True)
            if args.save_predictions:
                print("Pasada EXTRA de validacion, no medida...", flush=True)
                run_pass(save=True)

        def median(key):
            return float(np.median([r[key] for r in rows]))
        save_json(args.output / "summary.json", {
            "status": "completed", "batch_size": args.batch_size,
            "num_images_per_pass": len(samples), "measured_passes": len(rows),
            "median_pass_seconds": median("total_seconds"),
            "overall_images_per_second": len(samples) * len(rows) / sum(r["total_seconds"] for r in rows),
            "median_gpu_j_per_image": median("gpu_j_per_image"),
            "overall_gpu_j_per_image": sum(r["gpu_energy_j"] for r in rows) / (len(samples) * len(rows)),
            "median_batch_latency_ms": float(np.median([r["latency_ms"] for r in latency_rows])),
            "p95_batch_latency_ms": float(np.percentile([r["latency_ms"] for r in latency_rows], 95)),
            "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
            "max_peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
            "detections_per_image": reference_counts,
        })
        print(f"Completado: {args.output.resolve()}", flush=True)
    except torch.OutOfMemoryError as exc:
        save_json(args.output / "status.json", {"status": "oom", "batch_size": args.batch_size, "error": str(exc)})
        print("OOM registrado. Esta configuracion no cabe; no se reduce el batch automaticamente.", flush=True)
        raise SystemExit(2)
    except Exception as exc:
        save_json(args.output / "status.json", {"status": "error", "error": repr(exc)})
        raise
    finally:
        nvml.nvmlShutdown()


if __name__ == "__main__":
    main()
