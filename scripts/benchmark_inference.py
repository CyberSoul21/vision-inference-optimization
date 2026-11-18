"""End-to-end benchmark of SAM 3 with batch size 1."""

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


# Keep the adaptation used in previous tests.
# Prevent addmm_act from explicitly casting activations to BF16.
def linear_activation_fp32(activation, linear, x):
    return activation()(linear(x))


vitdet.addmm_act = linear_activation_fp32


def save_json(path, content):
    """Saving reports happens outside the measured interval."""
    path.write_text(
        json.dumps(content, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def git_info(directory):
    """Record the commit and whether there are local changes, when Git is available."""
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
    """One sequential pass: each call processes one image."""
    detection_counts = []

    for image_path in image_paths:
        # Reading, decoding and RGB conversion are included.
        with Image.open(image_path) as source:
            image = source.convert("RGB")

        # Includes input preparation and the vision encoder.
        # Create a fresh state: do not reuse features
        # computed for the same image in a previous repeat.
        state = processor.set_image(image)

        # Includes prompt processing, detection and masks.
        result = processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

        # The final result of this benchmark must be on the CPU.
        # These copies are part of the end-to-end time.
        # We do not convert to NumPy or save images or masks.
        cpu_result = {
            key: result[key].detach().cpu()
            for key in ("masks", "boxes", "scores")
        }

        detection_counts.append(int(cpu_result["scores"].numel()))

        # Release this image's references before the next one.
        # Do not accumulate predictions for the whole set in GPU or RAM.
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
        raise RuntimeError("This benchmark requires CUDA.")

    # Validate paths before loading the model.
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest["samples"]

    if not samples:
        raise ValueError("The manifest contains no images.")

    image_paths = [
        manifest_path.parent / sample["image"]
        for sample in samples
    ]
    for path in image_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    # Avoid overwriting a previous run.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "manifest.json", manifest)

    # Initialization and loading: recorded separately.
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
    print(f"Images per pass: {len(image_paths)}", flush=True)
    print(f"Loading and setup: {setup_seconds:.2f} s", flush=True)

    rows = []

    # Disable autograd during warmup and measurements.
    with torch.inference_mode():
        # Warmup goes through the full set.
        # Initializes execution paths and warms up caches.
        for index in range(args.warmup):
            run_images(processor, image_paths, args.prompt)
            torch.cuda.synchronize()
            print(
                f"Warmup {index + 1}/{args.warmup}",
                flush=True,
            )

        # Incremental CSV: keeps already completed repeats
        # if a later run fails.
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
                # Wait for previous work before starting the clock.
                torch.cuda.synchronize()

                # The peak includes weights and live tensors of the process.
                # Resetting statistics does not free memory.
                torch.cuda.reset_peak_memory_stats()

                start = time.perf_counter()
                counts = run_images(
                    processor, image_paths, args.prompt
                )

                # CUDA is asynchronous: wait before stopping the clock.
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

                # Printing stays outside the measured interval.
                print(
                    f"Repeat {index + 1}/{args.repeats}: "
                    f"{elapsed:.3f} s | "
                    f"{row['images_per_second']:.2f} images/s | "
                    f"peak: {row['peak_allocated_gib']:.2f} GiB",
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

    print("\nSummary:", flush=True)
    print(json.dumps(summary, indent=2))
    print(f"\nReports: {output}")


if __name__ == "__main__":
    main()