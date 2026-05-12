"""compile_image_with_calibration.py

Compile image ONNX on AI Hub with compile-time int8 quantization + real calibration images.

Why compile-time (not submit_quantize_job):
  - submit_quantize_job inserts Q/DQ nodes -> QNN can't fuse -> slower or broken
  - submit_compile_job with --quantize_full_type int8 does in-place fusion -> correct & fast

Calibration images come from our competition-like dataset
(track1_test_eval_v2/frozen_v1/calibration + blind + track1_sample/sample).

Usage:
  python compile_image_with_calibration.py \\
    --onnx_dir onnx_models/models_soup_fresh2_x_s05e1_a042 \\
    --label fresh2_img_int8 \\
    --txt_compile_id jpx7j983g

  # Or compile both if no txt_compile_id:
  python compile_image_with_calibration.py \\
    --onnx_dir onnx_models/models_soup_fresh2_x_s05e1_a042 \\
    --label fresh2_img_int8
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, ".")
from lpcvc2026.modules.data import BASE_DIR

SHARE_EMAIL = "lowpowervision@gmail.com"

IMG_SIZE = 224  # Standard MobileCLIP-B input


def load_calibration_images(extra_dirs=None):
    """Load images from competition-like dataset directories."""
    image_dirs = [
        os.path.join(BASE_DIR, "track1_test_eval_v2", "frozen_v1", "calibration", "images"),
        os.path.join(BASE_DIR, "track1_test_eval_v2", "frozen_v1", "blind", "images"),
        os.path.join(BASE_DIR, "track1_sample", "sample", "images"),
    ]
    if extra_dirs:
        image_dirs.extend(extra_dirs)

    paths = []
    for d in image_dirs:
        if not os.path.isdir(d):
            print(f"  [skip] not found: {d}")
            continue
        found = sorted(
            os.path.join(d, f)
            for f in os.listdir(d)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        print(f"  {d}: {len(found)} images")
        paths.extend(found)

    # Deduplicate by filename (in case dirs overlap)
    seen = set()
    unique_paths = []
    for p in paths:
        fn = os.path.basename(p)
        if fn not in seen:
            seen.add(fn)
            unique_paths.append(p)

    print(f"  total unique images: {len(unique_paths)}")
    if not unique_paths:
        raise RuntimeError("No calibration images found. Check image_dirs.")
    return unique_paths


def preprocess_image(path, img_size=IMG_SIZE):
    """Preprocess image: resize + center crop -> [0,1] float32 (1,3,H,W).
    
    Competition evaluator uses simple /255.0, no mean/std normalization.
    """
    img = Image.open(path).convert("RGB")
    # Resize shortest side to img_size
    w, h = img.size
    if w < h:
        new_w, new_h = img_size, int(h * img_size / w)
    else:
        new_h, new_w = img_size, int(w * img_size / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    # Center crop
    left = (new_w - img_size) // 2
    top = (new_h - img_size) // 2
    img = img.crop((left, top, left + img_size, top + img_size))
    arr = np.array(img, dtype=np.float32) / 255.0  # [0,1]
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr[np.newaxis]  # (1,3,H,W)
    return arr


def build_calibration_data(image_paths, img_size=IMG_SIZE, max_images=512):
    """Load and preprocess images into AI Hub dataset format."""
    if len(image_paths) > max_images:
        # Evenly subsample
        step = len(image_paths) / max_images
        image_paths = [image_paths[int(i * step)] for i in range(max_images)]
        print(f"  subsampled to {len(image_paths)} images (max_images={max_images})")

    arrays = []
    for p in image_paths:
        try:
            arrays.append(preprocess_image(p, img_size))
        except Exception as e:
            print(f"  [warn] skip {p}: {e}")

    print(f"  preprocessed: {len(arrays)} images, shape={arrays[0].shape}, dtype={arrays[0].dtype}")
    return {"image": arrays}


def wait_for_job(job, label, poll_sec=15):
    print(f"  waiting: {label} ({job.job_id})")
    while True:
        status = job.get_status()
        if getattr(status, "finished", False):
            print(f"  done: {label} ({job.job_id})")
            return
        time.sleep(poll_sec)


def get_latency_us(profile_job):
    result = profile_job.download_profile()
    return int(result["execution_summary"]["estimated_inference_time"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_dir",
        required=True,
        help="Directory that contains img_dir/ (and optionally txt_dir/).",
    )
    parser.add_argument("--img_size", type=int, default=IMG_SIZE, help="Image size (default 224).")
    parser.add_argument("--max_images", type=int, default=512, help="Max calibration images (default 512).")
    parser.add_argument("--label", default="img_int8_calib", help="Job name suffix.")
    parser.add_argument(
        "--txt_compile_id",
        default=None,
        help="Reuse an existing text compile job ID. If omitted, compile text from txt_dir/.",
    )
    parser.add_argument(
        "--txt_len",
        type=int,
        default=77,
        help="Text input length when compiling text encoder (default 77).",
    )
    parser.add_argument("--share_email", default=SHARE_EMAIL)
    parser.add_argument("--qairt_version", default=None, help="Optional QAIRT version override.")
    parser.add_argument(
        "--quantize_full_type",
        default="int8",
        choices=["int8", "int16"],
        help="Compile-time quantization type for image encoder (default int8).",
    )
    parser.add_argument("--profile_iterations", type=int, default=100)
    parser.add_argument("--no_profile", action="store_true", help="Skip profiling.")
    parser.add_argument("--poll_sec", type=int, default=15)
    parser.add_argument(
        "--extra_img_dirs",
        nargs="*",
        default=None,
        help="Additional image directories for calibration (e.g. train2014/).",
    )
    args = parser.parse_args()

    import qai_hub as hub

    img_dir = os.path.join(args.onnx_dir, "img_dir")
    txt_dir = os.path.join(args.onnx_dir, "txt_dir")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if args.txt_compile_id is None and not os.path.exists(txt_dir):
        raise FileNotFoundError(
            f"txt_dir not found: {txt_dir}. Provide --txt_compile_id to reuse existing."
        )

    device = hub.Device("XR2 Gen 2 (Proxy)")

    print("\n[1/6] Loading calibration images")
    image_paths = load_calibration_images(extra_dirs=args.extra_img_dirs)

    print("\n[2/6] Preprocessing images")
    calib_data = build_calibration_data(image_paths, img_size=args.img_size, max_images=args.max_images)

    print("\n[3/6] Uploading calibration dataset to AI Hub")
    dataset = hub.upload_dataset(
        calib_data,
        name=f"lpcvc_img_calib_{args.img_size}px_{args.label}",
    )
    print(f"  dataset_id: {dataset.dataset_id}")

    base_options = "--target_runtime qnn_dlc --truncate_64bit_io"
    if args.qairt_version:
        base_options += f" --qairt_version {args.qairt_version}"
    img_options = f"{base_options} --quantize_full_type {args.quantize_full_type}"

    print(f"\n[4/6] Submitting image compile (int8, {args.quantize_full_type}, calibration_data)")
    img_compile = hub.submit_compile_job(
        model=img_dir + "/",
        device=device,
        input_specs={"image": ((1, 3, args.img_size, args.img_size), "float32")},
        name=f"LPCVC_ImgEnc_{args.label}",
        options=img_options,
        calibration_data=dataset,
    )
    print(f"  image compile job: {img_compile.job_id}")
    if args.share_email:
        img_compile.modify_sharing(add_emails=[args.share_email])

    # Text: reuse or compile fresh (FP16, no quantization)
    if args.txt_compile_id:
        txt_compile = hub.get_job(args.txt_compile_id)
        txt_compile_id = args.txt_compile_id
        print(f"  reusing text compile: {txt_compile_id}")
    else:
        print("  submitting text compile (FP16, no quantization)")
        txt_compile = hub.submit_compile_job(
            model=txt_dir + "/",
            device=device,
            input_specs={"text": ((1, args.txt_len), "int64")},
            name=f"LPCVC_TxtEnc_{args.label}",
            options=base_options,
        )
        txt_compile_id = txt_compile.job_id
        print(f"  text compile job: {txt_compile_id}")
        if args.share_email:
            txt_compile.modify_sharing(add_emails=[args.share_email])

    print("\n[5/6] Waiting for compile jobs")
    wait_for_job(img_compile, "image compile", poll_sec=args.poll_sec)
    img_target = img_compile.get_target_model()
    if img_target is None:
        raise RuntimeError(f"Image compile FAILED: {img_compile.job_id}")

    if args.txt_compile_id is None:
        wait_for_job(txt_compile, "text compile", poll_sec=args.poll_sec)
    txt_target = txt_compile.get_target_model()
    if txt_target is None:
        raise RuntimeError(f"Text compile FAILED: {txt_compile_id}")

    print("\n[5/6] Compile completed")
    print(f"  Image: {img_compile.job_id}")
    print(f"  Text : {txt_compile_id}")

    if args.no_profile:
        print("\nSkipping profiling (--no_profile).")
        print(f"\nCompile Job IDs for submission:\n  Image: {img_compile.job_id}\n  Text : {txt_compile_id}")
        return

    print("\n[6/6] Profiling latency on XR2 Gen 2 (Proxy)")
    profile_opts = f"--max_profiler_iterations {args.profile_iterations}"
    img_profile = hub.submit_profile_job(
        model=img_target,
        device=device,
        name=f"LPCVC_ImgEnc_{args.label}_profile",
        options=profile_opts,
    )
    txt_profile = hub.submit_profile_job(
        model=txt_target,
        device=device,
        name=f"LPCVC_TxtEnc_{args.label}_profile",
        options=profile_opts,
    )
    print(f"  image profile: {img_profile.job_id}")
    print(f"  text  profile: {txt_profile.job_id}")
    if args.share_email:
        img_profile.modify_sharing(add_emails=[args.share_email])
        txt_profile.modify_sharing(add_emails=[args.share_email])

    wait_for_job(img_profile, "image profile", poll_sec=args.poll_sec)
    wait_for_job(txt_profile, "text profile", poll_sec=args.poll_sec)

    img_lat = get_latency_us(img_profile)
    txt_lat = get_latency_us(txt_profile)
    total_lat = img_lat + txt_lat

    print("\n" + "=" * 50)
    print("RESULTS")
    print(f"  Image Latency : {img_lat:,} us  (FP16 baseline: ~12,893 us)")
    print(f"  Text  Latency : {txt_lat:,} us")
    print(f"  Total Latency : {total_lat:,} us")
    delta_img = img_lat - 12893
    print(f"  Image delta   : {delta_img:+,} us vs FP16 baseline")
    print(f"\nCompile Job IDs for submission:")
    print(f"  Image: {img_compile.job_id}")
    print(f"  Text : {txt_compile_id}")
    print("=" * 50)


if __name__ == "__main__":
    main()
