"""compile_text_with_calibration.py

Compile text ONNX on AI Hub with compile-time quantization + custom calibration data.

Why this script:
- Uses submit_compile_job(..., calibration_data=dataset, options="... --quantize_full_type int8")
- Avoids submit_quantize_job() so it does not pre-insert Q/DQ nodes into ONNX.

Typical usage:
  python compile_text_with_calibration.py \
    --onnx_dir onnx_models/models_soup_r20_r20e1x0p7_fresh2x0p25_s05e1x0p05 \
    --text_len 77 \
    --label soup_r20_compile_calib \
    --img_compile_id jgklzmlo5
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import open_clip

sys.path.insert(0, ".")
from lpcvc2026.modules.data import BASE_DIR


SHARE_EMAIL = "lowpowervision@gmail.com"


def load_all_competition_texts():
    """Load unique texts from calibration + blind + sample sets."""
    texts = set()

    cal_path = os.path.join(
        BASE_DIR, "track1_test_eval_v2", "frozen_v1", "calibration", "test_texts.csv"
    )
    if os.path.exists(cal_path):
        with open(cal_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                texts.add(row["Unique_Texts"])
        print(f"  calibration loaded: {len(texts)}")

    blind_path = os.path.join(
        BASE_DIR, "track1_test_eval_v2", "frozen_v1", "blind", "test_texts.csv"
    )
    before = len(texts)
    if os.path.exists(blind_path):
        with open(blind_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                texts.add(row["Unique_Texts"])
        print(f"  blind added: {len(texts) - before}")

    sample_path = os.path.join(
        BASE_DIR, "track1_sample", "sample", "SampleDataset_Textnums_to_Texts 1.csv"
    )
    before = len(texts)
    if os.path.exists(sample_path):
        with open(sample_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                texts.add(row["Unique_Texts"])
        print(f"  sample added: {len(texts) - before}")

    texts = sorted(texts)
    print(f"  total unique texts: {len(texts)}")
    if not texts:
        raise RuntimeError("No texts loaded from competition CSV files.")
    return texts


def build_calibration_data(texts, text_len=77, arch="MobileCLIP-B"):
    """Tokenize texts and build AI Hub dataset entries."""
    tokenizer = open_clip.get_tokenizer(arch)
    all_tokens = []
    batch_size = 256

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        t = tokenizer(batch)  # (B, 77), int64
        if text_len != 77:
            t = t[:, :text_len]
        t = t.numpy().astype(np.int64)
        for j in range(len(batch)):
            all_tokens.append(t[j : j + 1])  # (1, text_len)

    print(
        f"  tokenized samples: {len(all_tokens)}, "
        f"shape={all_tokens[0].shape}, dtype={all_tokens[0].dtype}"
    )
    return {"text": all_tokens}


def wait_for_job(job, label, poll_sec=15):
    """Poll job status without using job.wait() to avoid cp950 unicode issues."""
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
        help="Directory that contains txt_dir/ (and optionally img_dir/).",
    )
    parser.add_argument(
        "--text_len",
        type=int,
        default=77,
        help="Text input length for compile input_specs (77 for eval_wrapper exports).",
    )
    parser.add_argument("--arch", default="MobileCLIP-B", help="Tokenizer arch name.")
    parser.add_argument("--label", default="compile_calib", help="Job name suffix.")
    parser.add_argument(
        "--img_compile_id",
        default=None,
        help="Reuse an existing image compile job ID. If omitted, compile image from img_dir/.",
    )
    parser.add_argument(
        "--share_email",
        default=SHARE_EMAIL,
        help="Share compile/profile jobs to this email.",
    )
    parser.add_argument(
        "--qairt_version",
        default=None,
        help="Optional QAIRT version override, e.g. 2.45.0",
    )
    parser.add_argument(
        "--quantize_full_type",
        default="int8",
        choices=["int8", "int16"],
        help="Compile-time quantization type for text compile job.",
    )
    parser.add_argument(
        "--profile_iterations",
        type=int,
        default=100,
        help="Profiler iterations for profile jobs.",
    )
    parser.add_argument("--no_profile", action="store_true", help="Skip profiling.")
    parser.add_argument("--poll_sec", type=int, default=15, help="Polling interval.")
    args = parser.parse_args()

    import qai_hub as hub

    txt_dir = os.path.join(args.onnx_dir, "txt_dir")
    img_dir = os.path.join(args.onnx_dir, "img_dir")
    if not os.path.exists(txt_dir):
        raise FileNotFoundError(f"txt_dir not found: {txt_dir}")
    if args.img_compile_id is None and not os.path.exists(img_dir):
        raise FileNotFoundError(f"img_dir not found: {img_dir}")

    device = hub.Device("XR2 Gen 2 (Proxy)")

    print("\n[1/6] Loading competition texts")
    texts = load_all_competition_texts()

    print("\n[2/6] Building calibration dataset entries")
    calib_data = build_calibration_data(texts, text_len=args.text_len, arch=args.arch)

    print("\n[3/6] Uploading calibration dataset")
    dataset = hub.upload_dataset(
        calib_data, name=f"lpcvc_compile_calib_{args.text_len}tok_{args.label}"
    )
    print(f"  dataset_id: {dataset.dataset_id}")

    base_options = "--target_runtime qnn_dlc --truncate_64bit_io"
    if args.qairt_version:
        base_options += f" --qairt_version {args.qairt_version}"
    txt_options = f"{base_options} --quantize_full_type {args.quantize_full_type}"

    print("\n[4/6] Submitting text compile (compile-time quantization + calibration_data)")
    txt_compile = hub.submit_compile_job(
        model=txt_dir + "/",
        device=device,
        input_specs={"text": ((1, args.text_len), "int64")},
        name=f"LPCVC_TxtEnc_{args.label}",
        options=txt_options,
        calibration_data=dataset,
    )
    print(f"  text compile job: {txt_compile.job_id}")
    if args.share_email:
        txt_compile.modify_sharing(add_emails=[args.share_email])

    if args.img_compile_id:
        img_compile = hub.get_job(args.img_compile_id)
        img_compile_id = args.img_compile_id
        print(f"  reusing image compile: {img_compile_id}")
    else:
        print("  submitting image compile")
        img_compile = hub.submit_compile_job(
            model=img_dir + "/",
            device=device,
            input_specs={"image": (1, 3, 224, 224)},
            name=f"LPCVC_ImgEnc_{args.label}",
            options=base_options,
        )
        img_compile_id = img_compile.job_id
        print(f"  image compile job: {img_compile_id}")
        if args.share_email:
            img_compile.modify_sharing(add_emails=[args.share_email])

    wait_for_job(txt_compile, "text compile", poll_sec=args.poll_sec)
    txt_target = txt_compile.get_target_model()
    if txt_target is None:
        raise RuntimeError(f"Text compile failed: {txt_compile.job_id}")

    if args.img_compile_id is None:
        wait_for_job(img_compile, "image compile", poll_sec=args.poll_sec)
    img_target = img_compile.get_target_model()
    if img_target is None:
        raise RuntimeError(f"Image compile failed: {img_compile_id}")

    print("\n[5/6] Compile completed")
    print(f"  Image: {img_compile_id}")
    print(f"  Text : {txt_compile.job_id}")

    if args.no_profile:
        return

    print("\n[6/6] Profiling")
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

    print("\nRESULTS")
    print(f"  Image Latency: {img_lat:,} us")
    print(f"  Text  Latency: {txt_lat:,} us")
    print(f"  Total Latency: {total_lat:,} us")
    print("\nCompile Job IDs for submission:")
    print(f"  Image: {img_compile_id}")
    print(f"  Text : {txt_compile.job_id}")


if __name__ == "__main__":
    main()
