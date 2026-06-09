"""Apply trained super-resolution models to CT images.

This is a readability-focused refactor of ``resolve_sr.py``.  The computation,
file naming conventions, metrics, and output layout are intentionally preserved.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import natsort
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim

import io_func
import quant_util
import util

NUM_CHANNELS = 1
QUANT_HEADER = (
    "chckpt-no, CNN rMSE, (+,-std), CNN PSNR [dB], (+,-std), "
    "CNN SSIM, (+,-std), CNN HD, (+,-std), BC rMSE, (+,-std), "
    "BC PSNR [dB], (+,-std), BC SSIM, (+,-std), BC HD, (+,-std)\n"
)


@dataclass
class MetricStore:
    """Collect metric values for CNN output and bicubic baseline."""

    lr_rmse: List[float] = field(default_factory=list)
    lr_psnr: List[float] = field(default_factory=list)
    lr_ssim: List[float] = field(default_factory=list)
    lr_hd: List[float] = field(default_factory=list)
    cnn_rmse: List[float] = field(default_factory=list)
    cnn_psnr: List[float] = field(default_factory=list)
    cnn_ssim: List[float] = field(default_factory=list)
    cnn_hd: List[float] = field(default_factory=list)

    def append_cnn(self, rmse: float, psnr: float, ssim: float, hd: float) -> None:
        self.cnn_rmse.append(rmse)
        self.cnn_psnr.append(psnr)
        self.cnn_ssim.append(ssim)
        self.cnn_hd.append(hd)

    def append_lr(self, rmse: float, psnr: float, ssim: float, hd: float) -> None:
        self.lr_rmse.append(rmse)
        self.lr_psnr.append(psnr)
        self.lr_ssim.append(ssim)
        self.lr_hd.append(hd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch application of trained weights on CT images"
    )
    parser.add_argument(
        "--model-name",
        "--m",
        type=str,
        default="cnn3",
        help="Network architecture name. Supported here: srgan, srwgan, fsrcnn.",
    )
    parser.add_argument(
        "--input-folder",
        type=str,
        required=True,
        help="Directory containing noisy input test images.",
    )
    parser.add_argument(
        "--gt-folder",
        type=str,
        default="",
        help="Directory containing test ground-truth images.",
    )
    parser.add_argument(
        "--model-folder",
        type=str,
        required=True,
        help="Directory containing saved checkpoints.",
    )
    parser.add_argument(
        "--output-folder", type=str, help="Path to save the output results."
    )
    parser.add_argument(
        "--normalization-type",
        type=str,
        required=True,
        help="None or unity_independent. See img_pair_normalization in utils.",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
    parser.add_argument(
        "--input-img-type", type=str, default="dicom", help="dicom, raw, tif, etc."
    )
    parser.add_argument(
        "--specific-epoch",
        action="store_true",
        help=(
            "Apply only one checkpoint selected by --chckpt-no. Otherwise, apply "
            "all checkpoints saved during training."
        ),
    )
    parser.add_argument(
        "--chckpt-no",
        type=int,
        default=-1,
        help=(
            "Epoch number of the checkpoint to load and apply. Default is the "
            "last checkpoint."
        ),
    )
    parser.add_argument(
        "--se-plot",
        action="store_true",
        help=(
            "When using a specific epoch, save denoised images inside the output "
            "folder. Otherwise, save only test statistics."
        ),
    )
    parser.add_argument(
        "--in-dtype",
        type=str,
        default="uint16",
        help="Input image data type. Only needed for .raw images.",
    )
    parser.add_argument(
        "--resolve-patient",
        action="store_true",
        help="Save outputs with patient-specific tags/directories.",
    )
    parser.add_argument(
        "--resolve-nps", action="store_true", help="Apply CNN to water phantom images."
    )
    parser.add_argument(
        "--rNx",
        type=int,
        default=None,
        help="Raw LR image size. Required for raw input images.",
    )
    parser.add_argument("--scale", type=int, default=1, help="Up-scaling factor.")
    parser.add_argument(
        "--quant-fname-tag",
        type=str,
        default=None,
        help="Additional tag to differentiate quantitative output file names.",
    )
    parser.add_argument(
        "--hd-thres",
        type=str,
        default="10p_cutoff",
        help="Histogram thresholding before Hellinger distance calculation.",
    )
    return parser.parse_args()


def print_args(args: argparse.Namespace) -> None:
    print("\n----------------------------------------")
    print("Command line arguments")
    print("----------------------------------------")
    for key, value in vars(args).items():
        print(key, ":", value)
    print("----------------------------------------\n")


def require_output_folder(args: argparse.Namespace) -> Path:
    if not args.output_folder:
        sys.exit("ERROR! --output-folder is required for saving results.\n")
    return Path(args.output_folder)


def build_model(model_name: str, scale: int):
    if model_name == "srgan":
        from models.gan import Generator

        return Generator(
            n_residual_blocks=16,
            upsample_factor=scale,
            base_filter=64,
            num_channel=NUM_CHANNELS,
        )

    if model_name == "srwgan":
        from models.wgan import RRDBNet

        return RRDBNet(in_nc=NUM_CHANNELS, out_nc=NUM_CHANNELS, nf=32, nb=4)

    if model_name == "fsrcnn":
        from models.fsrcnn import FSRCNN

        return FSRCNN(num_channels=NUM_CHANNELS, upscale_factor=scale)

    sys.exit("ERROR! Re-check DNN model (architecture) string!\n")


def checkpoint_paths(args: argparse.Namespace) -> List[str]:
    pattern = "checkpoint-gene*.*" if args.model_name in {"srgan", "srwgan"} else "*.*"
    paths = natsort.natsorted(glob.glob(os.path.join(args.model_folder, pattern)))

    if not paths:
        sys.exit("ERROR ! No checkpoints or incorrect model path.\n")

    if args.specific_epoch:
        checkpoint_index = args.chckpt_no - 1 if args.chckpt_no != -1 else -1
        paths = [paths[checkpoint_index]]

    return paths


def prepare_output_paths(
    args: argparse.Namespace, checkpoints: List[str]
) -> Tuple[Optional[Path], Optional[Path]]:
    """Return image output directory and quantitative results file path."""

    output_folder = require_output_folder(args)
    gt_available = bool(args.gt_folder.strip())
    quant_tag = args.quant_fname_tag or ""

    if args.specific_epoch:
        checkpoint_stem = (Path(checkpoints[0]).stem)[:-4]
        image_output_dir = output_folder / checkpoint_stem
        image_output_dir.mkdir(parents=True, exist_ok=True)

        quant_path = None
        if gt_available:
            quant_path = output_folder / (
                f"{checkpoint_stem}_{args.hd_thres}_thres_in_HellDist"
                f"{quant_tag}quant_vals.txt"
            )
        return image_output_dir, quant_path

    output_folder.mkdir(parents=True, exist_ok=True)
    quant_path = output_folder / f"{quant_tag}all_checkpoint_quant_vals.txt" if gt_available else None
    return None, quant_path


def image_paths(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.*")))


def read_image(path: str, args: argparse.Namespace, *, is_gt: bool = False) -> np.ndarray:
    if args.input_img_type == "dicom":
        return io_func.pydicom_imread(path)

    if args.input_img_type == "raw":
        if args.rNx is None:
            sys.exit("ERROR! --rNx must be set when --input-img-type raw is used.\n")
        size = int(args.rNx * args.scale) if is_gt else args.rNx
        return io_func.raw_imread(path, (size, size), dtype=args.in_dtype)

    return io_func.imageio_imread(path)


def ssim_2d(reference: np.ndarray, estimate: np.ndarray, data_range: float) -> float:
    height, width = estimate.shape
    return compare_ssim(
        estimate.reshape(height, width, 1),
        reference.reshape(height, width, 1),
        multichannel=True,
        data_range=data_range,
    )


def compute_metrics(
    gt_img: np.ndarray, estimate: np.ndarray, hd_threshold: str
) -> Tuple[float, float, float, float]:
    estimate_max = max(np.max(gt_img), np.max(estimate))
    estimate_min = min(np.min(gt_img), np.min(estimate))
    return (
        quant_util.relative_se(gt_img, estimate),
        quant_util.psnr(gt_img, estimate, estimate_max),
        ssim_2d(gt_img, estimate, estimate_max - estimate_min),
        quant_util.hellinger_distance(
            gt_img, estimate, thresholding=hd_threshold, n_bins=4096, img_range=(0, 4096)
        ),
    )


def print_per_image_metrics(
    image_name: str,
    cnn_metrics: Tuple[float, float, float, float],
    lr_metrics: Tuple[float, float, float, float],
) -> None:
    print(
        "IMG: %s || avg CNN [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f, HD: %.3f] "
        "|| avg BC [rMSE: %.4f, PSNR: %.4f, SSIM: %.4f, HD: %.3f]]"
        % (image_name, *cnn_metrics, *lr_metrics)
    )


def print_no_gt_stats(image_name: str, output: np.ndarray, lr_img: np.ndarray) -> None:
    print(
        "IMG: %s || OUT [min: %.4f, max: %.4f, img_type: %s] "
        "||  IN [min: %.4f, max: %.4f, img_type: %s]"
        % (
            image_name,
            np.min(output),
            np.max(output),
            output.dtype,
            np.min(lr_img),
            np.max(lr_img),
            lr_img.dtype,
        )
    )


def maybe_pad_for_scale_three(
    args: argparse.Namespace,
    cnn_output: np.ndarray,
    lr_img: np.ndarray,
    gt_img: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if args.scale != 3:
        return cnn_output, lr_img, gt_img

    pad = ((1, 1), (1, 1))
    cnn_output = np.pad(cnn_output, pad)
    lr_img = np.pad(lr_img, pad)
    gt_img = np.pad(gt_img, pad) if gt_img is not None else None
    return cnn_output, lr_img, gt_img


def save_specific_epoch_outputs(
    args: argparse.Namespace,
    output_dir: Path,
    image_path: str,
    image_no: str,
    cnn_output: np.ndarray,
    lr_img: np.ndarray,
    gt_img: Optional[np.ndarray],
) -> None:
    cnn_output, lr_img, gt_img = maybe_pad_for_scale_three(args, cnn_output, lr_img, gt_img)

    if args.resolve_patient:
        patient_tag = Path(image_path).parents[1].name
        patient_dir_cnn = output_dir / f"{patient_tag}_cnn"
        patient_dir_bc = output_dir / f"{patient_tag}_bc"
        patient_dir_gt = output_dir / f"{patient_tag}_gt"

        patient_dir_cnn.mkdir(parents=True, exist_ok=True)
        patient_dir_bc.mkdir(parents=True, exist_ok=True)
        patient_dir_gt.mkdir(parents=True, exist_ok=True)

        io_func.imsave_raw(cnn_output, str(patient_dir_cnn / f"{image_no}.raw"))
        io_func.imsave_raw(lr_img, str(patient_dir_bc / f"{image_no}.raw"))
        if gt_img is not None:
            io_func.imsave_raw(gt_img, str(patient_dir_gt / f"{image_no}.raw"))
        return

    io_func.imsave_raw(cnn_output, str(output_dir / f"{image_no}.raw"))


def checkpoint_number(checkpoint_path: str) -> int:
    return int((Path(checkpoint_path).stem.split("-")[-1]).split('.')[0])


def summarize_metrics(checkpoint_path: str, metrics: MetricStore, quant_file) -> None:
    checkpoint_name = (Path(checkpoint_path).stem)[:-4]
    print("\n------------------------------------------------")
    print("%s (applied on test data)" % checkpoint_name)
    print("--------------------------------------------------")

    values = (
        np.mean(metrics.cnn_rmse),
        np.std(metrics.cnn_rmse),
        np.mean(metrics.cnn_psnr),
        np.std(metrics.cnn_psnr),
        np.mean(metrics.cnn_ssim),
        np.std(metrics.cnn_ssim),
        np.mean(metrics.cnn_hd),
        np.std(metrics.cnn_hd),
        np.mean(metrics.lr_rmse),
        np.std(metrics.lr_rmse),
        np.mean(metrics.lr_psnr),
        np.std(metrics.lr_psnr),
        np.mean(metrics.lr_ssim),
        np.std(metrics.lr_ssim),
        np.mean(metrics.lr_hd),
        np.std(metrics.lr_hd),
    )

    print(
        "avg CNN (std) [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), "
        "SSIM: %.4f (%.4f), HD: %.3f (%.3f)] \n"
        "avg BC  (std) [rMSE: %.4f (%.4f), PSNR: %.4f (%.4f), "
        "SSIM: %.4f (%.4f), HD: %.3f (%.3f)]" % values
    )

    quant_file.write(
        "%9d,%9.4f,%9.4f,%14.4f,%9.4f,%9.4f,%9.4f,%7.3f,%9.3f,"
        "%8.4f,%9.4f,%13.4f,%9.4f,%8.4f,%9.4f,%6.3f,%9.3f\n"
        % (checkpoint_number(checkpoint_path), *values)
    )


def apply_checkpoint(
    args: argparse.Namespace,
    checkpoint_path: str,
    base_model,
    dir_min: float,
    output_dir: Optional[Path],
) -> Optional[MetricStore]:
    cuda = args.cuda
    gt_available = bool(args.gt_folder.strip())
    output_dtype = args.in_dtype

    model = base_model.eval()
    if cuda:
        model = model.cuda()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model"])

    lr_image_paths = image_paths(args.input_folder)
    gt_image_paths = image_paths(args.gt_folder) if gt_available else []
    metrics = MetricStore() if gt_available else None

    for index, lr_path in enumerate(lr_image_paths):
        lr_input = read_image(lr_path, args)
        gt_img = read_image(gt_image_paths[index], args, is_gt=True) if gt_available else None

        cnn_output = util.norm_n_apply_model_n_renorm(
            model,
            lr_input,
            dir_min,
            args.normalization_type,
            cuda,
            args.resolve_nps,
        ).astype(output_dtype)

        lr_img = util.interpolation_hr(lr_input, args.scale).astype(output_dtype)
        image_name = Path(lr_path).name
        image_no = Path(lr_path).stem

        if index == 0:
            print("----------------------------------------")
            print("Per image stats:")
            print("----------------------------------------\n")

        if gt_available and gt_img is not None and metrics is not None:
            gt_img = gt_img.astype(output_dtype)
            cnn_metrics = compute_metrics(gt_img, cnn_output, args.hd_thres)
            lr_metrics = compute_metrics(gt_img, lr_img, args.hd_thres)
            metrics.append_cnn(*cnn_metrics)
            metrics.append_lr(*lr_metrics)
            print_per_image_metrics(image_name, cnn_metrics, lr_metrics)
        else:
            print_no_gt_stats(image_name, cnn_output, lr_img)

        if args.specific_epoch and args.se_plot and output_dir is not None:
            save_specific_epoch_outputs(
                args,
                output_dir,
                lr_path,
                image_no,
                cnn_output,
                lr_img,
                gt_img,
            )

    return metrics


def main() -> None:
    args = parse_args()
    print_args(args)

    checkpoints = checkpoint_paths(args)
    output_dir, quant_path = prepare_output_paths(args, checkpoints)

    quant_file = None
    if quant_path is not None:
        quant_file = open(quant_path, "+w")
        quant_file.write(QUANT_HEADER)

    dir_min, dir_max = util.min_max_4rmdir(
        args.input_folder, args.input_img_type, args.in_dtype, rN=args.rNx
    )
    print("[min, max] of images in the input folder:[%.4f, %.4f]\n" %(dir_min, dir_max))

    base_model = build_model(args.model_name, args.scale)

    try:
        for checkpoint_path in checkpoints:
            metrics = apply_checkpoint(args, checkpoint_path, base_model, dir_min, output_dir)
            if metrics is not None and quant_file is not None:
                summarize_metrics(checkpoint_path, metrics, quant_file)
    finally:
        if quant_file is not None:
            quant_file.close()


if __name__ == "__main__":
    main()
