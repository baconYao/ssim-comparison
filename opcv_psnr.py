# https://docs.opencv.org/3.4/d5/dc4/tutorial_video_input_psnr_ssim.html

import cv2
import numpy as np
import sys
import argparse
from typing import Tuple, List


def psnr_args():
    """
    Parse command-line arguments using argparse.

    Returns:
        Tuple[str, str]: A tuple containing the reference file path and test file path.
    """
    parser = argparse.ArgumentParser(description="Calculate PSNR between two files. File can be image or video.")
    parser.add_argument("reference_file", type=str, help="Path to the reference file")
    parser.add_argument("test_file", type=str, help="Path to the test file")
    return parser


def get_psnr(I1: np.ndarray, I2: np.ndarray) -> float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        I1 (np.ndarray): Reference image.
        I2 (np.ndarray): Image to be compared with the reference.

    Returns:
        float: PSNR value indicating the quality of I2 compared to I1.
    """
    # Calculate the absolute difference
    s1 = cv2.absdiff(I1, I2)
    # cannot make a square on 8 bits
    s1 = np.float32(s1)
    # Calculate squared differences
    s1 = s1 * s1
    # Sum of squared differences per channel
    sse = s1.sum()
    # sum channels
    if sse <= 1e-10:
        # for small values return zero
        return 0.0
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
    return psnr


def get_average_psnr(reference_file_path: str, test_file_path: str) -> Tuple[float, List[float]]:
    """
    Calculate the average PSNR and PSNR for each frame between two files.
    Files can be image or video.

    Args:
        reference_file_path (str): Path to the reference file.
        test_file_path (str): Path to the test file.

    Returns:
        Tuple[float, List[float]]: A tuple containing the average PSNR value and a list of PSNR values for each frame.
    """
    capt_refrnc = cv2.VideoCapture(reference_file_path)
    capt_undTst = cv2.VideoCapture(test_file_path)

    if not capt_refrnc.isOpened() or not capt_undTst.isOpened():
        print("Error: Could not open reference or test file.")
        sys.exit(-1)

    ref_size = (int(capt_refrnc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capt_refrnc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    test_size = (int(capt_undTst.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capt_undTst.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if ref_size != test_size:
        print("Error: Files have different dimensions.")
        sys.exit(-1)
    total_frames = int(capt_refrnc.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total_frames)
    avg_psnr = 0.0
    psnr_each_frame = []

    for _ in range(total_frames):
        _, frameReference = capt_refrnc.read()
        _, frameUnderTest = capt_undTst.read()
        psnr = get_psnr(frameReference, frameUnderTest)
        psnr_each_frame.append(psnr)
        avg_psnr += psnr

    avg_psnr /= total_frames
    return avg_psnr, psnr_each_frame


def main() -> None:
    args = psnr_args().parse_args()
    avg_psnr, psnr_each_frame = get_average_psnr(args.reference_file, args.test_file)
    print("Average PSNR:", avg_psnr)


if __name__ == "__main__":
    main()
