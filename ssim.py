#!/usr/bin/env python3

import cv2
from skimage.metrics import structural_similarity as compare_ssim


def average_ssim(file1_path: str = '', file2_path: str = '') -> float:
    print('File 1: {}'.format(file1_path))
    print('File 2: {}'.format(file2_path))
    # Open the video files
    file1 = cv2.VideoCapture(file1_path)
    file2 = cv2.VideoCapture(file2_path)

    # Initialize variables to store SSIM values
    ssim_values = []

    # Loop through frames in both videos
    while True:
        # Read frames from both videos
        ret1, frame1 = file1.read()
        ret2, frame2 = file2.read()

        # Break the loop if either video reaches the end
        if not ret1 or not ret2:
            break

        # Convert frames to grayscale
        frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate SSIM between frames
        ssim = compare_ssim(frame1_gray, frame2_gray)
        ssim_values.append(ssim)

    # Release the video objects
    file1.release()
    file2.release()

    # Calculate average SSIM value
    average_ssim = sum(ssim_values) / len(ssim_values)

    # Print the result
    print("Average SSIM:", average_ssim)

    return average_ssim


print(' ===== Image Example 1 =====')
print('Description: Comparing two identical images would result in no discernible differences as they are exactly the same.')
average_ssim(
    file1_path='./data/jpeg/1920x1080-NV12.jpg',
    file2_path='./data/jpeg/1920x1080-NV12.jpg'
)

print(' ===== Image Example 2 =====')
print('Description: Comparing two images with different color formats but both with a resolution of 1920 x 1080 (NV12 vs YUY2).')
average_ssim(
    file1_path='./data/jpeg/1920x1080-NV12.jpg',
    file2_path='./data/jpeg/1920x1080-YUY2.jpg'
)

print(' ===== Image Example 3 =====')
print('Description: Comparing two images with the same color format but different resolutions.')
try:
    average_ssim(
        file1_path='./data/jpeg/1920x1080-NV12.jpg',
        file2_path='./data/jpeg/3840x2160-NV12.jpg'
    )
except Exception:
    print("Unable to compare two images with different resolution")


print()
print(' ===== Video Example 1 =====')
print('Description: Comparing two identical videos would result in no discernible differences as they are exactly the same.')
average_ssim(
    file1_path='./data/h264/640x480-NV12.mp4',
    file2_path='./data/h264/640x480-NV12.mp4'
)

print(' ===== Video Example 2 =====')
print('Description: Comparing two sets of videos with identical content, one of which has minor damage.')
average_ssim(
    file1_path='./data/h264/640x480-NV12.mp4',
    file2_path='./data/h264/bad-640x480-NV12.mp4'
)
