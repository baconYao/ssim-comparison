# ssim-comparison

This is a simple PoC to use the Structural Similarity index (SSI) algorithm to compare images and videos.

## How To Use

### Create Virtual Environment

```bash
$ python3 -m venv venv
$ source venv/bin/activate
```

### Install Dependencies

```bash
$ pip install -r requirements.txt
```

### Execute Program

```bash
$ python3 ssim.py
```

Output

```bash
 ===== Image Example 1 =====
Description: Comparing two identical images would result in no discernible differences as they are exactly the same.
File 1: ./data/jpeg/1920x1080-NV12.jpg
File 2: ./data/jpeg/1920x1080-NV12.jpg
Average SSIM: 1.0
 ===== Image Example 2 =====
Description: Comparing two images with different color formats but both with a resolution of 1920 x 1080 (NV12 vs YUY2).
File 1: ./data/jpeg/1920x1080-NV12.jpg
File 2: ./data/jpeg/1920x1080-YUY2.jpg
Average SSIM: 0.9998950622921191
 ===== Image Example 3 =====
Description: Comparing two images with the same color format but different resolutions.
File 1: ./data/jpeg/1920x1080-NV12.jpg
File 2: ./data/jpeg/3840x2160-NV12.jpg
Unable to compare two images with different resolution

 ===== Video Example 1 =====
Description: Comparing two identical videos would result in no discernible differences as they are exactly the same.
File 1: ./data/h264/640x480-NV12.mp4
File 2: ./data/h264/640x480-NV12.mp4
Average SSIM: 1.0
 ===== Video Example 2 =====
Description: Comparing two sets of videos with identical content, one of which has minor damage.
File 1: ./data/h264/640x480-NV12.mp4
File 2: ./data/h264/bad-640x480-NV12.mp4
Average SSIM: 0.9482682651870711
```

If the two files being compared are exactly the `same`, then you will get an SSIM (Structural Similarity Index) value of `1.0`
