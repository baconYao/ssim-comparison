import unittest
import numpy as np
from unittest.mock import patch, mock_open

from opcv_psnr import (
    psnr_args,
    _get_psnr,
    get_average_psnr,
    _get_frame_resolution,
)


class TestPsnrArgs(unittest.TestCase):
    def test_psnr_args(self):
        parser = psnr_args()
        args = parser.parse_args(["ref_file.mp4", "test_file.mp4"])
        self.assertEqual(args.reference_file, "ref_file.mp4")
        self.assertEqual(args.test_file, "test_file.mp4")
        self.assertFalse(args.show_psnr_each_frame)


class TestGetFrameResolution(unittest.TestCase):
    @patch("cv2.VideoCapture")
    def test_get_frame_resolution(self, mock_videocapture):
        mock_capture = mock_videocapture.return_value
        mock_capture.get.side_effect = [100, 200]
        width, height = _get_frame_resolution(mock_capture)
        self.assertEqual(width, 100)
        self.assertEqual(height, 200)


class TestPSNRCalculation(unittest.TestCase):
    def setUp(self):
        # Create two dummy files whose height and width is 100
        self.reference_frame = np.random.randint(
            0, 256, size=(100, 100, 3), dtype=np.uint8
        )
        self.test_frame = np.random.randint(
            128, 256, size=(100, 100, 3), dtype=np.uint8
        )

    def test_get_psnr(self):
        psnr = _get_psnr(self.reference_frame, self.test_frame)
        self.assertIsInstance(psnr, float)

    # FIXME
    @patch("opcv_psnr._get_frame_resolution")
    @patch("cv2.VideoCapture")
    def test_get_average_psnr(
        self, mock_videocapture, mock_get_frame_resolution
    ):
        # mock two files have same dimensions
        mock_get_frame_resolution.side_effect = [(100, 100), (100, 100)]

        mock_capture_ref = mock_videocapture.return_value
        mock_capture_ref.isOpened.return_value = True
        mock_capture_ref.isOpened.return_value = True
        # mock the total_frame_count as 10
        mock_capture_ref.get.side_effect = [10]
        mock_capture_ref.read.side_effect = [(True, self.reference_frame)] * 10

        mock_capture_test = mock_videocapture.return_value
        mock_capture_test.isOpened.return_value = True
        mock_capture_test.read.side_effect = [(True, self.test_frame)] * 10

        avg_psnr, psnr_each_frame = get_average_psnr(
            "ref_file.mp4", "test_file.mp4"
        )
        self.assertIsInstance(avg_psnr, float)
        self.assertIsInstance(psnr_each_frame, list)
        self.assertEqual(len(psnr_each_frame), 10)

        # Compute the expected average PSNR
        expected_avg_psnr = (
            sum(
                _get_psnr(self.reference_frame, self.test_frame)
                for _ in range(10)
            )
            / 10
        )
        self.assertAlmostEqual(avg_psnr, expected_avg_psnr, places=5)

    def test_get_average_psnr_file_not_found(self):
        with patch("cv2.VideoCapture") as mock_videocapture:
            mock_capture = mock_videocapture.return_value
            mock_capture.isOpened.return_value = False
            with self.assertRaises(SystemExit):
                get_average_psnr("nonexistent_file.mp4", "test_file.mp4")

    @patch("opcv_psnr._get_frame_resolution")
    @patch("cv2.VideoCapture")
    def test_get_average_psnr_different_dimensions(
        self, mock_videocapture, mock_get_frame_resolution
    ):
        mock_capture_ref = mock_videocapture.return_value
        mock_capture_test = mock_videocapture.return_value
        mock_capture_test.isOpened.return_value = True
        mock_capture_ref.isOpened.return_value = True
        mock_get_frame_resolution.side_effect = [(100, 100), (100, 150)]

        with self.assertRaises(SystemExit) as cm:
            get_average_psnr("ref_file.mp4", "test_file.mp4")

        self.assertEqual(
            str(cm.exception), "Error: Files have different dimensions."
        )


if __name__ == "__main__":
    unittest.main()
