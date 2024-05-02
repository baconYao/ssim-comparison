import unittest
import numpy as np
from unittest.mock import patch
from argparse import Namespace
from io import StringIO

from psnr import (
    main,
    psnr_args,
    _get_psnr,
    get_average_psnr,
    _get_frame_resolution,
)


class TestPSNRArgs(unittest.TestCase):

    @patch("psnr.argparse.ArgumentParser.parse_args")
    def test_psnr_args_with_defaults(self, mock_parse_args):
        mock_parse_args.return_value = Namespace(
            reference_file="ref.mp4",
            test_file="test.mp4",
            show_psnr_each_frame=False,
        )
        args = psnr_args()

        self.assertEqual(args.reference_file, "ref.mp4")
        self.assertEqual(args.test_file, "test.mp4")
        self.assertFalse(args.show_psnr_each_frame)

    @patch("psnr.argparse.ArgumentParser.parse_args")
    def test_psnr_args_with_custom_args(self, mock_parse_args):
        mock_parse_args.return_value = Namespace(
            reference_file="ref.jpg",
            test_file="test.jpg",
            show_psnr_each_frame=True,
        )

        args = psnr_args()

        self.assertEqual(args.reference_file, "ref.jpg")
        self.assertEqual(args.test_file, "test.jpg")
        self.assertTrue(args.show_psnr_each_frame)


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
    @patch("psnr._get_frame_resolution")
    @patch("cv2.VideoCapture")
    def Notest_get_average_psnr(
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

    @patch("psnr._get_frame_resolution")
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


class TestMainFunction(unittest.TestCase):

    @patch("sys.stdout", new_callable=StringIO)
    @patch("psnr.get_average_psnr")
    @patch("psnr.argparse.ArgumentParser.parse_args")
    def test_main_prints_avg_psnr(
        self, mock_parse_args, mock_get_average_psnr, mock_stdout
    ):
        mock_parse_args.return_value = Namespace(
            reference_file="ref.mp4",
            test_file="test.mp4",
            show_psnr_each_frame=False,
        )

        mock_get_average_psnr.return_value = (30.0, [28.5, 31.2, 29.8])

        main()

        expected_output = "Average PSNR:  30.0\n"
        self.assertEqual(mock_stdout.getvalue(), expected_output)
        mock_get_average_psnr.assert_called_once_with("ref.mp4", "test.mp4")

    @patch("sys.stdout", new_callable=StringIO)
    @patch("psnr.get_average_psnr")
    @patch("psnr.argparse.ArgumentParser.parse_args")
    def test_main_prints_psnr_each_frame(
        self, mock_parse_args, mock_get_average_psnr, mock_stdout
    ):
        # Mock argument parsing
        mock_parse_args.return_value = Namespace(
            reference_file="ref_file",
            test_file="test_file",
            show_psnr_each_frame=True,
        )

        # Mock get_average_psnr function
        mock_get_average_psnr.return_value = (30.0, [28.5, 31.2, 29.8])

        # Call the main function
        main()

        # Check if the PSNR each frame is printed
        expected_output = (
            "Average PSNR:  30.0\nPSNR each frame:  [28.5, 31.2, 29.8]\n"
        )
        self.assertEqual(mock_stdout.getvalue(), expected_output)
        mock_get_average_psnr.assert_called_once_with("ref_file", "test_file")


if __name__ == "__main__":
    unittest.main()
