import unittest
from unittest.mock import MagicMock, patch
from contextlib import redirect_stdout
import io
import numpy as np

from horizoncv import horizon, plotter
import run_demo

class HorizonTest(unittest.TestCase):

    def test_split_img_by_line_simple(self):
        img = np.ones((8, 8, 3))
        m = 0.0
        b = 4.0
        seg1, seg2 = horizon.split_img_by_line(img, m, b)
        self.assertEqual((8 * 8 / 2, 3), seg1.shape)
        self.assertEqual((8 * 8 / 2, 3), seg2.shape)

    def test_split_img_by_line_diagonal(self):
        img = np.ones((8, 8, 3))
        m = 1.0
        b = 0.0
        seg1, seg2 = horizon.split_img_by_line(img, m, b)
        self.assertEqual((8 * 8 / 2 + 4, 3), seg1.shape)
        self.assertEqual((8 * 8 / 2 - 4, 3), seg2.shape)

    def test_horizon_altitude_level_camera_sign(self):
        result = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 54.0, 60.0, 0.0,
            horizon_validated=True, calibration_validated=True)
        self.assertAlmostEqual(0.0, result['alpha_deg'], places=10)
        self.assertAlmostEqual(0.0, result['height_m'], places=8)

    def test_pixel_offset_sign_above_and_below_image_centre(self):
        below = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 60.0, 0.0)
        above = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 44.0, 60.0, 0.0)
        self.assertGreater(below['pixel_offset'], 0.0)
        self.assertGreater(below['delta_deg'], 0.0)
        self.assertLess(above['pixel_offset'], 0.0)
        self.assertLess(above['delta_deg'], 0.0)

    def test_resize_transform_preserves_line_geometry(self):
        source_shape = (14, 26, 3)
        target_shape = (720, 1280, 3)
        m, b = 0.25, 5.0
        scaled_m, scaled_b = horizon.scale_horizon_line(
            m, b, source_shape, target_shape)
        scale_x = target_shape[1] / source_shape[1]
        scale_y = target_shape[0] / source_shape[0]
        for x_source in (0.0, 10.0, 25.0):
            x_target = x_source * scale_x
            y_target_expected = (m * x_source + b) * scale_y
            y_target_actual = scaled_m * x_target + scaled_b
            self.assertAlmostEqual(y_target_expected, y_target_actual, places=10)
        source_center_y = horizon.horizon_line_geometry(
            source_shape, m, b)['y_center']
        target_center_y = horizon.horizon_line_geometry(
            target_shape, scaled_m, scaled_b)['y_center']
        self.assertAlmostEqual(
            source_center_y * (target_shape[0] / source_shape[0]),
            target_center_y,
            places=10,
        )

    def test_horizon_altitude_compensates_positive_and_negative_camera_pitch(self):
        image_height = 108
        vfov_deg = 60.0
        fy = (image_height / 2.0) / np.tan(np.radians(vfov_deg / 2.0))
        for camera_pitch_deg in (5.0, -5.0):
            y_h = image_height / 2.0 + fy * np.tan(np.radians(camera_pitch_deg))
            result = horizon.estimate_horizon_altitude(
                (image_height, 192, 3), 0.0, y_h, vfov_deg,
                camera_pitch_deg, horizon_validated=True,
                calibration_validated=True)
            self.assertAlmostEqual(0.0, result['alpha_deg'], places=10)
            self.assertAlmostEqual(0.0, result['height_m'], places=8)

    def test_validated_altitude_uses_spherical_earth_formula(self):
        image_height = 108
        vfov_deg = 60.0
        alpha_deg = 5.0
        fy = (image_height / 2.0) / np.tan(np.radians(vfov_deg / 2.0))
        y_h = image_height / 2.0 + fy * np.tan(np.radians(alpha_deg))
        result = horizon.estimate_horizon_altitude(
            (image_height, 192, 3), 0.0, y_h, vfov_deg, 0.0,
            horizon_validated=True, calibration_validated=True)
        expected_height = horizon.EARTH_RADIUS_M * (
            1.0 / np.cos(np.radians(alpha_deg)) - 1.0)
        self.assertAlmostEqual(alpha_deg, result['alpha_deg'], places=10)
        self.assertAlmostEqual(expected_height, result['height_m'], places=8)

    def test_horizon_altitude_suppressed_without_validation_or_calibration(self):
        unvalidated = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 60.0, 0.0, horizon_validated=False)
        uncalibrated = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 60.0, 0.0,
            horizon_validated=True, calibration_validated=False)
        unknown_fov = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, None, 0.0, horizon_validated=True)
        unknown_pitch = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 60.0, None, horizon_validated=True)
        self.assertIsNone(unvalidated['height_m'])
        self.assertIsNone(uncalibrated['height_m'])
        self.assertIsNone(unknown_fov['height_m'])
        self.assertIsNone(unknown_pitch['height_m'])
        self.assertFalse(unvalidated['valid'])
        self.assertFalse(uncalibrated['valid'])
        self.assertFalse(unknown_fov['valid'])
        self.assertFalse(unknown_pitch['valid'])

    def test_demo_only_estimate_is_separate_from_validated_altitude(self):
        demo = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0,
            horizon_validated=False, calibration_validated=False,
            demo_only=True)
        safe_default = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0,
            horizon_validated=False, calibration_validated=False)
        self.assertIsNotNone(demo['simulated_height_m'])
        self.assertIsNone(demo['height_m'])
        self.assertFalse(demo['valid'])
        self.assertFalse(demo['horizon_validated'])
        self.assertFalse(demo['calibration_validated'])
        self.assertIsNone(safe_default['simulated_height_m'])
        self.assertIsNone(safe_default['height_m'])
        self.assertFalse(safe_default['demo_only'])

    def test_synthetic_demo_generates_configured_positive_depression(self):
        from horizoncv import demo

        analysis = demo.synthetic_altitude_demo(is_display=False)
        self.assertAlmostEqual(
            demo.SYNTHETIC_DEPRESSION_DEG, analysis['alpha_deg'], places=10)
        self.assertGreater(analysis['simulated_height_m'], 0.0)
        self.assertIsNone(analysis['height_m'])
        self.assertFalse(analysis['horizon_validated'])
        self.assertFalse(analysis['calibration_validated'])

    def test_overlay_distinguishes_real_video_validation_states(self):
        uncalibrated = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0)
        calibration_only = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0,
            calibration_validated=True, horizon_validated=False)
        validated = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0,
            calibration_validated=True, horizon_validated=True)
        self.assertTrue(plotter._altitude_status_label(uncalibrated).startswith(
            'REAL VIDEO - UNCALIBRATED ASSUMPTIONS: N/A'))
        self.assertTrue(plotter._altitude_status_label(calibration_only).startswith(
            'REAL VIDEO - HORIZON NOT VALIDATED: N/A'))
        self.assertTrue(plotter._altitude_status_label(validated).startswith(
            'REAL VIDEO ALTITUDE (VALIDATED): '))

    def test_demo_only_rejects_invalid_depression_angles(self):
        above_horizon = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 40.0, 45.0, 0.0, demo_only=True)
        excessive_depression = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 107.0, 179.0, -5.0, demo_only=True)
        for result in (above_horizon, excessive_depression):
            self.assertIsNone(result['simulated_height_m'])
            self.assertIsNone(result['height_m'])
            self.assertFalse(result['valid'])
            self.assertEqual(
                'horizon depression angle is outside the valid range',
                result['reason'])

    def test_overlay_explicitly_labels_demo_value_as_simulated(self):
        analysis = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0, demo_only=True)
        labels = []
        with patch('horizoncv.plotter.cv2.putText',
                   side_effect=lambda image, text, *args: labels.append(text) or image):
            plotter.add_line_to_frame(
                np.zeros((108, 192, 3), dtype=np.uint8),
                0.0, 64.0, 0.0, 0.0, analysis)
        self.assertTrue(any(
            label.startswith('SIMULATED ALTITUDE - UNCALIBRATED: ')
            and label.endswith(' m')
            for label in labels))

    def test_overlay_preserves_bgr_frame_colors(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[10, 80] = (255, 0, 0)

        rendered = plotter.add_line_to_frame(
            image, 0.0, 70.0, 0.5, 0.0)

        np.testing.assert_array_equal(rendered[10, 80], (255, 0, 0))
        np.testing.assert_array_equal(image[10, 80], (255, 0, 0))

    def test_roll_advisory_reports_the_crossed_threshold_band(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        messages = []
        with patch('horizoncv.plotter.cv2.rectangle') as rectangle, \
                patch('horizoncv.plotter.cv2.putText',
                      side_effect=lambda img, text, *args: messages.append(text) or img):
            plotter.add_attitude_limit_warning(
                image,
                roll_deg=-32.0,
                roll_thresholds=((20.0, 'CAUTION'), (30.0, 'WARNING')))

        rectangle.assert_called_once()
        self.assertEqual(
            ['ADVISORY - ROLL WARNING: -32.0 deg (>30.0 deg)'], messages)

    def test_roll_advisory_uses_caution_range_and_no_alert_below_range(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        messages = []
        with patch('horizoncv.plotter.cv2.rectangle') as rectangle, \
                patch('horizoncv.plotter.cv2.putText',
                      side_effect=lambda img, text, *args: messages.append(text) or img):
            result = plotter.add_attitude_limit_warning(
                image,
                roll_deg=25.0,
                roll_thresholds=((20.0, 'CAUTION'), (30.0, 'WARNING')))

        self.assertIs(result, image)
        rectangle.assert_called_once()
        self.assertEqual(
            ['ADVISORY - ROLL CAUTION: 25.0 deg (>20.0 deg)'], messages)

        below = np.zeros((100, 200, 3), dtype=np.uint8)
        with patch('horizoncv.plotter.cv2.rectangle') as below_rectangle:
            plotter.add_attitude_limit_warning(
                below,
                roll_deg=19.0,
                roll_thresholds=((20.0, 'CAUTION'), (30.0, 'WARNING')))
        below_rectangle.assert_not_called()

    def test_attitude_warning_identifies_unavailable_line_estimate(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        messages = []
        with patch('horizoncv.plotter.cv2.rectangle'), \
                patch('horizoncv.plotter.cv2.putText',
                      side_effect=lambda img, text, *args: messages.append(text) or img):
            plotter.add_attitude_limit_warning(
                image, roll_deg=0.0,
                roll_thresholds=((20.0, 'CAUTION'), (30.0, 'WARNING')),
                attitude_valid=False)

        self.assertEqual(
            ['ADVISORY - ATTITUDE ESTIMATE INVALID - CHECK INDEPENDENT INSTRUMENTS'],
            messages)

    def test_attitude_warning_supports_optional_pitch_bands(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        messages = []
        with patch('horizoncv.plotter.cv2.rectangle'), \
                patch('horizoncv.plotter.cv2.putText',
                      side_effect=lambda img, text, *args: messages.append(text) or img):
            plotter.add_attitude_limit_warning(
                image,
                roll_deg=0.0,
                roll_thresholds=((20.0, 'CAUTION'), (30.0, 'WARNING')),
                pitch_deg=16.0,
                pitch_thresholds=((10.0, 'CAUTION'), (15.0, 'WARNING')))

        self.assertEqual(
            ['ADVISORY - PITCH WARNING: 16.0 deg (>15.0 deg)'], messages)

    def test_score_grid_discards_off_frame_and_excessive_slope_candidates(self):
        image = np.random.default_rng(1).integers(
            0, 256, size=(20, 40, 3), dtype=np.uint8)
        grid = [(0.0, 10.0), (0.0, 30.0), (2.0, 10.0), (0.0, 25.0)]

        m, b, scores, valid_grid, _ = horizon.score_grid(image, grid)

        self.assertTrue(all(abs(candidate_m) <= horizon.MAX_ABS_HORIZON_SLOPE
                             for candidate_m, _ in valid_grid))
        self.assertTrue(all(0 <= candidate_m * 20 + candidate_b < 20
                            for candidate_m, candidate_b in valid_grid))
        self.assertEqual(len(valid_grid), len(scores))
        self.assertIn((m, b), valid_grid)

    def test_real_time_search_resets_global_for_out_of_frame_prior(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        result = ((0.0, 10.0), [], [], 0.5, 0.0)
        with patch.object(horizon, 'optimize_global', return_value=result) as global_search, \
                patch.object(horizon, 'optimize_local') as local_search:
            actual = horizon.optimize_real_time(image, image, 0.0, 100.0, 1.0)

        self.assertIs(actual, result)
        global_search.assert_called_once_with(image, image, 1.0)
        local_search.assert_not_called()

    def test_real_video_demo_only_overlay_uses_experimental_label(self):
        analysis = horizon.estimate_horizon_altitude(
            (108, 192, 3), 0.0, 64.0, 45.0, 0.0,
            demo_only=True, demo_source='real_video', horizon_validated=False)
        label = plotter._altitude_status_label(analysis)
        self.assertTrue(label.startswith('EXPERIMENTAL ALTITUDE: '))
        self.assertIn('UNCALIBRATED DEMO', label)
        self.assertIn(' m | UNCALIBRATED DEMO', label)

    def test_video_demo_passes_experimental_mode_without_changing_validation(self):
        from horizoncv import demo

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with patch.object(demo, 'DEMO_ONLY_ALTITUDE', True), \
                patch.object(demo, 'CAMERA_CALIBRATION_VALIDATED', False), \
                patch.object(demo, 'HORIZON_GEOMETRY_VALIDATED', False), \
                patch.object(demo, 'getVideoSource', return_value=[frame]), \
                patch.object(demo, 'optimize_real_time',
                             return_value=((0.0, 8.0), [], [], 0.5, 0.0)), \
                patch.object(demo.cv2, 'waitKey', return_value=-1), \
                patch.object(demo.cv2, 'destroyAllWindows'):
            with patch.object(
                    demo, 'estimate_horizon_altitude',
                    wraps=demo.estimate_horizon_altitude) as estimator:
                demo.video_demo('unused.mp4', is_display=False, n_frames=1)

        self.assertTrue(estimator.call_args.kwargs['demo_only'])
        self.assertEqual('real_video', estimator.call_args.kwargs['demo_source'])
        self.assertFalse(estimator.call_args.kwargs['calibration_validated'])
        self.assertFalse(estimator.call_args.kwargs['horizon_validated'])
        self.assertFalse(demo.CAMERA_CALIBRATION_VALIDATED)
        self.assertFalse(demo.HORIZON_GEOMETRY_VALIDATED)

    def test_video_demo_displays_actual_frame_and_closes_windows(self):
        from horizoncv import demo

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        shown = []
        output = io.StringIO()
        with patch.object(demo, 'getVideoSource', return_value=[frame]), \
                patch.object(demo, 'optimize_real_time',
                             return_value=((0.0, 8.0), [], [], 0.5, 0.0)), \
                patch.object(demo.cv2, 'imshow',
                             side_effect=lambda title, image: shown.append((title, image.shape))), \
                patch.object(demo.cv2, 'waitKey', return_value=ord('q')) as wait_key, \
                patch.object(demo.cv2, 'destroyAllWindows') as destroy:
            with redirect_stdout(output):
                demo.video_demo('turn1.mp4', is_display=True, n_frames=1)

        self.assertEqual([('frame', frame.shape)], shown)
        wait_key.assert_called_once_with(1)
        destroy.assert_called_once_with()
        self.assertIn('Video ended (1 frames processed)', output.getvalue())

    def test_video_demo_keeps_low_resolution_analysis_tall_enough(self):
        from horizoncv import demo

        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        optimized_shapes = []

        def record_analysis_shape(img, highres, m, b, scaling_factor):
            optimized_shapes.append(img.shape)
            return ((0.0, img.shape[0] / 2.0), [], [], 0.5, 0.0)

        with patch.object(demo, 'getVideoSource', return_value=[frame]), \
                patch.object(demo, 'optimize_real_time',
                             side_effect=record_analysis_shape), \
                patch.object(demo.cv2, 'destroyAllWindows'):
            demo.video_demo('low-resolution.mp4', is_display=False, n_frames=1)

        self.assertEqual(1, len(optimized_shapes))
        self.assertGreaterEqual(
            optimized_shapes[0][0], demo.MIN_ANALYSIS_HEIGHT)

    def test_video_demo_upscales_low_resolution_frame_for_display(self):
        from horizoncv import demo

        frame = np.zeros((144, 256, 3), dtype=np.uint8)
        shown = []
        with patch.object(demo, 'getVideoSource', return_value=[frame]), \
                patch.object(demo, 'optimize_real_time',
                             return_value=((0.0, 8.0), [], [], 0.5, 0.0)), \
                patch.object(demo.cv2, 'imshow',
                             side_effect=lambda title, image: shown.append(image.shape)), \
                patch.object(demo.cv2, 'waitKey', return_value=ord('q')), \
                patch.object(demo.cv2, 'destroyAllWindows'):
            demo.video_demo('low-resolution.mp4', is_display=True, n_frames=1)

        self.assertEqual([(720, 1280, 3)], shown)

    def test_video_source_fails_clearly_when_capture_cannot_open(self):
        from horizoncv import demo

        capture = MagicMock()
        capture.isOpened.return_value = False
        with patch.object(demo.cv2, 'VideoCapture', return_value=capture):
            with self.assertRaisesRegex(FileNotFoundError, 'Unable to open video'):
                next(demo.getVideoSource('missing.mp4'))
        capture.release.assert_called_once_with()

    def test_video_only_entry_point_skips_synthetic_preview(self):
        with patch.object(run_demo.demo, 'synthetic_altitude_demo') as synthetic, \
                patch.object(run_demo.demo, 'video_demo') as video:
            output = io.StringIO()
            with redirect_stdout(output):
                run_demo.main(['--video-only'])
        synthetic.assert_not_called()
        video.assert_called_once_with('rocket_camera_feed.webm')
        self.assertIn('Starting real video', output.getvalue())

    def test_default_entry_point_shows_timed_synthetic_then_video(self):
        with patch.object(run_demo.demo, 'synthetic_altitude_demo') as synthetic, \
                patch.object(run_demo.demo, 'video_demo') as video:
            output = io.StringIO()
            with redirect_stdout(output):
                run_demo.main([])
        synthetic.assert_called_once_with(display_ms=3000)
        video.assert_called_once_with('rocket_camera_feed.webm')
        self.assertLess(
            output.getvalue().index('Starting synthetic demo'),
            output.getvalue().index('Starting real video'))

    def test_img_line_mask(self):
        rows = 15
        columns = 24
        slope_range = list(np.arange(-4, 4, 0.25))
        intercept_range = list(np.arange( 1, rows - 2, 0.5))

        # TODO
        # for m, b in zip(slope_range, intercept_range):
        #     print(m, b)
        #     mask1 = horizon.img_line_mask(rows, columns, m, b)
        #     mask2 = horizon.img_line_mask2(rows, columns, m, b)
        #     self.assertTrue(np.array_equal(mask1, mask2), str(mask1) + '2' + str(mask2))