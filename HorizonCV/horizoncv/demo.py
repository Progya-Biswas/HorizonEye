
import cv2
import numpy as np
from pathlib import Path
from horizoncv.horizon import (
    optimize_real_time,
    optimize_global,
    convert_m_b_to_pitch_bank,
    estimate_horizon_altitude,
    horizon_line_geometry,
    scale_horizon_line,
    valid_pixel_mask,
    fit_horizon_by_color,
    get_sigma_below,
)
from . import plotter

# These numeric values are unverified assumptions, not calibration data.
# Replace them with measured values and set CAMERA_CALIBRATION_VALIDATED only
# after calibration has been checked before using validated altitude output.
# Camera pitch is positive when the optical axis points upward from local
# horizontal. The
# image y-axis points downward. Keep horizon validation false until visual or
# external checks confirm the detector is tracking the geometric horizon,
# rather than a skyline, clouds, or another visual boundary.
CAMERA_VFOV_DEG = 45.0
CAMERA_PITCH_DEG = 0.0
CAMERA_CALIBRATION_VALIDATED = False
HORIZON_GEOMETRY_VALIDATED = False
# Explicit prototype mode: calculate only the separate experimental value on
# real frames where positive depression geometry permits it. The validated
# height field and both validation flags remain untouched.
DEMO_ONLY_ALTITUDE = True
SYNTHETIC_DEPRESSION_DEG = 1.0
DIAGNOSTIC_FRAME_COUNT = 3
# Optionally detect on a grayscale copy of the frame. Tested: it matches RGB on
# clean footage and is worse on the porthole rocket video, so it is off.
USE_GRAYSCALE_DETECTION = False
# Ignore dark vignette / porthole rims / letterbox bars when scoring lines.
USE_VALID_MASK = True
# Fit the line between sky/cloud colours (blue, white, grey) and ground colours,
# and raise a horizon alert when only sky or only ground is visible.
USE_COLOR_FIT = True
MIN_ANALYSIS_HEIGHT = 48
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720
# Demonstration-only advisory bands, not UK-wide or aircraft-approved limits.
# Replace with thresholds from the specific aircraft's approved guidance.
ROLL_ADVISORY_THRESHOLDS = (
    (20.0, 'CAUTION'),
    (30.0, 'WARNING'),
    (45.0, 'HIGH'),
)
# This video-only detector does not measure aircraft pitch independently.
PITCH_ADVISORY_THRESHOLDS = None


def resize_for_display(image):
    """Scale a rendered frame to fit a 1280x720 display area."""
    height, width = image.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / width, DISPLAY_MAX_HEIGHT / height)
    if scale == 1.0:
        return image
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=interpolation,
    )


def load_img(path):
    """ Load an image located at a path.
        Args:
            path (str): relative path to img
        Returns:
            resized (np.array): resized img
            img (np.array): unaltered loaded img
    """
    print ('load img...')
    img = cv2.imread(path)
    print('Image shape: ', img.shape) # rows, columns, depth (height x width x color)
    print('Resize...')
    resized = cv2.resize(img, dsize=None, fx=0.2, fy=0.2)
    # blur = cv2.GaussianBlur(resized,(3,3),0) # blurs the horizon too much
    print('Resized shape:', resized.shape, img.shape)
    return resized, img


def getVideoSource(filename):
    """ Generator for cleanly loading a video file (will only work if OpenCV was
            compiled with FFMPEG support). Videos are 1920 x 1080 original, 960 x 540 resized
    """ 
    print('Load video {}...'.format(filename))
    video_path = Path(__file__).resolve().parent.parent / 'media' / filename
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError('Unable to open video: {}'.format(video_path))
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def optimization_surface_demo():
    """ Show the optimization surface in 2D and 3D views for a sample image. """
    #'taxi_rotate.png runway1.JPG taxi_empty.jpg ocean sunset grass
    img, full_res = load_img('./media/taxi_rotate.png')
    
    # img, full_res = load_img('../img/grass_pitch_low.jpg')  
    print('Optimize scores...')
    answer, scores, grid, pitch, bank = optimize_global(img, full_res, 1.0)
    # print('Max:', max_index)

    m = answer[0]
    b = answer[1]
    print('m:', m, '  b:', b)
    plotter.plot2D(full_res, scores, m, b*5, pitch, bank)

    X = [option[0] for option in grid] # m
    Y = [option[1] for option in grid] # b
    Z = list(map(lambda x: x * (10**8), scores))
    print(len(X), len(Y), len(Z))
    plotter.scatter3D(X, Y, Z) # scatter3D(X[::10], Y[::10], Z[::10])


def timer_demo():
    """ Time the LaneCV library to retrieve FPS (processing speed). """
    import timeit
    n_iterations = 1
    n_frames = 100
    result = timeit.timeit('demo.video_demo(filename="turn1.mp4", is_display=True, n_frames={})'.format(n_frames), 
                        setup='from horizoncv import demo;', 
                        number=n_iterations)
    seconds = result / n_iterations
    print('Timing: {} seconds for {} frames of video.'.format(seconds, n_frames))
    print('{} frames / second'.format(n_frames / seconds))


def synthetic_altitude_demo(is_display=True, display_ms=3000):
    """Show the angle-to-altitude calculation on a generated test image only.

    The synthetic horizon is constructed from the configured depression angle,
    vertical FOV, and mounting-pitch assumption. It is not derived from a video
    frame and is labelled simulated in the display.
    """
    if not DEMO_ONLY_ALTITUDE:
        print('Synthetic altitude demonstration is disabled.')
        return None

    image_height, image_width = 360, 640
    cy = image_height / 2.0
    fy = (image_height / 2.0) / np.tan(np.radians(CAMERA_VFOV_DEG / 2.0))
    image_offset_deg = SYNTHETIC_DEPRESSION_DEG + CAMERA_PITCH_DEG
    y_h = cy + fy * np.tan(np.radians(image_offset_deg))
    synthetic_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    synthetic_image[:int(round(y_h)), :] = (150, 90, 35)
    synthetic_image[int(round(y_h)):, :] = (45, 105, 45)

    analysis = estimate_horizon_altitude(
        synthetic_image.shape,
        0.0,
        y_h,
        camera_vfov_deg=CAMERA_VFOV_DEG,
        camera_pitch_deg=CAMERA_PITCH_DEG,
        horizon_validated=False,
        calibration_validated=False,
        demo_only=True,
    )
    print(
        'Synthetic demonstration (not video): configured depression={:.2f} deg, '
        'computed alpha={} deg, simulated_height={} m'.format(
            SYNTHETIC_DEPRESSION_DEG,
            analysis['alpha_deg'],
            analysis['simulated_height_m'],
        )
    )
    if is_display:
        image = plotter.add_line_to_frame(
            synthetic_image, 0.0, y_h, 0.0, 0.0, analysis)
        cv2.imshow('Synthetic altitude demonstration - not video', image)
        cv2.waitKey(display_ms)
    return analysis


def video_demo(filename, is_display=True, n_frames=-1):
    """ Run LaneCV on a sample video. """
    highres_scale = 0.2
    scaling_factor = 0.1

    # cap = cv2.VideoCapture('./media/flying_turn.avi') #framerate of 25
    diagnostic_frames = 0
    m = None
    b = None
    processed_frames = 0
    try:
        for i, frame in enumerate(getVideoSource(filename)):
            print('---------------------\tFrame {}'.format(i+1))
            if n_frames > 0 and i >= n_frames:
                break
            highres = cv2.resize(frame, dsize=None, fx=1.0 * highres_scale, fy=1.0 * highres_scale)
            # Keep enough rows for the global horizon search to generate
            # candidate intercepts, even for low-resolution source videos.
            analysis_scale = max(
                scaling_factor,
                MIN_ANALYSIS_HEIGHT / highres.shape[0],
            )
            img = cv2.resize(
                highres,
                dsize=None,
                fx=analysis_scale,
                fy=analysis_scale,
            )
            if USE_GRAYSCALE_DETECTION:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            valid = valid_pixel_mask(img) if USE_VALID_MASK else None
            if USE_COLOR_FIT:
                m, b, sky_fraction, agreement, state = fit_horizon_by_color(img, valid)
                if state != 'OK':
                    print('Horizon alert: {} (sky {:.0%}, fit {:.0%})'.format(
                        state, sky_fraction, agreement))
                    processed_frames += 1
                    if is_display:
                        alert_frame = plotter.add_horizon_alert(
                            resize_for_display(frame.copy()), state)
                        cv2.imshow('frame', alert_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                pitch, bank = convert_m_b_to_pitch_bank(
                    m, b, get_sigma_below(img, m, b, valid))
            else:
                answer, scores, grid, pitch, bank = optimize_real_time(img, img, m, b, 1.0, valid) #scaling_factor)
                m, b = answer[0], answer[1]
            horizon_analysis = estimate_horizon_altitude(
                img.shape,
                m,
                b,
                camera_vfov_deg=CAMERA_VFOV_DEG,
                camera_pitch_deg=CAMERA_PITCH_DEG,
                horizon_validated=HORIZON_GEOMETRY_VALIDATED,
                calibration_validated=CAMERA_CALIBRATION_VALIDATED,
                demo_only=DEMO_ONLY_ALTITUDE,
                demo_source='real_video',
            )
            draw_m, draw_b = scale_horizon_line(m, b, img.shape, frame.shape)

            if diagnostic_frames < DIAGNOSTIC_FRAME_COUNT:
                detection_geometry = horizon_analysis['geometry']
                draw_geometry = horizon_line_geometry(frame.shape, draw_m, draw_b)
                original_height, original_width = frame.shape[:2]
                detection_height, detection_width = img.shape[:2]
                print(
                    '[Horizon diagnostic frame {}] original={}x{} detection={}x{} '
                    'm={:.6f} b={:.6f} detection_line_endpoints={} '
                    'y_h={:.6f} cy={:.6f} offset={:.6f} px fy={} px delta={} deg '
                    'camera_vfov={} deg camera_pitch={} deg alpha={} deg '
                    'calibration_validated={} demo_only={} simulated_height_m={} '
                    'draw_m={:.6f} draw_b={:.6f} original_line_endpoints={} '
                    'original_y_center={} original_cy={} original_offset={} px '
                    'mapped_detection_y_center={}'.format(
                        i + 1,
                        original_width,
                        original_height,
                        detection_width,
                        detection_height,
                        m,
                        b,
                        detection_geometry['endpoints'] if detection_geometry else None,
                        detection_geometry['y_center'] if detection_geometry else None,
                        horizon_analysis['cy'],
                        horizon_analysis['pixel_offset'],
                        horizon_analysis['fy_px'],
                        horizon_analysis['delta_deg'],
                        CAMERA_VFOV_DEG,
                        CAMERA_PITCH_DEG,
                        horizon_analysis['alpha_deg'],
                        CAMERA_CALIBRATION_VALIDATED,
                        horizon_analysis['demo_only'],
                        horizon_analysis['simulated_height_m'],
                        draw_m,
                        draw_b,
                        draw_geometry['endpoints'] if draw_geometry else None,
                        draw_geometry['y_center'] if draw_geometry else None,
                        original_height / 2.0,
                        draw_geometry['y_center'] - original_height / 2.0
                        if draw_geometry else None,
                        detection_geometry['y_center'] * (original_height / detection_height)
                        if detection_geometry else None,
                    )
                )
                diagnostic_frames += 1

            if is_display:
                prediction = plotter.add_line_to_frame(
                    frame,
                    draw_m,
                    draw_b,
                    pitch,
                    bank,
                    horizon_analysis,
                )
                display_frame = resize_for_display(prediction)
                geometry = horizon_analysis.get('geometry')
                horizon_y = geometry.get('y_center') if geometry else None
                attitude_valid = (
                    horizon_y is not None
                    and 0.0 <= horizon_y < img.shape[0]
                )
                display_frame = plotter.add_attitude_limit_warning(
                    display_frame,
                    roll_deg=bank,
                    pitch_deg=None,
                    roll_thresholds=ROLL_ADVISORY_THRESHOLDS,
                    pitch_thresholds=PITCH_ADVISORY_THRESHOLDS,
                    attitude_valid=attitude_valid,
                )
                cv2.imshow('frame', display_frame)
            processed_frames += 1
            if is_display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        if is_display:
            cv2.destroyAllWindows()
        print('Video ended ({} frames processed)'.format(processed_frames))
1