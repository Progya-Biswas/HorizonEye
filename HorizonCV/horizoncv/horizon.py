""" 
    horizon.py
    Nicholas S. Bradford

"""

import cv2
import numpy as np
import math

DEBUG_VERBOSE = False
EARTH_RADIUS_M = 6371000.0
MAX_ABS_HORIZON_SLOPE = 1.5


def printV(text):
    """ Wrapper allowing for verbosity flag. """
    if DEBUG_VERBOSE:
        print(text)


def convert_m_b_to_pitch_bank(m, b, sigma_below):
    """ Method from original paper:
            Pitch angle (Theta) = size(ground) / size(ground) + size(sky)
            Bank angle (Phi) = tan^-1(m)
        Args:
            m
            b
            sigma_below
        Returns:
            pitch
            bank
    """
    bank = math.degrees(math.atan(m))
    pitch = sigma_below
    return pitch, bank


def horizon_line_geometry(image_shape, m, b):
    """Return line endpoints and its row at the image centre.

    The detector represents the line as y = m*x + b in pixel coordinates;
    image x increases rightward and image y increases downward. Endpoints are
    evaluated at the left and right image edges.
    """
    height, width = image_shape[:2]
    if height <= 0 or width <= 0 or not (math.isfinite(m) and math.isfinite(b)):
        return None
    x0, x1 = 0.0, float(width - 1)
    y0, y1 = m * x0 + b, m * x1 + b
    # Use W/2 to match the principal-point convention used by cy = H/2 and
    # the resize transform (source center maps to target center).
    x_center = width / 2.0
    return {
        'endpoints': ((x0, y0), (x1, y1)),
        'slope': float(m),
        'intercept': float(b),
        'y_center': float(m * x_center + b),
    }


def scale_horizon_line(m, b, source_shape, target_shape):
    """Map ``y = m*x + b`` between resized image coordinate systems.

    Resizing may produce different effective horizontal and vertical scales
    because output dimensions are rounded to integer pixels. Transforming
    slope and intercept independently preserves the same geometric line.
    """
    source_height, source_width = source_shape[:2]
    target_height, target_width = target_shape[:2]
    if source_height <= 0 or source_width <= 0:
        raise ValueError('source image dimensions must be positive')
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    return float(m) * scale_y / scale_x, float(b) * scale_y


def estimate_horizon_altitude(image_shape, m, b, camera_vfov_deg,
                              camera_pitch_deg, horizon_validated=False,
                              calibration_validated=False, demo_only=False,
                              demo_source='synthetic'):
    """Estimate spherical-Earth altitude from a validated geometric horizon.

    Camera pitch is positive when the camera optical axis points upward from
    local horizontal. Image y increases downward, so a horizon below the image
    centre has positive pixel offset and corresponds to positive depression.
    The camera pitch is subtracted from that image-derived angle.

    ``height_m`` is populated only when camera calibration and geometric
    horizon validation are established. ``simulated_height_m`` is populated
    only in explicit demo-only mode and is never marked as a valid altitude.
    """
    geometry = horizon_line_geometry(image_shape, m, b)
    result = {
        'geometry': geometry,
        'cy': None,
        'pixel_offset': None,
        'fy_px': None,
        'delta_deg': None,
        'alpha_deg': None,
        'height_m': None,
        'simulated_height_m': None,
        'valid': False,
        'demo_only': demo_only,
        'demo_source': demo_source,
        'calibration_validated': calibration_validated,
        'horizon_validated': horizon_validated,
        'reason': None,
    }
    if geometry is None:
        result['reason'] = 'invalid horizon line'
        return result
    if camera_vfov_deg is None or not math.isfinite(camera_vfov_deg) \
            or not 0.0 < camera_vfov_deg < 180.0:
        result['reason'] = 'camera vertical field of view is unknown or invalid'
        return result

    height, _ = image_shape[:2]
    y_h = geometry['y_center']
    cy = height / 2.0
    result['cy'] = cy
    result['pixel_offset'] = y_h - cy
    if not (0.0 <= y_h < height):
        result['reason'] = 'horizon line is outside the image'
        return result

    fy = (height / 2.0) / math.tan(math.radians(camera_vfov_deg / 2.0))
    delta = math.degrees(math.atan((y_h - cy) / fy))
    result['fy_px'] = fy
    result['delta_deg'] = delta
    if camera_pitch_deg is None or not math.isfinite(camera_pitch_deg) \
            or not -90.0 < camera_pitch_deg < 90.0:
        result['reason'] = 'camera mounting pitch is unknown or invalid'
        return result

    # Positive camera pitch points up; positive alpha means horizon below
    # local horizontal. Thus alpha = image depression - camera up-pitch.
    alpha = delta - camera_pitch_deg
    if abs(alpha) < 1e-12:
        alpha = 0.0
    result['alpha_deg'] = alpha

    if not math.isfinite(alpha) or not 0.0 <= alpha < 90.0:
        result['reason'] = 'horizon depression angle is outside the valid range'
        return result

    if demo_only:
        result['simulated_height_m'] = EARTH_RADIUS_M * (
            1.0 / math.cos(math.radians(alpha)) - 1.0)
        result['reason'] = 'SIMULATED ONLY: camera assumptions are uncalibrated; not a real altitude measurement'
        return result

    if not calibration_validated:
        result['reason'] = 'camera calibration has not been validated'
        return result

    # A visual color boundary may be a skyline, not the geometric horizon.
    # Require explicit external validation before publishing altitude.
    if not horizon_validated:
        result['reason'] = 'geometric horizon has not been validated (may be skyline)'
        return result
    result['height_m'] = EARTH_RADIUS_M * (1.0 / math.cos(math.radians(alpha)) - 1.0)
    result['valid'] = math.isfinite(result['height_m'])
    if not result['valid']:
        result['height_m'] = None
        result['reason'] = 'altitude calculation is not finite'
    return result


def img_line_mask(rows, columns, m, b):
    """ Produce a mask used for splitting an image into two parts with a line.
        Params:
            rows (int)
            columns (int)
            m (double)
            b (double)
        Returns:
            rows x columns np.array boolean mask with True for all values above the line
    """
    ys = np.arange(rows)[:, np.newaxis]
    xs = np.arange(columns)[np.newaxis, :]
    mask = ys >= m * xs + b
    return mask


def split_img_by_line(img, m, b, valid=None):
    """ Split image into two parts using a line.
        Params:
            m: slope
            b: y-intercept
        Returns:
            (arr1, arr2): two np.arrays with one column per channel (3 for BGR, 1 for
            grayscale) and N rows (one for each pixel)
    """
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    mask = img_line_mask(rows=img.shape[0], columns=img.shape[1], m=m, b=b)
    assert len(mask.shape) == 2
    assert mask.shape[0] == img.shape[0]
    assert mask.shape[1] == mask.shape[1]
    if valid is None:
        valid = np.ones(mask.shape, dtype=bool)
    segment1 = img[mask & valid]
    segment2 = img[~mask & valid]
    reshape1 = segment1.reshape(-1, segment1.shape[-1])
    reshape2 = segment2.reshape(-1, segment2.shape[-1])
    assert reshape1.shape[1] == reshape2.shape[1]
    if not (segment1.shape[0] > 1 and segment2.shape[0] > 1): 
        print('!!    Warning: Invalid hypothesis: ' + str(m) + ' ' + str(b))
    return (reshape1, reshape2)


def compute_variance_score(segment1, segment2):
    """ Params:
            segment1 (np.array): n x 3, where n is number of pixels in first segment
            segment2 (np.array): n x 3, where n is number of pixels in first segment
        Returns:
            F (np.double): the score for these two segments (higher = better line hypothesis)
        Note that linalg.eigh() is more stable than np.linalg.eig, but only for symmetric matrices.
    """
    channels = segment1.shape[1]
    assert segment2.shape[1] == channels
    if not (segment1.shape[0] > 1 and segment2.shape[0] > 1):
        return -1.0
    cov1 = np.atleast_2d(np.cov(segment1.T))
    cov2 = np.atleast_2d(np.cov(segment2.T))
    assert cov1.shape == cov2.shape == (channels, channels)
    evals1, evecs1 = np.linalg.eigh(cov1)
    evals2, evecs2 = np.linalg.eigh(cov2)
    det1 = evals1.prod() #np.linalg.det(cov1)
    det2 = evals2.prod() #np.linalg.det(cov2)
    score =  det1 + det2 + (np.sum(evals1) ** 2) + (np.sum(evals2) ** 2)
    return score ** -1


def score_line(img, m, b, valid=None):
    """ Computes the score of a given line on an image, using the cost function.
        Params:
            img
            m
            b
        Returns:
            scrore (float)
    """
    # print('Score', img.shape, m, b)
    seg1, seg2 = split_img_by_line(img, m=m, b=b, valid=valid)
    # print('\tSegment shapes: ', seg1.shape, seg2.shape)
    score = compute_variance_score(seg1, seg2)
    return score


def score_grid(img, grid, valid=None):
    height, width = img.shape[:2]
    center_x = width / 2.0
    grid = [
        (m, b) for m, b in grid
        if abs(m) <= MAX_ABS_HORIZON_SLOPE
        and 0.0 <= b < height
        and 0.0 <= m * center_x + b < height
    ]
    if not grid:
        raise ValueError('No plausible horizon candidates intersect the image')
    scores = [score_line(img, m, b, valid) for m, b in grid]
    max_index = np.argmax(scores)
    m, b = grid[max_index]
    return m, b, scores, grid, scores[max_index]


def accelerated_search(img, m, b, current_score, valid=None):
    """ Hill-climbing bisection search of the (m, b) space in search for the 
            approximate optimum, given a starting point.
    """
    max_iter = 10
    delta_m = 0.25
    delta_b = 1.0
    delta_factor = 0.75
    printV('\tAccel. search begin m: {} b: {}'.format(m, b))
    for i in range(max_iter):
        # print('\tDelta', delta_m, delta_b, 'M&B', m, b)
        grid = [
                    (m + delta_m, b),
                    (m - delta_m, b),
                    (m, b + delta_b),
                    (m, b - delta_b),
                    # (m + delta_m, b + delta_b),
                    # (m - delta_m, b - delta_b),
                    # (m - delta_m, b + delta_b),
                    # (m + delta_m, b - delta_b),
            ]
        mTmp, bTmp, scores, grid, max_score = score_grid(img, grid, valid)
        if max_score >= current_score:
            m = mTmp
            b = bTmp
        # else:
        #     print ('Reached a peak?')
        delta_m *= delta_factor
        delta_b *= delta_factor
    printV('\tAccel. search end   m: {} b: {}'.format(m, b))
    return m, b    


def get_sigma_below(img, m, b, valid=None):
    """ Returns % of image below the horizon line. """
    seg1, seg2 = split_img_by_line(img, m, b, valid)
    return seg1.size / (seg1.size + seg2.size)


def optimize_scores(img, highres, slope_range, intercept_range, scaling_factor, valid=None):
    """ Searches for the approximate highest-scoring (m, b) parameter pair.
        Params:
            img:
            highres:
            slope_range:
            intercept_range:
            scaling_factor:
        Returns:
            Answer: Tuple of (m, b)
            Scores: list of np.double scores corresponding to grid elements
            Grid: list of (m, b) tuples that were examined
            pitch: pitch angle
            bank: bank angle
    """
    print(img.shape)
    printV('Optimize... img shape {} highres shape {}'.format(img.shape, highres.shape))
    grid = [(m, b) for b in intercept_range for m in slope_range]    

    print(len(grid))
    m, b, scores, grid, max_score = score_grid(img, grid, valid)
    m2, b2 = accelerated_search(img, m, b * scaling_factor, max_score, valid)
    b2 /= scaling_factor
    pitch, bank = convert_m_b_to_pitch_bank(m=m2, b=b2, sigma_below=get_sigma_below(img, m2, b2, valid))
    print('\tPitch: {0:.2f}% \t Bank: {1:.2f} degrees'.format(pitch * 100, bank))
    return (m2, b2), scores, grid, pitch, bank


def optimize_global(img, highres, scaling_factor, valid=None):
    """ Search an entire image for the horizon. """
    printV('optimize_global()')
    return optimize_scores(img, highres,
                        slope_range=np.arange(
                            -MAX_ABS_HORIZON_SLOPE,
                            MAX_ABS_HORIZON_SLOPE + 0.01,
                            0.1,
                        ),
                        intercept_range=np.arange( 1, img.shape[0] - 2, 0.1), #0.5
                        scaling_factor=scaling_factor, valid=valid)


def optimize_local(img, highres, m, b, scaling_factor, valid=None):
    """ Search around a local space. """
    printV('optimize_local()')
    return optimize_scores(img, highres,
                        slope_range=np.arange(
                            max(-MAX_ABS_HORIZON_SLOPE, m - 0.5),
                            min(MAX_ABS_HORIZON_SLOPE, m + 0.5) + 0.01,
                            0.1,
                        ),
                        intercept_range=np.arange(max(1.0, b - 4.0), min(img.shape[0], b + 4.0), 0.5),
                        scaling_factor=scaling_factor, valid=valid)


def optimize_real_time(img, highres, m, b, scaling_factor, valid=None):
    """ Begin by searching globally for the horizon, and then optimize by continuing the
            search in only the area around the previously detected horizon. Note that the
            original paper discourages this due to the potential for serious issues upon 
            miscalibration or a few bad frames; this should be rectified with a Kalman filter
            and periodic Global checks.
    """
    if m is None or b is None:
        return optimize_global(img, highres, scaling_factor, valid)

    height, width = img.shape[:2]
    line_y_at_center = m * (width / 2.0) + b
    previous_line_is_plausible = (
        math.isfinite(m)
        and math.isfinite(b)
        and abs(m) <= MAX_ABS_HORIZON_SLOPE
        and 0.0 <= line_y_at_center < height
    )
    if not previous_line_is_plausible:
        return optimize_global(img, highres, scaling_factor, valid)
    return optimize_local(img, highres, m, b, scaling_factor, valid)


def valid_pixel_mask(img, dark_threshold=110, erode_frac=0.04):
    """ Mask of pixels worth scoring. Removes dark *borders* only:
        - porthole / lens vignette: detected when all four frame corners are
          mostly dark; then only the largest bright region is kept;
        - letterbox bars: dark, narrow, full-height vertical blocks.
        Dark ground is kept, so it is never mistaken for a border. Falls back
        to all pixels if too little survives.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dark = gray <= dark_threshold
    cy, cx = max(1, int(0.15 * height)), max(1, int(0.15 * width))
    corners = (dark[:cy, :cx], dark[:cy, -cx:], dark[-cy:, :cx], dark[-cy:, -cx:])
    valid = np.ones(gray.shape, dtype=bool)

    if all(corner.mean() > 0.4 for corner in corners):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (~dark).astype(np.uint8), connectivity=4)
        if count < 2:
            return valid
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        valid = labels == largest
    else:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            dark.astype(np.uint8), connectivity=4)
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if (y == 0 and y + h == height and w < 0.25 * width
                    and area > 0.8 * w * h):
                valid[labels == label] = False
    if not valid.all():
        k = max(1, int(round(erode_frac * max(height, width))))
        valid = cv2.erode(valid.astype(np.uint8),
                          np.ones((2 * k + 1, 2 * k + 1), np.uint8),
                          borderType=cv2.BORDER_REPLICATE).astype(bool)
    if valid.mean() < 0.2:
        return np.ones(gray.shape, dtype=bool)
    return valid


SKY_ONLY_FRACTION = 0.93
GROUND_ONLY_FRACTION = 0.15
MIN_FIT_AGREEMENT = 0.62


def sky_color_mask(img):
    """ Per-pixel guess of "sky or cloud" from colour alone (BGR in).
        Sky/cloud: blue-to-lavender hues (blue sky, cyan haze, tinted cloud) or
        bright near-neutral pixels (white/grey/beige cloud and overcast sky).
        Sky must be bright: dark bluish pixels are ground/haze. Everything else
        (pink, brown, green, dark) counts as ground, so a
        blue/white cloud edge is NOT treated as the horizon.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32) * 2.0
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    blue_family = (hue >= 150.0) & (hue <= 275.0) & (sat >= 0.05) & (val >= 0.6)
    neutral_bright = (sat < 0.10) & (val > 0.65)
    return blue_family | neutral_bright


def fit_horizon_by_color(img, valid=None):
    """ Find the line that best separates sky/cloud colours (above) from ground
        colours (below), by maximising pixel agreement with sky_color_mask.
        Returns:
            m, b: line y = m*x + b (None, None if no usable pixels)
            sky_fraction: share of valid pixels classified as sky/cloud
            agreement: share of valid pixels consistent with the fitted line
            state: 'OK', 'SKY ONLY', 'GROUND ONLY' or 'NO HORIZON'
    """
    height, width = img.shape[:2]
    if valid is None:
        valid = np.ones((height, width), dtype=bool)
    if valid.sum() < 10:
        return None, None, 0.0, 0.0, 'NO HORIZON'
    sky = sky_color_mask(img)
    sky_fraction = float(sky[valid].mean())
    if sky_fraction >= SKY_ONLY_FRACTION:
        return None, None, sky_fraction, sky_fraction, 'SKY ONLY'
    if sky_fraction <= GROUND_ONLY_FRACTION:
        return None, None, sky_fraction, 1.0 - sky_fraction, 'GROUND ONLY'

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]
    sky_v = (sky & valid)
    ground_v = (~sky & valid)
    total = float(valid.sum())
    best = (-1.0, None, None)
    center_x = width / 2.0
    for angle in np.arange(-75.0, 75.1, 2.0):
        m = math.tan(math.radians(angle))
        line = m * xs
        for b in np.arange(0.5, height, 0.5):
            if not 0.0 <= m * center_x + b < height:
                continue
            below = ys >= line + b
            agree = (np.count_nonzero(ground_v & below)
                     + np.count_nonzero(sky_v & ~below))
            if agree > best[0]:
                best = (agree, float(m), float(b))
    agreement = best[0] / total
    state = 'OK' if agreement >= MIN_FIT_AGREEMENT else 'NO HORIZON'
    return best[1], best[2], sky_fraction, agreement, state
