
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def _altitude_status_label(horizon_analysis):
    if horizon_analysis and horizon_analysis['demo_only']:
        simulated_height = horizon_analysis['simulated_height_m']
        if horizon_analysis.get('demo_source') == 'real_video':
            if simulated_height is not None:
                return 'EXPERIMENTAL ALTITUDE: {:.1f} m | UNCALIBRATED DEMO'.format(
                    simulated_height)
            return 'EXPERIMENTAL ALTITUDE: N/A ({}) | UNCALIBRATED DEMO'.format(
                horizon_analysis['reason'])
        if simulated_height is not None:
            return 'SIMULATED ALTITUDE - UNCALIBRATED: {:.1f} m'.format(
                simulated_height)
        return 'SIMULATED ALTITUDE - UNCALIBRATED: N/A ({})'.format(
            horizon_analysis['reason'])
    if horizon_analysis and horizon_analysis['valid']:
        return 'REAL VIDEO ALTITUDE (VALIDATED): {:.1f} m'.format(
            horizon_analysis['height_m'])
    reason = horizon_analysis['reason'] if horizon_analysis else 'not calculated'
    if not horizon_analysis or not horizon_analysis['calibration_validated']:
        return 'REAL VIDEO - UNCALIBRATED ASSUMPTIONS: N/A ({})'.format(reason)
    return 'REAL VIDEO - HORIZON NOT VALIDATED: N/A ({})'.format(reason)


def _crossed_limit_band(value, thresholds):
    """Return the highest crossed (threshold, label) pair, if any."""
    crossed = [
        (threshold, label)
        for threshold, label in thresholds
        if abs(value) > threshold
    ]
    return max(crossed, key=lambda item: item[0]) if crossed else None


def add_attitude_limit_warning(img, roll_deg, roll_thresholds,
                              pitch_deg=None, pitch_thresholds=None,
                              attitude_valid=True):
    """Overlay configurable, non-operational attitude advisory bands.

    Thresholds are ordered as ``(absolute_degrees, label)`` pairs. They must
    be selected for the specific aircraft and are not regulatory limits.
    """
    warnings = []
    if not attitude_valid:
        warnings.append('ATTITUDE ESTIMATE INVALID - CHECK INDEPENDENT INSTRUMENTS')
    else:
        roll_band = _crossed_limit_band(roll_deg, roll_thresholds)
        if roll_band:
            threshold, label = roll_band
            warnings.append(
                'ROLL {}: {:.1f} deg (>{:.1f} deg)'.format(
                    label, roll_deg, threshold))
        pitch_band = (
            _crossed_limit_band(pitch_deg, pitch_thresholds)
            if pitch_deg is not None and pitch_thresholds else None
        )
        if pitch_band:
            threshold, label = pitch_band
            warnings.append(
                'PITCH {}: {:.1f} deg (>{:.1f} deg)'.format(
                    label, pitch_deg, threshold))
    if not warnings:
        return img

    height, width = img.shape[:2]
    banner_height = max(36, 30 + 26 * (len(warnings) - 1))
    cv2.rectangle(img, (0, 0), (width, banner_height), (0, 0, 180), -1)
    for index, warning in enumerate(warnings):
        cv2.putText(
            img,
            'ADVISORY - ' + warning,
            (10, 25 + index * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return img


def add_line_to_frame(img, m, b, pitch, bank, horizon_analysis=None):
    """Draw the overlays in OpenCV's native BGR color order."""
    output_img = img.copy()
    pt1 = (0, int(round(float(b))))
    pt2 = (img.shape[1] - 1, int(round(float(m) * (img.shape[1] - 1) + float(b))))
    cv2.line(img=output_img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)

    labels = [
        'Roll: {:.1f} deg'.format(bank),
        'Horizon depression{}: {}'.format(
            '' if horizon_analysis and horizon_analysis['calibration_validated'] else ' (uncalibrated)',
            'N/A' if not horizon_analysis or horizon_analysis['alpha_deg'] is None
            else '{:.2f} deg'.format(horizon_analysis['alpha_deg'])),
          _altitude_status_label(horizon_analysis),
    ]
    for index, label in enumerate(labels):
        cv2.putText(output_img, label, (10, 25 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 255), 1, cv2.LINE_AA)

    return output_img


def plot2D(img, scores, m, b, pitch, bank):
    """ 2d plot of the optimization surface next to the img. """
    font = cv2.FONT_HERSHEY_SIMPLEX
    label = 'Pitch: ' + "{:.1f}".format(pitch) + ' %     Bank: ' + "{:.1f}".format(bank) + ' degrees'
    # cv2.putText(img, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    pt1 = (0, b.astype(np.int64))
    pt2 = img.shape[1] - 1, (m * (img.shape[1] - 1) + b).astype(np.int64)
    cv2.line(img=img, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Input')
    
    plt.subplot(122)
    y = list(map(lambda x: x * (10**8), scores))
    x = list(range(len(scores)))
    plt.fill_between(x, y, [0 for x in range(len(scores))], color='grey')
    # plt.imshow(resized),plt.title('Output')
    plt.show()


def scatter3D(X, Y, Z):
    """ 3d scatterplot of the optimization surface. """
    print('Plot in 3D...')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c=np.abs(Z), cmap=cm.coolwarm)
    ax.set_xlabel('M (slope)')
    ax.set_ylabel('B (intercept)')
    ax.set_zlabel('Z Label')
    plt.show()


def add_horizon_alert(img, text):
    """Draw a full-width alert banner when no sky/ground horizon is visible."""
    height, width = img.shape[:2]
    cv2.rectangle(img, (0, 0), (width, 44), (0, 0, 200), -1)
    cv2.putText(img, 'HORIZON ALERT - ' + text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return img
