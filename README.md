# HorizonEye

Real-time horizon detection with computer vision in Python, for UAV applications. A part of WPI's [ACAP](http://www.acap.io) (Autonomous Cargo Aircraft Project), completed 2016-2017 by Nicholas Bradford (view the MQP report [here](https://web.wpi.edu/Pubs/E-project/Available/E-project-042717-143558/unrestricted/ACAPFinalReport.pdf)). Algorithm source found in [paper](http://www.tandfonline.com/doi/pdf/10.1163/156855303769156983?needAccess=true) and [master's thesis](https://pdfs.semanticscholar.org/9037/e7955e4b7c9339dc34bd75e563a86636cca8.pdf).

![Results screenshot](media/results/screenshot.png "Results screenshot")

    /horizoncv                  Python module.
        demo.py                 Running useful real-time demos
        horizon.py              All the important computation
        plotter.py              Plotting graphs
        archive.py              Archived code that might be useful to view in the future.
    /proto
    /media                      Footage for testing
    requirements.txt            Install with $ python install -r 
    runner.py                   Run tests and a demo.

## Algorithm

* Use algorithm from "Vision-guided flight stability and control for micro air vehicles" (Ettinger et al. ). 
    * Intuition: horizon will be a line dividing image into two segments with low variance, which can be modeled as minimizing the product of the three eigenvalues of the covariance matrix (the determinant). 
    * When the covariance matrix is nearly singular (due to color issues), the determinant will also be driven to zero. Thus, we introduce additional terms to supplement the score when this case occurs (the determinant dominates it in the normal case): where g=GROUND and s=SKY (covariance matrices) 

    F = [det(G) + det(S) + (eigG1 + eigG1 + eigG1)^2 + (eigS1 + eigS1 + eigS1)^2]^-1

* Note: The pitch angle cannot be exactly calculated from an arbitrary horizon line, however the pitch angle will be closely proportional to the percentage of the image above or below the line.
* Experimental altitude estimation: HorizonCV's optimizer returns a pixel-space line `y = m*x + b`, not an angle. The demo evaluates that line at the image centre, converts its pixel offset using a calibrated vertical field of view, then estimates altitude with a spherical-Earth model. This estimate is valid only if the detected line is the true geometric horizon; a skyline or other image boundary can produce a false result. Altitude is disabled by default and must not be treated as accurate without validation against known heights.
* Performance: In its current form, can run 10000 iterations in 9.1 seconds, or about 30 per iteration at 30Hz. Java performance benefit of 10x would mean 300 per iteration, and moving to 10Hz would leave ~1000 per iteration. The initial optimized search grid is on a 12x12 grid (144 values), which is then refined on a full-resolution image using a gradient-descent-like sampling technique. (requires 4 checks at each step and ~7 steps = ~28, but will but at higher resolution). Total requirements: must be able to run at least ~200 checks/second

### Horizon-based altitude calibration (experimental)

Set `CAMERA_VFOV_DEG` and `CAMERA_PITCH_DEG` in `horizoncv/demo.py` from the actual camera calibration and mount measurement. The default values (45° and 0°) are unverified assumptions, not calibration data. The safe validated mode is the default: `CAMERA_CALIBRATION_VALIDATED` and `HORIZON_GEOMETRY_VALIDATED` are both false, so no altitude is emitted. Set those validation flags true only after calibration and true-horizon validation, respectively. Vertical FOV is the camera's vertical field of view, not its horizontal FOV. Camera pitch is positive when the optical axis points upward from local horizontal; image y increases downward. Therefore a horizon below the image centre has a positive pixel offset, and its image-derived angle is corrected by subtracting the upward camera pitch. With a level camera, `CAMERA_PITCH_DEG = 0`; a level, zero-altitude test horizon at the image centre must give zero depression and approximately zero estimated altitude.

`run_demo.py` runs a separate synthetic altitude demonstration before opening the real video. `DEMO_ONLY_ALTITUDE = True` enables an experimental estimate from each real frame where the measured depression is physically valid, using the assumed FOV and mounting pitch. The video overlay labels such values **EXPERIMENTAL ALTITUDE: ... m | UNCALIBRATED DEMO**. Invalid/negative depression remains N/A; the sign is not flipped and the absolute angle is not used. The validated `height_m` result remains gated by both validation flags, and neither flag is changed by demo mode.

The generated-image test separately uses `SYNTHETIC_DEPRESSION_DEG = 1.0` and displays **SIMULATED ALTITUDE - UNCALIBRATED**. It is not derived from video and is never passed off as a real-frame estimate. Current `turn1.mp4` detections place the line above the image centre in the inspected first frames (negative offset/depression), so those frames correctly have no experimental numerical estimate. All assumed parameters remain unverified and neither path establishes real-world altitude accuracy.

The implementation uses `fy = (H / 2) / tan(VFOV / 2)`, `delta = atan((y_h - H/2) / fy)`, `alpha = delta - CAMERA_PITCH_DEG`, and `height = 6371000 * (1/cos(alpha) - 1)`. The code keeps `CAMERA_CALIBRATION_VALIDATED = False` and `HORIZON_GEOMETRY_VALIDATED = False` by default. Only enable the latter after checking representative frames that the detector follows the unobstructed geometric horizon, and only enable calibration validation after camera FOV and mounting pitch are measured and checked. Until both are validated altitude remains `N/A`. A missing/invalid line, unknown camera calibration, or invalid depression angle also suppresses altitude. This is an experimental estimate, not a certified altimeter.

The video detector runs on a repeatedly resized image whose dimensions are rounded to integers. When rendering the line on the original frame, the code scales x and y separately and transforms both slope and intercept; multiplying the intercept alone by a nominal resize factor would not reliably preserve the line.


## Usage

### Requirements

Note that you'll need OpenCV compiled with FFMPEG support in order to load videos. Use this [script](https://github.com/nsbradford/ExuberantCV/blob/master/installOpenCV.sh) and some tutorials to understand:

* MacOS
    * The [easy](http://www.pyimagesearch.com/2016/12/19/install-opencv-3-on-macos-with-homebrew-the-easy-way/) way using Homebrew
    * The [hard](http://www.pyimagesearch.com/2016/11/28/macos-install-opencv-3-and-python-2-7/?__s=6qbo7sdne7fzcniijrik) way
    * With FFMPEG support [here](http://blog.jiashen.me/2014/12/23/build-opencv-3-on-mac-os-x-with-python-3-and-ffmpeg-support/)
* Ubuntu
    * [General](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
    * [With CUDA](http://www.pyimagesearch.com/2016/07/11/compiling-opencv-with-cuda-support/)
    * With ffmpeg support (needed for videos) [here](https://ubuntuforums.org/showthread.php?t=2219550)
    * [Compiling OpenCV with FFMPEG](http://www.wiomax.com/compile-opencv-and-ffmpeg-on-ubuntu/)
* [Installing ffmpeg](http://tipsonubuntu.com/2016/11/02/install-ffmpeg-3-2-via-ppa-ubuntu-16-04/)
* [Compiling ffmpeg from source](http://blog.mycodesite.com/compile-opencv-with-ffmpeg-for-ubuntudebian/)
* Installing CUDA
    * Go [here](https://developer.nvidia.com/cuda-downloads) and download .deb for Ubuntu, DO NOT try the automatic runner it’s a pain 
    * Official documentation [guide](http://developer.download.nvidia.com/compute/cuda/7.5/Prod/docs/sidebar/CUDA_Installation_Guide_Linux.pdf)
    * Helpful blog [post](http://kislayabhi.github.io/Installing_CUDA_with_Ubuntu/)

Then to finish, activate your new `cv` virtual environment and install the other requirements:

    $ workon cv 
    $ pip install -r requirements.txt

### Demo

Run a demo:

    $ python runner.py   

### Protobuf compilation

After modifying the `horizon.proto` definition, use `protoc` to recompile the python and java files.

    $ cd ./horizon/proto
    $ protoc -I=. --python_out=. horizoncv.proto
    $ protoc -I=. --java_out=.  horizoncv.proto 

## TODO

### Priorities

* Test suite.
* Optimization (find slow part!) to run in real-time (C++/GPU required? )
    * Slowdown when horizon is nearly flat
    * Refinement to run on high-res images
* Search space in [pitch, bank] instead of [slope, intercept] form
    * Why large bank angles isn't found properly? Because the slope-intercept ranges are limited

### Backlog

* Add ZMQ messaging integration.    
* Cleaning and packaging
* Kalman filtering
* Extreme attitude detection (when no horizon is visible)



