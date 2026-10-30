import argparse

from horizoncv import demo


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run the HorizonEye demo.')
    parser.add_argument(
        '--video-only',
        action='store_true',
        help='skip the synthetic preview and start the real video immediately',
    )
    args = parser.parse_args(argv)

    if not args.video_only:
        print('Starting synthetic demo', flush=True)
        # Generated example is shown for three seconds and is not video data.
        demo.synthetic_altitude_demo(display_ms=3000)
    print('Starting real video', flush=True)
    # demo.video_demo("WhatsApp Video 2026-10-24 at 11.42.32 AM.mp4")
    demo.video_demo("turn1.mp4")
    # demo.video_demo("rocket_camera_feed.webm")
    


if __name__ == "__main__":
    main()
