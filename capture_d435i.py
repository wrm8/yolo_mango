"""Save D435i RGB photographs for annotation; no model or depth training needed."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'datasets/raw')
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--interval', type=float, default=2.0, help='Seconds between saved images')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--serial')
    parser.add_argument('--show', action='store_true', help='Preview; Q to stop (requires desktop)')
    args = parser.parse_args()
    import math
    if min(args.count, args.width, args.height, args.fps) <= 0:
        parser.error('Count, dimensions and fps must be positive')
    if not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('Interval must be finite and positive')
    try:
        import cv2
        import pyrealsense2 as rs
        import numpy as np
    except ImportError as exc:
        parser.error(f'{exc}. Install: python -m pip install -r requirements-realsense.txt')
    session = args.output / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    started = False
    saved = 0
    try:
        profile = pipeline.start(config)
        started = True
        session.mkdir(parents=True, exist_ok=False)
        device = profile.get_device()
        metadata = {
            'camera': device.get_info(rs.camera_info.name),
            'serial': device.get_info(rs.camera_info.serial_number),
            'width': args.width, 'height': args.height, 'fps': args.fps,
            'interval_s': args.interval, 'purpose': 'RGB training images',
        }
        (session / 'session.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Saving RGB images to: {session}', flush=True)
        print('Change scene, distance or lighting between shots. Ctrl+C to stop.', flush=True)
        # Allow initial automatic exposure/white balance to settle.
        for _ in range(args.fps):
            pipeline.wait_for_frames(5000)
        next_save = time.monotonic()
        with (session / 'frames.jsonl').open('w', encoding='utf-8') as manifest:
            while saved < args.count:
                frames = pipeline.wait_for_frames(5000)
                color = frames.get_color_frame()
                if not color:
                    continue
                bgr = np.asanyarray(color.get_data())
                now = time.monotonic()
                if now >= next_save:
                    filename = f'{session.name}_{saved + 1:05d}.png'
                    if not cv2.imwrite(str(session / filename), bgr):
                        raise RuntimeError(f'Cannot save image: {filename}')
                    manifest.write(json.dumps({'file': filename, 'timestamp_unix_s': time.time()}) + '\n')
                    manifest.flush()
                    saved += 1
                    next_save = now + args.interval
                    print(f'{saved}/{args.count}: {filename}', flush=True)
                if args.show:
                    cv2.imshow('D435i RGB capture - Q to quit', bgr)
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if started:
                pipeline.stop()
        finally:
            if args.show:
                cv2.destroyAllWindows()
            print(f'Saved {saved} images. Folder: {session}', flush=True)


if __name__ == '__main__':
    main()