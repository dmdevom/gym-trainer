"""
Run every test photo through the analyzer. No server, no HTTP — just the maths.

    python scripts/test_photos.py                   # data/photos/*.jpg
    python scripts/test_photos.py data/photos/old   # the stock photos

Swap models to compare (this is your README benchmark, for free):

    python scripts/test_photos.py > /tmp/mp.txt
    POSE_BACKEND=yolo python scripts/test_photos.py > /tmp/yolo.txt
    diff -y /tmp/mp.txt /tmp/yolo.txt
"""

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from analyzer import analyze  # noqa: E402
from backends import get_backend  # noqa: E402


def main() -> int:
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data/photos")
    photos = sorted(p for p in folder.glob("*.jpg"))
    if not photos:
        print(f"No .jpg files in {folder}/")
        return 1

    info = get_backend().info()
    model = info.get("model", "—")
    print(f"\nbackend: {info['backend']}   model: {model}\n")

    header = f"{'file':<18} {'angle':>7} {'side':>6} {'conf':>5} {'ms':>6}  verdict"
    print(header)
    print("-" * len(header))

    failures = 0
    times = []

    for p in photos:
        t0 = time.perf_counter()
        r = analyze(str(p))
        ms = (time.perf_counter() - t0) * 1000
        times.append(ms)

        if "error" in r:
            failures += 1
            print(f"{p.name:<18} {'-':>7} {'-':>6} {'-':>5} {ms:>6.0f}  {r['error']}")
            continue

        # A near-180 angle is the floating-point cliff. If the clip in
        # calc_angle were missing, these are the rows that would have blown up.
        flag = "  <- near the 180 cliff" if r["elbow_angle"] > 175 else ""
        print(
            f"{p.name:<18} {r['elbow_angle']:>7.1f} {r['side_analyzed']:>6} "
            f"{r['confidence']:>5.2f} {ms:>6.0f}  {r['phase']}{flag}"
        )

    avg = sum(times) / len(times)
    print("-" * len(header))
    print(f"{len(photos)} photos, {failures} refused, {avg:.0f} ms/photo average")

    # 30s of video at 30fps, every 3rd frame. The number that decides Phase 2.
    print(f"projected: {avg * 300 / 1000:.1f}s for a 300-frame video\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())