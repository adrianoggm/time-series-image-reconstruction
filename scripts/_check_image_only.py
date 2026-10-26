"""Image-only decoding: the calibration enters every decoder only as the final affine map (review, 2026-10-26).

Each representation's decoder receives the stored image and the channel calibration (max, min). If the
calibration were used for anything but the final map to physical units (choosing a reflection, a branch, a
threshold...), decoding the same image with another calibration would not be an affine function of the first
decoding. For every representation and a set of random series, this script decodes the same 8-bit image with
the true calibration and with an arbitrary one and checks that
    decode(image, mm2) == mn2 + (decode(image, mm) - mn) / (mx - mn) * (mx2 - mn2).
Decoders never receive the series: src/representations.py passes them the image and (max, min) only.

    python -m scripts._check_image_only
"""
import numpy as np


def main():
    from src.representations import REPRESENTATIONS

    rng = np.random.default_rng(0)
    series = [np.cumsum(rng.normal(size=128)), rng.normal(size=128), np.sin(np.linspace(0, 9, 128)) + 0.1 * rng.normal(size=128)]
    worst = {}
    for name, rep in REPRESENTATIONS.items():
        err = 0.0
        for x in series:
            mx, mn = float(x.max()), float(x.min())
            img = rep.to_uint8(rep.encode(x, (mx, mn)))
            r1 = np.asarray(rep.decode(rep.from_uint8(img), (mx, mn)), dtype=float)
            mx2, mn2 = 7.5, -3.0                               # an unrelated calibration
            r2 = np.asarray(rep.decode(rep.from_uint8(img), (mx2, mn2)), dtype=float)
            expected = mn2 + (r1 - mn) / (mx - mn) * (mx2 - mn2)
            err = max(err, float(np.max(np.abs(r2 - expected)) / (mx2 - mn2)))
        worst[name] = err
        print(f"{name:18} max relative deviation from an affine map: {err:.1e}")
    bad = {k: v for k, v in worst.items() if v > 1e-9}
    assert not bad, f"calibration used beyond the final affine map: {bad}"
    print("image-only check ok: the calibration is only the final affine map for every decoder")


if __name__ == "__main__":
    main()
