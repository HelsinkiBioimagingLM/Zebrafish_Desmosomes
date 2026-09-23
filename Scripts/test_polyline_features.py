import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from skimage.draw import polygon_perimeter
from skimage.morphology import skeletonize, dilation, disk
from polyline_features import measure_skeleton_segments

def raster(rows, cols, shape=(400, 400)):
    """Draw a dense curve, thicken, skeletonize -> realistic 1-px skeleton."""
    m = np.zeros(shape, bool)
    m[np.round(rows).astype(int), np.round(cols).astype(int)] = True
    return skeletonize(dilation(m, disk(2)))

t = np.linspace(0, 1, 5000)

def test_straight_diagonal_line():
    df = measure_skeleton_segments(raster(50 + 250 * t, 50 + 250 * t))
    r = df.iloc[0]
    assert abs(r.tortuosity - 1) < 0.01
    assert r.mean_abs_curvature < 0.005
    assert abs(r.length - 250 * np.sqrt(2)) / (250 * np.sqrt(2)) < 0.02
    assert abs(r.orientation_deg % 180 - 135) < 1 or abs(r.orientation_deg - 45) < 1

def test_semicircle_radius_100():
    a = np.pi * t
    df = measure_skeleton_segments(raster(200 + 100 * np.sin(a), 200 + 100 * np.cos(a)))
    r = df.iloc[0]
    assert abs(r.median_abs_curvature - 0.01) < 0.001          # 1/R
    assert abs(r.tortuosity - np.pi / 2) < 0.03                 # arc/chord = pi/2
    assert abs(r.total_abs_turning - np.pi) < 0.15
    assert r.n_inflections == 0

def test_sine_wave_inflections():
    x = 20 + 360 * t
    y = 200 + 40 * np.sin(2 * np.pi * 3 * t)   # 3 periods -> 5 interior zero crossings
    r = measure_skeleton_segments(raster(y, x)).iloc[0]
    assert r.n_inflections == 5
    assert r.tortuosity > 1.2

def test_closed_circle():
    a = 2 * np.pi * t
    r = measure_skeleton_segments(raster(200 + 80 * np.sin(a), 200 + 80 * np.cos(a))).iloc[0]
    assert r.is_closed
    assert abs(r.length - 2 * np.pi * 80) / (2 * np.pi * 80) < 0.02
    assert abs(abs(r.net_turning) - 2 * np.pi) < 0.1
    assert np.isnan(r.tortuosity)

def test_pixel_size_scaling():
    a = np.pi * t
    m = raster(200 + 100 * np.sin(a), 200 + 100 * np.cos(a))
    r1 = measure_skeleton_segments(m).iloc[0]
    r2 = measure_skeleton_segments(m, pixel_size=0.5).iloc[0]
    assert np.isclose(r2.length, r1.length * 0.5)
    assert np.isclose(r2.mean_abs_curvature, r1.mean_abs_curvature * 2)
    assert np.isclose(r2.tortuosity, r1.tortuosity)
