"""
Shape descriptors for 2D polylines extracted from a skeletonized binary mask.

Input: a 2D boolean mask where each connected component is a simple path
(skeleton with branch/junction nodes already removed), i.e. every segment is
"2-degree connected" apart from its two endpoints.

Pipeline per segment:
    1. label connected components (8-connectivity)
    2. order the pixels of each component into a path (shortest path between
       the two geodesically farthest pixels; closed loops handled separately)
    3. smooth the pixel coordinates (removes the pixel-grid "staircase", which
       otherwise inflates length and curvature)
    4. resample at uniform arc-length spacing
    5. compute descriptors

Compatible with Python >= 3.10 (tested on 3.11; no 3.12-only features used).
Dependencies: numpy, scipy, scikit-image, pandas.

References
----------
- Tortuosity (arc-chord ratio), sum-of-angles metric (SOAM), inflection count:
  Bullitt E. et al. (2003) "Measuring tortuosity of the intracerebral
  vasculature from MRA images", IEEE Trans Med Imaging 22(9):1163-1171.
- Curvature of a parametric curve k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2):
  https://en.wikipedia.org/wiki/Curvature#In_terms_of_a_general_parametrization
- A full-featured skeleton analysis library (graph building, branch stats):
  skan, Nunez-Iglesias et al. (2018) PeerJ 6:e4312, https://skeleton-analysis.org
"""

import numpy as np
import pandas as pd
import os
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage.measure import label
from skimage.io import imsave

# --------------------------------------------------------------------------
# 1-2. Segment extraction and pixel ordering
# --------------------------------------------------------------------------

def _pixel_graph(coords):
    """Sparse graph linking 8-connected pixels (edge weight 1 or sqrt(2))."""
    index = {(r, c): i for i, (r, c) in enumerate(coords)}
    rows, cols, weights = [], [], []
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                j = index.get((r + dr, c + dc))
                if j is not None and j != i:
                    rows.append(i)
                    cols.append(j)
                    weights.append(np.hypot(dr, dc))
    n = len(coords)
    return coo_matrix((weights, (rows, cols)), shape=(n, n)).tocsr()


def _path_from_predecessors(predecessors, start, end):
    path = [end]
    while path[-1] != start:
        path.append(predecessors[path[-1]])
    return path[::-1]


def order_segment_pixels(coords):
    """
    Order the pixels of one skeleton segment along the curve.

    Parameters
    ----------
    coords : (N, 2) int array of (row, col) pixel coordinates.

    Returns
    -------
    ordered : (M, 2) array of (row, col), M <= N (redundant corner pixels in
              the pixel staircase are skipped).
    is_closed : bool, True if the segment is a closed loop.
    """
    coords = np.asarray(coords)
    if len(coords) < 3:
        return coords, False

    graph = _pixel_graph([tuple(p) for p in coords])
    n_neighbors = np.diff(graph.indptr)
    is_closed = bool(np.all(n_neighbors >= 2)) and len(coords) >= 4

    if is_closed:
        # Cut one edge of the loop, then walk the long way round.
        start = 0
        stop = graph.indices[graph.indptr[start]]
        graph = graph.tolil()
        graph[start, stop] = 0
        graph[stop, start] = 0
        graph = graph.tocsr()
        graph.eliminate_zeros()
    else:
        # Endpoints = the two geodesically farthest pixels ("double sweep").
        dist = dijkstra(graph, indices=0)
        start = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))
        dist = dijkstra(graph, indices=start)
        stop = int(np.argmax(np.where(np.isfinite(dist), dist, -1)))

    _, predecessors = dijkstra(graph, indices=start, return_predecessors=True)
    path = _path_from_predecessors(predecessors, start, stop)
    return coords[path], is_closed


# --------------------------------------------------------------------------
# 3-4. Smoothing and resampling
# --------------------------------------------------------------------------

def smooth_polyline(points, sigma=4.0, is_closed=False):
    """
    Gaussian-smooth x and y along the curve. sigma is in samples.

    Open curves are padded by point-reflection through each endpoint
    (p[-i] = 2*p[0] - p[i]), which continues the curve straight instead of
    bending it, so endpoints stay fixed and no false curvature appears there.
    """
    points = np.asarray(points, dtype=float)
    if sigma <= 0 or len(points) < 3:
        return points
    if is_closed:
        return ndimage.gaussian_filter1d(points, sigma, axis=0, mode="wrap")
    pad = min(int(4 * sigma), len(points) - 1)
    head = 2 * points[0] - points[1:pad + 1][::-1]
    tail = 2 * points[-1] - points[-pad - 1:-1][::-1]
    padded = np.vstack([head, points, tail])
    smoothed = ndimage.gaussian_filter1d(padded, sigma, axis=0, mode="nearest")
    return smoothed[pad:pad + len(points)]


def resample_polyline(points, step=1.0, is_closed=False):
    """Linearly resample a polyline at (approximately) uniform arc-length spacing."""
    points = np.asarray(points, dtype=float)
    if is_closed:
        points = np.vstack([points, points[:1]])
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))])
    if s[-1] == 0:
        return points[:1]
    n = max(int(round(s[-1] / step)), 2)
    s_new = np.linspace(0, s[-1], n + 1)
    if is_closed:
        s_new = s_new[:-1]  # do not repeat the first point
    return np.column_stack([np.interp(s_new, s, points[:, k]) for k in range(2)])


# --------------------------------------------------------------------------
# 5. Descriptors
# --------------------------------------------------------------------------

def polyline_length(points, is_closed=False):
    points = np.asarray(points, dtype=float)
    if is_closed:
        points = np.vstack([points, points[:1]])
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def chord_length(points):
    """Straight-line distance between first and last point."""
    return float(np.linalg.norm(points[-1] - points[0]))


def signed_curvature(points, is_closed=False):
    """
    Signed curvature (1 / length unit) at each point of a uniformly resampled
    curve, k = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2).
    Positive = turning counter-clockwise in (x=col, y=row) image coordinates.
    """
    y, x = points[:, 0], points[:, 1]
    if is_closed:
        # periodic finite differences
        dx = (np.roll(x, -1) - np.roll(x, 1)) / 2
        dy = (np.roll(y, -1) - np.roll(y, 1)) / 2
        ddx = np.roll(x, -1) - 2 * x + np.roll(x, 1)
        ddy = np.roll(y, -1) - 2 * y + np.roll(y, 1)
    else:
        dx, dy = np.gradient(x), np.gradient(y)
        ddx, ddy = np.gradient(dx), np.gradient(dy)
    denom = (dx**2 + dy**2) ** 1.5
    return np.divide(dx * ddy - dy * ddx, denom, out=np.zeros_like(denom), where=denom > 0)


def count_inflections(curvature, ds, min_lobe_turning=0.2):
    """
    Number of inflection points (curvature sign changes).

    The curve is split into "lobes" (runs of same-sign curvature). Lobes that
    turn by less than min_lobe_turning radians in total are treated as noise
    and dropped before counting sign changes between the remaining lobes.
    """
    signs = np.sign(curvature)
    run_starts = np.flatnonzero(np.diff(signs) != 0) + 1
    lobe_turning = np.add.reduceat(curvature * ds, np.r_[0, run_starts])
    kept = np.sign(lobe_turning[np.abs(lobe_turning) >= min_lobe_turning])
    return int(np.sum(kept[1:] != kept[:-1]))


def max_chord_deviation(points):
    """Largest perpendicular distance from any point to the end-to-end chord."""
    a, b = points[0], points[-1]
    chord = b - a
    norm = np.linalg.norm(chord)
    if norm == 0:
        return float(np.max(np.linalg.norm(points - a, axis=1)))
    cross = chord[0] * (points[:, 1] - a[1]) - chord[1] * (points[:, 0] - a[0])
    return float(np.max(np.abs(cross)) / norm)


def measure_polyline(points, is_closed=False, sigma=4.0, step=1.0,
                     min_lobe_turning=0.2, pixel_size=1.0, z=None, image_name=None, batch=None, condition=None, fish_id=None):
    """
    Compute shape descriptors for one ordered polyline.

    Parameters
    ----------
    points : (N, 2) ordered (row, col) coordinates in pixels.
    is_closed : treat as closed loop.
    sigma : Gaussian smoothing along the curve, in pixels (0 = none).
    step : resampling spacing in pixels.
    min_lobe_turning : minimum turning angle (radians) a curved stretch must
        have to count towards inflections (filters pixel noise).
    pixel_size : physical size of one pixel (e.g. um); lengths are scaled by
        it, curvatures by 1/pixel_size.

    Returns
    -------
    dict of descriptors (lengths in physical units, angles in radians).
    """
    raw = np.asarray(points, dtype=float)
    pts = resample_polyline(raw, step, is_closed)             # uniform spacing first
    pts = smooth_polyline(pts, sigma / step, is_closed)       # then smooth
    pts = resample_polyline(pts, step, is_closed)             # re-uniform after smoothing

    length = polyline_length(pts, is_closed)
    chord = 0.0 if is_closed else chord_length(pts)
    k = signed_curvature(pts, is_closed)
    ds = length / (len(pts) if is_closed else max(len(pts) - 1, 1))
    total_abs_turning = float(np.sum(np.abs(k)) * ds)
    net_turning = float(np.sum(k) * ds)
    direction = pts[-1] - pts[0]

    return {
        "is_closed": is_closed,
        "n_pixels": len(raw),
        "length_raw": polyline_length(raw, is_closed) * pixel_size,  # pixel-chain length
        "length": length * pixel_size,                                # smoothed length
        "chord": chord * pixel_size,
        "tortuosity": length / chord if chord > 0 else np.nan,       # arc / chord (>= 1)
        "straightness": chord / length if length > 0 else np.nan,    # chord / arc (0-1]
        "max_chord_deviation": np.nan if is_closed else max_chord_deviation(pts) * pixel_size,
        "mean_abs_curvature": float(np.mean(np.abs(k))) / pixel_size,
        "median_abs_curvature": float(np.median(np.abs(k))) / pixel_size,
        "max_abs_curvature": float(np.max(np.abs(k))) / pixel_size,
        "std_curvature": float(np.std(k)) / pixel_size,
        "total_abs_turning": total_abs_turning,                       # rad
        "net_turning": net_turning,                                   # rad, signed
        "soam": total_abs_turning / (length * pixel_size) if length > 0 else np.nan,  # rad / unit length
        "n_inflections": count_inflections(k, ds, min_lobe_turning),
        "orientation_deg": float(np.degrees(np.arctan2(-direction[0], direction[1])) % 180),
        "centroid_row": float(pts[:, 0].mean()),
        "centroid_col": float(pts[:, 1].mean()),
        "start_row": float(raw[0, 0]), "start_col": float(raw[0, 1]),
        "end_row": float(raw[-1, 0]), "end_col": float(raw[-1, 1])

    }

def filter_segments(label_img,segments):
    region_ids = [region_id for region_id, _, _ in segments]
    indicies = np.arange(np.max(label_img))
    keep_mask = np.isin(indicies,region_ids)
    keep_bool = keep_mask[label_img.reshape(-1)].reshape(label_img.shape)
    return label_img*keep_bool
    
# --------------------------------------------------------------------------
# Whole-mask entry point
# --------------------------------------------------------------------------

def extract_segments(mask, min_pixels=5,path=None,name=None):
    """
    Split a node-free skeleton mask into ordered polylines.

    Returns a list of (label_id, ordered_points, is_closed).
    """
    labels = label(np.asarray(mask, bool), connectivity=2)
    segments = []
    for region_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        coords = np.argwhere(labels[sl] == region_id) + [sl[0].start, sl[1].start]
        if len(coords) < min_pixels:
            continue
        ordered, is_closed = order_segment_pixels(coords)
        segments.append((region_id, ordered, is_closed))
    filt_seg = filter_segments(mask,segments)
    imsave(os.path.join(path,f'{name}_filtered_segs.tif'),filt_seg,check_contrast=False)
    return segments


def measure_skeleton_segments(mask, min_pixels=5, path=None, name=None, **measure_kwargs):
    """
    Measure every segment in a node-free skeleton mask.

    Parameters
    ----------
    mask : 2D bool array (skeleton, junction pixels removed).
    min_pixels : ignore segments with fewer pixels than this.
    **measure_kwargs : passed to measure_polyline (sigma, step, pixel_size, ...).

    Returns
    -------
    pandas.DataFrame with one row per segment, indexed by label id.
    """
    rows = []
    for region_id, points, is_closed in extract_segments(mask, min_pixels, path, name):
        row = measure_polyline(points, is_closed, **measure_kwargs)
        row["label"] = region_id
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame()
