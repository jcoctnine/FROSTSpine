"""Spine mask cleanup and axis-aligned crop geometry."""
from dataclasses import dataclass
import math
from typing import Sequence
import numpy as np
Point = tuple[float, float]

@dataclass(frozen=True)
class SpineBox:
    center: Point
    corners: tuple[Point, Point, Point, Point]
    aabb_xyxy: tuple[float, float, float, float]

def largest_component(mask: np.ndarray) -> np.ndarray:
    from skimage.measure import label, regionprops

    labels = label(mask.astype(bool), connectivity=2)
    if labels.max() == 0:
        return mask.astype(bool)
    region = max(regionprops(labels), key=lambda item: item.area)
    return labels == region.label

def clean_mask(
    mask: np.ndarray,
    min_object_size: int,
    close_radius: int,
    dilate_radius: int,
) -> np.ndarray:
    from scipy.ndimage import binary_fill_holes
    from skimage.morphology import binary_closing, binary_dilation, disk, remove_small_objects

    result = mask.astype(bool)
    if min_object_size > 0:
        result = remove_small_objects(result, min_size=min_object_size)
    result = largest_component(result)
    if close_radius > 0:
        result = binary_closing(result, disk(close_radius))
    result = binary_fill_holes(result)
    if dilate_radius > 0:
        result = binary_dilation(result, disk(dilate_radius))
    return largest_component(result)

def centerline_from_mask(
    mask: np.ndarray,
    row_step: int,
    smooth_window: int,
    min_row_pixels: int,
) -> list[Point]:
    """以每个采样行中前景像素的 x 中位数估计中心线。"""
    foreground_rows = np.flatnonzero(mask.any(axis=1))
    if foreground_rows.size == 0:
        return []

    y_min, y_max = int(foreground_rows.min()), int(foreground_rows.max())
    points: list[Point] = []
    for y in range(y_min, y_max + 1, max(1, row_step)):
        xs = np.flatnonzero(mask[y])
        if xs.size >= min_row_pixels:
            points.append((float(np.median(xs)), float(y)))

    if len(points) < 2:
        _, all_xs = np.where(mask)
        if all_xs.size == 0:
            return []
        x = float(np.median(all_xs))
        return [(x, float(y_min)), (x, float(y_max))]

    window = max(1, smooth_window)
    if window > 1 and len(points) >= 3:
        if window % 2 == 0:
            window += 1
        half = window // 2
        xs = np.asarray([point[0] for point in points])
        padded = np.pad(xs, (half, half), mode="edge")
        smooth_xs = [float(padded[i : i + window].mean()) for i in range(len(points))]
        points = [(x, y) for x, (_, y) in zip(smooth_xs, points)]
    return points

def interpolate_x_at_y(points: Sequence[Point], query_y: float) -> float:
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        if abs(y1 - y0) < 1e-9:
            continue
        if (query_y - y0) * (query_y - y1) <= 0:
            ratio = (query_y - y0) / (y1 - y0)
            return float(x0 + ratio * (x1 - x0))
    return float(min(points, key=lambda point: abs(point[1] - query_y))[0])

def make_box(center: Point, height: float, width: float) -> SpineBox:
    cx, cy = center
    half_width, half_height = width / 2.0, height / 2.0
    left, top = cx - half_width, cy - half_height
    right, bottom = cx + half_width, cy + half_height
    return SpineBox(
        center=(float(cx), float(cy)),
        corners=((left, top), (right, top), (right, bottom), (left, bottom)),
        aabb_xyxy=(left, top, right, bottom),
    )

def generate_boxes(
    points: Sequence[Point], height: float, width: float, step: float,
    offset: float = 0.0,
) -> list[SpineBox]:
    if len(points) < 2:
        return []
    y_min = min(point[1] for point in points)
    y_max = max(point[1] for point in points)
    start_y, end_y = y_min + height / 2.0 + offset, y_max - height / 2.0
    if end_y < start_y:
        return []
    return [
        make_box((interpolate_x_at_y(points, float(y)), float(y)), height, width)
        for y in np.arange(start_y, end_y + 1e-9, step)
    ]

def crop_bounds(box: SpineBox, image_width: int, image_height: int) -> tuple[int, ...]:
    left, top, right, bottom = box.aabb_xyxy
    return (
        max(0, min(math.floor(left), image_width)),
        max(0, min(math.floor(top), image_height)),
        max(0, min(math.ceil(right), image_width)),
        max(0, min(math.ceil(bottom), image_height)),
    )
