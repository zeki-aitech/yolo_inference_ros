from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np

from sensor_msgs.msg import CameraInfo


@dataclass
class BoundingBox3DData:
    """Pure Python data structure to hold 3D bounding box results"""
    x: float
    y: float
    z: float
    w: float
    h: float
    d: float  # depth/thickness

class DepthProcessor:
    def __init__(self, depth_image_units_divisor: int = 1000):
        self.depth_image_units_divisor = depth_image_units_divisor

    def convert_to_3d_bbox(
        self, 
        depth_image: np.ndarray, 
        depth_info: CameraInfo,
        center_x: float, center_y: float, size_x: float, size_y: float
    ) -> Optional[BoundingBox3DData]:

        u_min = max(int(center_x - size_x // 2), 0)
        u_max = min(int(center_x + size_x // 2), depth_image.shape[1] - 1)
        v_min = max(int(center_y - size_y // 2), 0)
        v_max = min(int(center_y + size_y // 2), depth_image.shape[0] - 1)

        if u_max <= u_min or v_max <= v_min:
            return None

        depth_roi = depth_image[v_min:v_max, u_min:u_max]

        roi_h, roi_w = depth_roi.shape
        y_grid, x_grid = np.meshgrid(
            np.arange(roi_h) + v_min, np.arange(roi_w) + u_min, indexing="ij"
        )
        pixel_coords = np.column_stack([x_grid.flatten(), y_grid.flatten()])                    

        if not np.any(np.isfinite(depth_roi)) or not np.any(depth_roi):
            return None    

        valid_depths = depth_roi.flatten()

        try:
            if depth_image.dtype.kind in ['u', 'i']: # u=uint16, i=int16 (Millimeters)
                valid_depths = np.asarray(valid_depths, dtype=np.float64) / float(self.depth_image_units_divisor)
            else: # Float32 (Meters)
                valid_depths = np.asarray(valid_depths, dtype=np.float64)
        except (ValueError, TypeError):
            return None

        valid_mask = (valid_depths > 0) & np.isfinite(valid_depths)
        valid_depths = valid_depths[valid_mask]
        valid_coords = pixel_coords[valid_mask] 

        if len(valid_depths) == 0:
            return None
        
        spatial_weights = self._compute_spatial_weights(
            valid_coords, center_x, center_y, size_x, size_y
        )

        z_center, z_min, z_max = self._compute_depth_bounds_weighted(
            valid_depths, spatial_weights
        )

        if not np.isfinite(z_center) or z_center == 0:
            return None

        y_center, y_min, y_max = self._compute_height_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info
        )

        if not all(np.isfinite([y_center, y_min, y_max])):
            return None

        x_center, x_min, x_max = self._compute_width_bounds(
            valid_coords, valid_depths, spatial_weights, depth_info
        )

        if not all(np.isfinite([x_center, x_min, x_max])):
            return None
        
        return BoundingBox3DData(
            x=x_center, 
            y=y_center, 
            z=z_center, 
            w=float(x_max - x_min), 
            h=float(y_max - y_min), 
            d=float(z_max - z_min)
        )

    @staticmethod
    def _compute_spatial_weights(
        coords: np.ndarray, center_x: int, center_y: int, size_x: int, size_y: int
    ) -> np.ndarray:
        """
        Compute spatial weights for depth values based on distance from 2D bbox center.
        Pixels near the center get higher weight to handle occlusions better.

        Args:
            coords: Nx2 array of pixel coordinates [x, y]
            center_x: X coordinate of bbox center
            center_y: Y coordinate of bbox center
            size_x: Width of bbox
            size_y: Height of bbox

        Returns:
            Array of weights (0-1) for each coordinate
        """
        # Compute normalized distance from center
        dx = (coords[:, 0] - center_x) / (size_x / 2 + 1e-6)
        dy = (coords[:, 1] - center_y) / (size_y / 2 + 1e-6)
        normalized_dist = np.sqrt(dx**2 + dy**2)

        # Use Gaussian-like weighting: higher weight at center, lower at edges
        # sigma = 0.8 means ~80% of bbox radius has high weight
        weights = np.exp(-0.5 * (normalized_dist / 0.8) ** 2)

        # Ensure minimum weight of 0.3 to not completely ignore edge pixels
        weights = np.maximum(weights, 0.3)

        return weights

    @staticmethod
    def _compute_height_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D height (y-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (y_center, y_min, y_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            py, fy = k[5], k[4]

            # Validate camera parameters
            if fy == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            y_coords_pixel = valid_coords[:, 1]
            y_3d = valid_depths * (y_coords_pixel - py) / fy

            # Validate result
            if not np.all(np.isfinite(y_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(y_3d)), float(np.min(y_3d)), float(np.max(y_3d))

        # Convert pixel coordinates to 3D y-coordinates
        k = depth_info.k
        py, fy = k[5], k[4]

        # Validate camera parameters
        if fy == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        y_coords_pixel = valid_coords[:, 1]
        y_3d = valid_depths * (y_coords_pixel - py) / fy

        # Validate result
        if not np.any(np.isfinite(y_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(y_3d)
        sorted_y = y_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        y_median = sorted_y[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(y_3d - y_median)
        mad = np.median(deviations)

        # Filter outliers: keep points within 4.5*MAD from median
        # Balanced threshold to handle tall objects while avoiding background
        threshold = np.clip(4.5 * mad, 0.06, 0.50)
        valid_mask = deviations <= threshold
        filtered_y = y_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_y) < max(4, len(y_3d) * 0.12):
            filtered_y = y_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_y = sorted_y[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                y_center = np.average(trimmed_y, weights=trimmed_weights)
            else:
                y_center = np.median(filtered_y)
        else:
            y_center = np.median(filtered_y)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_y)
        sorted_y = filtered_y[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        y_min = sorted_y[p3_idx]
        y_max = sorted_y[p97_idx]

        # Ensure minimum height of 2cm
        min_height = 0.02
        if (y_max - y_min) < min_height:
            half_min = min_height / 2
            y_min = y_center - half_min
            y_max = y_center + half_min

        return float(y_center), float(y_min), float(y_max)

    @staticmethod
    def _compute_width_bounds(
        valid_coords: np.ndarray,
        valid_depths: np.ndarray,
        spatial_weights: np.ndarray,
        depth_info: CameraInfo,
    ) -> Tuple[float, float, float]:
        """
        Compute 3D width (x-axis) statistics from valid depth points.
        Uses actual 3D point positions instead of just projecting 2D bbox.

        Args:
            valid_coords: Nx2 array of pixel coordinates [x, y]
            valid_depths: N array of depth values in meters
            spatial_weights: N array of spatial weights
            depth_info: Camera intrinsic parameters

        Returns:
            Tuple of (x_center, x_min, x_max) in meters
        """
        # Input validations
        try:
            valid_depths = np.asarray(valid_depths, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(valid_coords) == 0 or len(valid_depths) == 0:
            return 0.0, 0.0, 0.0

        if len(valid_coords) < 4:
            # Fallback: just use simple projection
            k = depth_info.k
            px, fx = k[2], k[0]

            # Validate camera parameters
            if fx == 0:
                return 0.0, 0.0, 0.0

            # Validate depths are finite
            if not np.all(np.isfinite(valid_depths)):
                return 0.0, 0.0, 0.0

            x_coords_pixel = valid_coords[:, 0]
            x_3d = valid_depths * (x_coords_pixel - px) / fx

            # Validate result
            if not np.all(np.isfinite(x_3d)):
                return 0.0, 0.0, 0.0

            return float(np.median(x_3d)), float(np.min(x_3d)), float(np.max(x_3d))

        # Convert pixel coordinates to 3D x-coordinates
        k = depth_info.k
        px, fx = k[2], k[0]

        # Validate camera parameters
        if fx == 0:
            return 0.0, 0.0, 0.0

        # Validate depths are finite before calculation
        if not np.all(np.isfinite(valid_depths)):
            return 0.0, 0.0, 0.0

        x_coords_pixel = valid_coords[:, 0]
        x_3d = valid_depths * (x_coords_pixel - px) / fx

        # Validate result
        if not np.any(np.isfinite(x_3d)):
            return 0.0, 0.0, 0.0

        # Filter outliers using robust statistics
        # Compute weighted median as reference
        sorted_idx = np.argsort(x_3d)
        sorted_x = x_3d[sorted_idx]
        sorted_weights = spatial_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0
        median_idx = np.searchsorted(cumsum_weights, 0.5)
        x_median = sorted_x[median_idx]

        # Compute MAD (Median Absolute Deviation)
        deviations = np.abs(x_3d - x_median)
        mad = np.median(deviations)

        # Adaptive threshold based on depth variance (helps with occlusions)
        # Check if object has varying depth (might indicate occlusion)
        depth_std = np.std(valid_depths)
        if depth_std > 0.15:  # High depth variation - likely occlusion or 3D object
            # Use tighter threshold to avoid including background
            threshold = np.clip(4.0 * mad, 0.06, 0.40)
        else:  # Uniform depth - flat object
            # Can be more permissive
            threshold = np.clip(4.5 * mad, 0.08, 0.50)

        valid_mask = deviations <= threshold
        filtered_x = x_3d[valid_mask]
        filtered_weights = spatial_weights[valid_mask]

        # Ensure we have enough points (at least 12% of data)
        if len(filtered_x) < max(4, len(x_3d) * 0.12):
            filtered_x = x_3d
            filtered_weights = spatial_weights

        # Compute weighted center using trimmed mean
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Trim 5% from each end for robust center estimation
        trim_low_idx = np.searchsorted(cumsum_weights, 0.05)
        trim_high_idx = np.searchsorted(cumsum_weights, 0.95)

        if trim_high_idx > trim_low_idx:
            trimmed_x = sorted_x[trim_low_idx:trim_high_idx]
            trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]
            if np.sum(trimmed_weights) > 0:
                x_center = np.average(trimmed_x, weights=trimmed_weights)
            else:
                x_center = np.median(filtered_x)
        else:
            x_center = np.median(filtered_x)

        # Compute extent using balanced percentiles (3rd and 97th)
        # Good balance between capturing object extent and avoiding outliers
        sorted_idx = np.argsort(filtered_x)
        sorted_x = filtered_x[sorted_idx]
        sorted_weights = filtered_weights[sorted_idx]
        cumsum_weights = np.cumsum(sorted_weights)
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        p3_idx = np.searchsorted(cumsum_weights, 0.03)
        p97_idx = np.searchsorted(cumsum_weights, 0.97)

        x_min = sorted_x[p3_idx]
        x_max = sorted_x[p97_idx]

        # Ensure minimum width of 2cm
        min_width = 0.02
        if (x_max - x_min) < min_width:
            half_min = min_width / 2
            x_min = x_center - half_min
            x_max = x_center + half_min

        return float(x_center), float(x_min), float(x_max)

    @staticmethod
    def _compute_depth_bounds_weighted(
        depth_values: np.ndarray, spatial_weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute robust depth statistics with spatial weighting to handle occlusions.

        Args:
            depth_values: 1D array of valid depth values (> 0)
            spatial_weights: 1D array of spatial weights (0-1) for each depth

        Returns:
            Tuple of (z_center, z_min, z_max) representing the object's depth
        """
        # Input validations
        try:
            depth_values = np.asarray(depth_values, dtype=np.float64)
            spatial_weights = np.asarray(spatial_weights, dtype=np.float64)
        except (ValueError, TypeError):
            return 0.0, 0.0, 0.0

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        # Validate that all values are finite
        valid_mask = np.isfinite(depth_values) & np.isfinite(spatial_weights)
        depth_values = depth_values[valid_mask]
        spatial_weights = spatial_weights[valid_mask]

        if len(depth_values) == 0:
            return 0.0, 0.0, 0.0

        if len(depth_values) < 4:
            z_center = float(np.median(depth_values))
            return z_center, float(np.min(depth_values)), float(np.max(depth_values))

        # Step 1: Multi-scale histogram analysis for robust mode detection
        depth_range = np.ptp(depth_values)
        if not np.isfinite(depth_range) or depth_range <= 0:
            n_bins = 30
        else:
            n_bins = max(20, min(60, int(depth_range / 0.01)))

        # Create weighted histogram
        hist, bin_edges = np.histogram(depth_values, bins=n_bins, weights=spatial_weights)

        # Smooth histogram to reduce noise while preserving peaks
        if len(hist) >= 5:
            # Simple moving average smoothing
            kernel_size = min(5, len(hist) // 4)
            kernel = np.ones(kernel_size) / kernel_size
            hist_smooth = np.convolve(hist, kernel, mode="same")
        else:
            hist_smooth = hist

        # Find peak (mode) - highest weighted density region
        peak_bin_idx = np.argmax(hist_smooth)
        mode_depth = (bin_edges[peak_bin_idx] + bin_edges[peak_bin_idx + 1]) / 2

        # Step 2: Adaptive outlier filtering with less aggressive thresholds
        deviations = np.abs(depth_values - mode_depth)

        # Compute robust MAD without inverse weighting to avoid over-filtering
        mad = np.median(deviations)

        # More permissive threshold - adjust based on object size and uniformity
        # Check depth distribution uniformity
        q25 = np.percentile(depth_values, 25)
        q75 = np.percentile(depth_values, 75)
        iqr = q75 - q25

        # Adaptive threshold: looser for varied depth, tighter for uniform
        if iqr < 0.03:  # Very uniform depth (<3cm IQR)
            # For flat objects, use tighter bounds
            threshold = np.clip(3.5 * mad, 0.08, 0.30)
        elif iqr < 0.10:  # Moderate variation (<10cm IQR)
            # Standard threshold
            threshold = np.clip(4.0 * mad, 0.12, 0.40)
        else:  # High variation (>10cm IQR)
            # For complex 3D objects, use very permissive bounds
            threshold = np.clip(5.0 * mad, 0.15, 0.60)

        # Keep depths within threshold
        object_mask = deviations <= threshold
        object_depths = depth_values[object_mask]
        object_weights = spatial_weights[object_mask]

        # Fallback if filtering was too aggressive
        min_points = max(6, int(len(depth_values) * 0.15))  # Keep at least 15% of points
        if len(object_depths) < min_points:
            # Use weighted percentiles with wider range
            sorted_idx = np.argsort(depth_values)
            cumsum_weights = np.cumsum(spatial_weights[sorted_idx])
            cumsum_weights /= cumsum_weights[-1]

            # Find 2nd and 85th weighted percentiles (wider range)
            p2_idx = np.searchsorted(cumsum_weights, 0.02)
            p85_idx = np.searchsorted(cumsum_weights, 0.85)

            p2_val = depth_values[sorted_idx[p2_idx]]
            p85_val = depth_values[sorted_idx[p85_idx]]

            object_mask = (depth_values >= p2_val) & (depth_values <= p85_val)
            object_depths = depth_values[object_mask]
            object_weights = spatial_weights[object_mask]

        if len(object_depths) == 0:
            object_depths = depth_values
            object_weights = spatial_weights

        # Step 3: Compute robust weighted center using trimmed mean
        if np.sum(object_weights) > 0:
            # Use weighted average, but trim extreme 2% on each side first
            sorted_idx = np.argsort(object_depths)
            sorted_depths = object_depths[sorted_idx]
            sorted_weights = object_weights[sorted_idx]

            cumsum_weights = np.cumsum(sorted_weights)
            cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

            # Trim 2% from each end
            trim_low_idx = np.searchsorted(cumsum_weights, 0.02)
            trim_high_idx = np.searchsorted(cumsum_weights, 0.98)

            if trim_high_idx > trim_low_idx:
                trimmed_depths = sorted_depths[trim_low_idx:trim_high_idx]
                trimmed_weights = sorted_weights[trim_low_idx:trim_high_idx]

                if np.sum(trimmed_weights) > 0:
                    z_center = np.average(trimmed_depths, weights=trimmed_weights)
                else:
                    z_center = np.median(object_depths)
            else:
                z_center = np.average(object_depths, weights=object_weights)
        else:
            z_center = np.median(object_depths)

        # Step 4: Compute extent using balanced weighted percentiles
        sorted_idx = np.argsort(object_depths)
        cumsum_weights = np.cumsum(object_weights[sorted_idx])
        cumsum_weights /= cumsum_weights[-1] if cumsum_weights[-1] > 0 else 1.0

        # Use 1st and 99th percentiles for depth (slightly more coverage than width/height)
        p1_idx = np.searchsorted(cumsum_weights, 0.01)
        p99_idx = np.searchsorted(cumsum_weights, 0.99)

        z_min = object_depths[sorted_idx[p1_idx]]
        z_max = object_depths[sorted_idx[p99_idx]]

        # Validate and adjust bounds relative to center
        # Ensure center is within bounds (sanity check)
        if z_center < z_min or z_center > z_max:
            # Recompute bounds symmetrically around center
            depth_extent = max(z_max - z_min, 0.02)  # At least 2cm
            z_min = z_center - depth_extent / 2
            z_max = z_center + depth_extent / 2

        # Ensure minimum depth size of 2cm (more realistic for real objects)
        min_depth_size = 0.02
        if (z_max - z_min) < min_depth_size:
            # Expand around center
            half_min = min_depth_size / 2
            z_min = z_center - half_min
            z_max = z_center + half_min

        return float(z_center), float(z_min), float(z_max)