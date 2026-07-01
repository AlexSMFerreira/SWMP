"""
Stereo rectifier with REFINED extrinsics — experimental, drop-in replacement.

Identical to ros2_stereo_rectifier.py in every way except the stereo rotation
R_stereo, which is refined from the recorded imagery (Scripts/refine_stereo_calib.py):
a small corrective rotation (~0.1-0.2 deg) is applied to the original calibration
to reduce the residual vertical epipolar error. Measured on the recorded bag, this
lowers the systematic vertical disparity of stereo correspondences from ~14 px to
~5 px at native resolution (~3.6 -> ~1.3 px at the 0.25 output scale, the latter at
the SIFT noise floor). T_stereo is unchanged (the error was purely rotational; the
baseline/depth scale is preserved, so downstream point-cloud depths and wave heights
are unaffected beyond the tiny alignment correction).

This is deliberately a SEPARATE node so the original, validated rectifier is left
untouched: to A/B test, run this instead of ros2_stereo_rectifier.py in
start_pipeline.sh (window `rectify`); to revert, switch back — no other change needed.
Both publish the same node name and topics, so exactly one must run at a time.
"""

import numpy as np
import rclpy

from ros2_stereo_rectifier import RectifyNode  # original node, unchanged


# Original R_stereo (kept here for reference / easy revert):
#   [[ 0.99998433,  0.00309469,  0.00466459],
#    [-0.00307396,  0.9999854,  -0.0044443 ],
#    [-0.00467827,  0.00442989,  0.99997924]]
# Refined (corrective rotation ~[-0.194, 0.006, 0.105] deg applied to the original):
R_STEREO_REFINED = np.array([
    [ 0.99998540,  0.00106639,  0.00529683],
    [-0.00106084,  0.99999889, -0.00105004],
    [-0.00529794,  0.00104441,  0.99998542],
])


class RefinedRectifyNode(RectifyNode):
    def __init__(self):
        super().__init__()
        # Override the extrinsic rotation before any CameraInfo arrives (the
        # rectification maps are built lazily in the CameraInfo callback).
        self.R_stereo = R_STEREO_REFINED
        self.get_logger().warn(
            'Using REFINED stereo extrinsics (ros2_stereo_rectifier_refined.py) — '
            'experimental. Revert to ros2_stereo_rectifier.py if needed.')


def main(args=None):
    rclpy.init(args=args)
    node = RefinedRectifyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
