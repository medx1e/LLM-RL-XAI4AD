# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""General visualization functions for non-waymax data using matplotlib.

Note there is no batch dimension, and should not rely on any customized data
structure.
"""

import matplotlib as mpl
import numpy as np
from matplotlib.patches import Polygon


def plot_numpy_bounding_boxes(
    ax: mpl.axes.Axes,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha: float | None = 1.0,
    as_center_pts: bool = False,
    overlap: bool = False,
) -> None:
    """Plot multiple bounding boxes on an axis.

    Draws bounding boxes either as full rectangles or as center points and supports an overlap mode
    where only borders are drawn.

    Args:
        ax: Matplotlib axis used for plotting.
        bboxes: Array of bounding boxes with shape (num_bbox, 5).
        color: RGB color array of size 3.
        alpha: Alpha transparency of the boxes.
        as_center_pts: If True, draws only center points.
        overlap: If True, draws boxes in overlap mode.

    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5 or color.shape != (3,):
        raise ValueError(
            "Expect bboxes rank 2, last dimension of bbox 5, color of size 3,"
            f" got{bboxes.ndim}, {bboxes.shape[1]}, {color.shape} respectively",
        )

    if as_center_pts:
        ax.plot(bboxes[:, 0], bboxes[:, 1], "o", color=color, ms=4, alpha=alpha, zorder=4)
    else:
        c = np.cos(bboxes[:, 4])
        s = np.sin(bboxes[:, 4])
        pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
        length, width = bboxes[:, 2], bboxes[:, 3]
        u = np.array((c, s))
        ut = np.array((s, -c))

        # Compute box corner coordinates.
        tl = pt + length / 2 * u - width / 2 * ut
        tr = pt + length / 2 * u + width / 2 * ut
        br = pt - length / 2 * u + width / 2 * ut
        bl = pt - length / 2 * u - width / 2 * ut

        if overlap:
            ax.plot(
                [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
                [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
                color=color,
                zorder=4,
                alpha=alpha,
            )
        else:
            # Rectangles
            car_patch = Polygon(
                np.concatenate((tl.T, bl.T, br.T, tr.T)),
                alpha=alpha,
                facecolor=color,
                edgecolor=color,
                lw=2,
                zorder=4,
            )
            ax.add_patch(car_patch)
