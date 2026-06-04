"""Linearly transform coordinates (port of ggh4x ``position_lineartrans.R``).

This module ports the R ggh4x ``position_lineartrans()`` constructor and the
``PositionLinearTrans`` ggproto class.  The position adjustment applies a 2x2
linear transformation matrix ``M`` to the ``x``/``y`` coordinates of a layer.

The transformation matrix is either supplied directly (``M``) or assembled from
convenience arguments ``scale``, ``shear`` and ``angle`` in the fixed order
*scaling -> shearing -> rotating* (R ``position_lineartrans.R:18-23``).

R source: ``ggh4x/R/position_lineartrans.R``.

Notes
-----
* :meth:`PositionLinearTrans.setup_params` reconstructs R's **column-major**
  matrix builds exactly.  R's ``matrix(c(1, shear, 1), ncol = 2)`` with
  ``shear = c(s0, s1)`` fills column-by-column, giving ``[[1, s1], [s0, 1]]``;
  the rotation ``matrix(c(cos, sin, -sin, cos), ncol = 2)`` gives
  ``[[cos, -sin], [sin, cos]]``.  ``M * scale`` is an **elementwise column
  broadcast** (multiplying column ``j`` of ``M`` by ``scale[j]``), not a matrix
  product.
* :meth:`PositionLinearTrans.compute_layer` overrides the base ``compute_layer``
  wholesale (no per-``PANEL`` split), mirroring R: it applies the transform to
  every row at once via ``xy @ M.T`` (equivalent to R's ``t(M %*% t(coord))``).
* :meth:`PositionLinearTrans.setup_data` is an identity pass-through, suppressing
  the base required-aesthetic check (there are no required aesthetics).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ggplot2_py.position import Position

__all__ = [
    "position_lineartrans",
    "PositionLinearTrans",
]

ArrayLike = Union[Sequence[float], np.ndarray]


class PositionLinearTrans(Position):
    """Apply a 2x2 linear transformation to ``x``/``y`` coordinates.

    Subclass of :class:`ggplot2_py.position.Position` ported from R
    ``PositionLinearTrans`` (``position_lineartrans.R:115-152``).

    Attributes
    ----------
    scale : sequence of float or None
        Length-2 multipliers for the ``x`` and ``y`` coordinates respectively.
    shear : sequence of float or None
        Length-2 shear amounts.  The first number is the vertical shear, the
        second the horizontal shear.
    angle : float or None
        Clockwise rotation angle in degrees.
    M : array-like or None
        An explicit 2x2 transformation matrix.  When supplied it overrides
        ``scale``/``shear``/``angle``.
    """

    scale: Optional[ArrayLike] = (1, 1)
    shear: Optional[ArrayLike] = (0, 0)
    angle: Optional[float] = 0
    M: Optional[ArrayLike] = None

    def __init__(self, **kwargs: Any) -> None:
        """Store constructor members directly on the instance.

        Parameters
        ----------
        **kwargs : Any
            Members (``scale``, ``shear``, ``angle``, ``M``) assigned verbatim,
            mirroring the ``ggproto(NULL, PositionLinearTrans, ...)`` clone.
        """
        for k, v in kwargs.items():
            setattr(self, k, v)

    def setup_params(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Build the 2x2 transformation matrix ``M``.

        Port of R ``PositionLinearTrans$setup_params``
        (``position_lineartrans.R:127-151``).  When ``self.M`` is supplied it is
        returned verbatim.  Otherwise the matrix is assembled in the order
        scale -> shear -> rotate, reproducing R's column-major
        ``matrix(..., ncol = 2)`` fills exactly.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data (unused; present for signature parity).

        Returns
        -------
        dict
            ``{"M": ndarray}`` with a float64 2x2 transformation matrix.
        """
        if self.M is not None:
            return {"M": np.asarray(self.M, dtype="float64")}

        M = np.eye(2, dtype="float64")

        # Scale: R ``M <- M * self$scale`` recycles a length-2 ``scale`` down
        # the columns => multiply column j of M by scale[j].  Broadcasting a
        # row vector across rows does exactly this for a 2x2 M.
        if self.scale is not None:
            scale = np.asarray(self.scale, dtype="float64")
            M = M * scale

        # Shear: R ``matrix(c(1, shear, 1), ncol = 2)`` with shear = c(s0, s1)
        # fills column-major => column 0 = (1, s0), column 1 = (s1, 1), i.e.
        # [[1, s1], [s0, 1]].  Then M <- M %*% shear_matrix.
        if self.shear is not None:
            shear = np.asarray(self.shear, dtype="float64")
            if shear.shape == (2,):
                s0, s1 = float(shear[0]), float(shear[1])
                shear_mat = np.array([[1.0, s1], [s0, 1.0]], dtype="float64")
                M = M @ shear_mat

        # Rotation: theta = -angle * pi / 180; R builds
        # matrix(c(cos, sin, -sin, cos), ncol = 2) (column-major) =>
        # [[cos, -sin], [sin, cos]]; then M <- rotation %*% M.
        if self.angle is not None:
            theta = -float(self.angle) * np.pi / 180.0
            c, s = np.cos(theta), np.sin(theta)
            rot = np.array([[c, -s], [s, c]], dtype="float64")
            M = rot @ M

        return {"M": M}

    def setup_data(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """Return *data* unchanged (identity pass-through).

        Port of R ``PositionLinearTrans$setup_data``
        (``position_lineartrans.R:124-126``).  Overriding the base
        :meth:`ggplot2_py.position.Position.setup_data` suppresses the base
        required-aesthetic check; there are no required aesthetics here.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data.
        params : dict
            Position parameters (unused).

        Returns
        -------
        pandas.DataFrame
            *data* unchanged.
        """
        return data

    def compute_layer(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any],
        layout: Any,
    ) -> pd.DataFrame:
        """Apply the linear transform to every row's ``x``/``y``.

        Port of R ``PositionLinearTrans$compute_layer``
        (``position_lineartrans.R:117-123``):
        ``coord <- t(params$M %*% t(coord))`` which equals ``xy @ M.T``.  This
        override bypasses the base per-``PANEL`` split, transforming all rows at
        once exactly as R does.

        Parameters
        ----------
        data : pandas.DataFrame
            Layer data containing ``x`` and ``y`` columns.
        params : dict
            Position parameters carrying the transformation matrix ``M``.
        layout : Any
            Plot layout (unused; present for signature parity).

        Returns
        -------
        pandas.DataFrame
            *data* with ``x``/``y`` replaced by the transformed coordinates.
        """
        M = np.asarray(params["M"], dtype="float64")
        xy = data[["x", "y"]].to_numpy(dtype="float64")
        out = xy @ M.T  # == (M @ xy.T).T == R's t(M %*% t(coord))
        data = data.copy()
        data["x"] = out[:, 0]
        data["y"] = out[:, 1]
        return data


def position_lineartrans(
    scale: Optional[ArrayLike] = (1, 1),
    shear: Optional[ArrayLike] = (0, 0),
    angle: float = 0,
    M: Optional[ArrayLike] = None,
) -> PositionLinearTrans:
    """Create a linear-transformation position adjustment.

    Port of R ``position_lineartrans()`` (``position_lineartrans.R:102-107``).
    Transforms ``x``/``y`` coordinates in two dimensions for layers with an
    ``x``/``y`` parametrisation.

    Linear transformation matrices are 2x2 real matrices.  ``scale``, ``shear``
    and ``angle`` are convenience arguments combined in the order
    *scaling -> shearing -> rotating*.  To apply transformations in another
    order, build a custom ``M``.

    Parameters
    ----------
    scale : sequence of float, optional
        Length-2 relative units multiplying the ``x`` and ``y`` coordinates
        respectively.  Default ``(1, 1)``.
    shear : sequence of float, optional
        Length-2 relative shear units.  The first number shears vertically, the
        second horizontally.  Default ``(0, 0)``.
    angle : float, optional
        Angle in degrees by which to rotate the input clockwise.  Default ``0``.
    M : array-like, optional
        A 2x2 real transformation matrix.  Overrides ``scale``/``shear``/
        ``angle`` when provided.

    Returns
    -------
    PositionLinearTrans
        A position object that can be passed to a layer's ``position`` argument.

    Examples
    --------
    >>> position_lineartrans(angle=30)  # doctest: +ELLIPSIS
    <ggh4x.position_lineartrans.PositionLinearTrans object at ...>
    """
    return PositionLinearTrans(
        scale=scale,
        shear=shear,
        angle=angle,
        M=M,
    )
