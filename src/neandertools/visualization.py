"""Visualization helpers for image collections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def cutouts_grid(
    images: Sequence[Any],
    ncols: int = 5,
    titles: Sequence[str] | None = None,
    figsize_per_cell: tuple[float, float] = (3.2, 3.2),
    qmin: float = 0.0,
    qmax: float = 0.99,
    match_background: bool = True,
    match_noise: bool = False,
    sigma_clip: float = 3.0,
    sigma_clip_iters: int = 5,
    add_colorbar: bool = False,
    cmap: str = "gray_r",
    show: bool = True,
):
    """Display images in a grid with linear quantile normalization.

    Parameters
    ----------
    images : sequence
        Sequence of image-like objects. Supported forms are LSST-like objects
        exposing ``obj.image.array`` and array-like objects exposing ``obj.array``.
    ncols : int, optional
        Number of columns in the grid.
    titles : sequence of str, optional
        Optional per-image titles.
    figsize_per_cell : tuple of float, optional
        Width and height per subplot cell.
    qmin : float, optional
        Lower quantile used for ``vmin`` (NaN-aware).
    qmax : float, optional
        Upper quantile used for ``vmax`` (NaN-aware).
    match_background : bool, optional
        If ``True``, subtract a robust sigma-clipped background estimate from
        each cutout before plotting.
    match_noise : bool, optional
        If ``True``, divide each cutout by its robust background RMS estimate
        after background subtraction.
    sigma_clip : float, optional
        Sigma threshold used for iterative clipping when estimating per-cutout
        background/noise.
    sigma_clip_iters : int, optional
        Maximum number of sigma-clipping iterations.
    add_colorbar : bool, optional
        If ``True``, draw one colorbar per subplot.
    cmap : str, optional
        Matplotlib colormap name.
    show : bool, optional
        If ``True``, call ``plt.show()`` before returning.

    Returns
    -------
    tuple
        ``(fig, axes)`` from matplotlib.
    """
    n = len(images)
    if n == 0:
        raise ValueError("No images provided.")
    if not (0.0 <= qmin <= 1.0 and 0.0 <= qmax <= 1.0):
        raise ValueError("qmin and qmax must be in [0, 1]")
    if qmax < qmin:
        raise ValueError("qmax must be >= qmin")
    if sigma_clip <= 0:
        raise ValueError("sigma_clip must be > 0")
    if sigma_clip_iters < 1:
        raise ValueError("sigma_clip_iters must be >= 1")

    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
    )

    arrays = []
    for obj in images:
        if hasattr(obj, "image"):
            arr = np.asarray(obj.image.array)
        else:
            arr = np.asarray(obj.array)
        arrays.append(arr)

    proc_arrays = []
    for arr in arrays:
        if match_background or match_noise:
            bg, rms = _sigma_clipped_bg_rms(arr, sigma=sigma_clip, maxiters=sigma_clip_iters)
            arr_proc = arr.astype(np.float32, copy=True)
            if match_background:
                arr_proc = arr_proc - bg
            if match_noise:
                arr_proc = arr_proc / max(rms, 1e-12)
            proc_arrays.append(arr_proc)
        else:
            proc_arrays.append(arr)

    shared_scale = match_background or match_noise
    shared_vmin: float | None = None
    shared_vmax: float | None = None
    if shared_scale:
        finite_parts = [arr[np.isfinite(arr)] for arr in proc_arrays if np.any(np.isfinite(arr))]
        if not finite_parts:
            raise ValueError("No finite pixels available to determine display scale.")
        all_values = np.concatenate(finite_parts)
        shared_vmin = float(np.quantile(all_values, qmin))
        shared_vmax = float(np.quantile(all_values, qmax))
        if shared_vmax <= shared_vmin:
            shared_vmax = shared_vmin + 1e-12

    for i, arr in enumerate(proc_arrays):
        r, c = divmod(i, ncols)
        ax = axes[r][c]
        if shared_scale:
            assert shared_vmin is not None and shared_vmax is not None
            vmin = shared_vmin
            vmax = shared_vmax
        else:
            vmin = np.nanquantile(arr, qmin)
            vmax = np.nanquantile(arr, qmax)
            if vmax <= vmin:
                vmax = vmin + 1e-12

        im = ax.imshow(
            arr,
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            interpolation="nearest",
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        if titles is not None:
            ax.set_title(titles[i], fontsize=10)

        if add_colorbar:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(n, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    if show:
        plt.show()

    return fig, axes


def _sigma_clipped_bg_rms(arr: np.ndarray, sigma: float, maxiters: int) -> tuple[float, float]:
    values = np.asarray(arr, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0

    clipped = values
    for _ in range(maxiters):
        med = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - med)))
        rms = 1.4826 * mad
        if not np.isfinite(rms) or rms <= 0:
            break
        keep = np.abs(clipped - med) <= sigma * rms
        if keep.all() or keep.sum() == 0:
            break
        clipped = clipped[keep]

    bg = float(np.median(clipped))
    mad = float(np.median(np.abs(clipped - bg)))
    rms = 1.4826 * mad
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.std(clipped))
    if not np.isfinite(rms) or rms <= 0:
        rms = 1.0
    return bg, rms
