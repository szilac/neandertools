"""Butler-backed cutout service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Iterable, Optional, Union

from .errors import MissingDependencyError

DataId = dict[str, Any]
SkyResolver = Callable[[float, float, Optional[Union[datetime, str]]], Iterable[DataId]]


@dataclass(frozen=True)
class CutoutSpec:
    """Specification for a square cutout.

    Parameters
    ----------
    radius : float
        Radius in pixels. The implementation creates a square with side
        ``2 * radius + 1`` around the center.
    """

    radius: float


class ButlerCutoutService:
    """Generate cutouts from an LSST Butler repository."""

    def __init__(
        self,
        butler: Any,
        dataset_type: str = "calexp",
        sky_resolver: Optional[SkyResolver] = None,
    ) -> None:
        self._butler = butler
        self._dataset_type = dataset_type
        self._sky_resolver = sky_resolver

    def cutout(
        self,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        time: Optional[Union[datetime, str]] = None,
        radius: float = 10.0,
        *,
        visit: Optional[int] = None,
        detector: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Any]:
        """Return a list of cutout images.

        Two call styles are supported:

        - ``cutout(ra=..., dec=..., time=..., radius=...)``
        - ``cutout(visit=..., detector=..., radius=...)``
        """
        _validate_request(ra=ra, dec=dec, radius=radius, visit=visit, detector=detector)

        spec = CutoutSpec(radius=radius)

        if visit is not None:
            ref = self._make_ref({"visit": visit, "detector": detector})
            return [self._extract_cutout(ref, spec)]

        assert ra is not None and dec is not None  # validated above
        if self._sky_resolver is None:
            raise NotImplementedError(
                "Sky-position cutouts require a sky_resolver. "
                "Pass one to cutouts_from_butler(..., sky_resolver=...)."
            )

        data_ids = list(self._sky_resolver(ra, dec, time))
        if limit is not None:
            data_ids = data_ids[:limit]

        return [self._extract_cutout(self._make_ref(data_id), spec) for data_id in data_ids]

    def _make_ref(self, data_id: DataId) -> Any:
        return {"datasetType": self._dataset_type, "dataId": data_id}

    def _extract_cutout(self, ref: Any, spec: CutoutSpec) -> Any:
        image = self._butler.get(ref["datasetType"], dataId=ref["dataId"])

        # Works for afw Exposure-like types that provide getBBox/Factory,
        # otherwise returns the original image object.
        if not hasattr(image, "getBBox") or not hasattr(image, "Factory"):
            return image

        bbox = image.getBBox()
        x_center = (bbox.getMinX() + bbox.getMaxX()) // 2
        y_center = (bbox.getMinY() + bbox.getMaxY()) // 2
        r = int(spec.radius)

        try:
            from lsst.geom import Box2I, Point2I
        except Exception:  # pragma: no cover
            return image

        cutout_box = Box2I(Point2I(x_center - r, y_center - r), Point2I(x_center + r, y_center + r))
        try:
            cutout_box.clip(bbox)
        except Exception:
            pass
        return image.Factory(image, cutout_box)


class _LsstButlerFactory:
    """Lazy creator for lsst.daf.butler.Butler."""

    @staticmethod
    def create(repo: str, collections: Optional[Union[str, list[str]]] = None) -> Any:
        try:
            from lsst.daf.butler import Butler
        except Exception as exc:  # pragma: no cover
            raise MissingDependencyError(
                "lsst.daf.butler is not available. Install the LSST stack to use this feature."
            ) from exc

        if collections is None:
            base = Butler(repo)
            discovered = list(base.collections.query("*"))
            if discovered:
                return Butler(repo, collections=discovered)
            return base
        return Butler(repo, collections=collections)


def _available_dataset_types(butler: Any) -> set[str]:
    try:
        return {dt.name for dt in butler.registry.queryDatasetTypes("*")}
    except Exception:
        return set()


def _default_dataset_type(butler: Any) -> str:
    available = _available_dataset_types(butler)
    fallback_order = ("preliminary_visit_image", "visit_image", "calexp")
    for candidate in fallback_order:
        if candidate in available:
            return candidate
    return "preliminary_visit_image"


def cutouts_from_butler(
    repo: str,
    *,
    collections: Optional[Union[str, list[str]]] = None,
    dataset_type: Optional[str] = None,
    butler: Optional[Any] = None,
    sky_resolver: Optional[SkyResolver] = None,
) -> ButlerCutoutService:
    """Create a cutout service backed by a Butler repository.

    Parameters
    ----------
    repo : str
        Butler repo URI.
    collections : str | list[str] | None
        Collections forwarded to ``lsst.daf.butler.Butler``.
    dataset_type : str | None
        Dataset type to read for each cutout. If omitted, a default is chosen
        from ``preliminary_visit_image``, ``visit_image``, then ``calexp``.
    butler : Any | None
        Optional pre-created Butler-like object. Useful for testing.
    sky_resolver : callable | None
        Optional function that maps ``(ra, dec, time)`` to an iterable of
        Butler data IDs.
    """
    if butler is None:
        butler = _LsstButlerFactory.create(repo, collections=collections)
    resolved_dataset_type = dataset_type or _default_dataset_type(butler)
    return ButlerCutoutService(
        butler=butler,
        dataset_type=resolved_dataset_type,
        sky_resolver=sky_resolver,
    )


def _validate_request(
    *,
    ra: Optional[float],
    dec: Optional[float],
    radius: float,
    visit: Optional[int],
    detector: Optional[int],
) -> None:
    if radius <= 0:
        raise ValueError("radius must be > 0")

    visit_mode = visit is not None or detector is not None
    sky_mode = ra is not None or dec is not None

    if visit_mode and sky_mode:
        raise ValueError("Use either (visit, detector) or (ra, dec), not both")

    if visit_mode and (visit is None or detector is None):
        raise ValueError("Both visit and detector must be provided together")

    if not visit_mode and (ra is None or dec is None):
        raise ValueError("Provide either both ra/dec or visit/detector")
