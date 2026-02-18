from __future__ import annotations

import pytest

import neandertools as nt


class FakeBBox:
    def getMinX(self):
        return 0

    def getMaxX(self):
        return 100

    def getMinY(self):
        return 0

    def getMaxY(self):
        return 100


class FakeImage:
    def __init__(self, token="root"):
        self.token = token

    def getBBox(self):
        return FakeBBox()

    @staticmethod
    def Factory(_image, _bbox):
        return FakeImage(token="cutout")


class FakeButler:
    def __init__(self):
        self.calls = []

    def get(self, dataset_type, dataId=None):
        self.calls.append((dataset_type, dataId))
        return {"data": dataId} if dataset_type == "raw" else FakeImage()

    class registry:  # minimal interface for dataset type resolution
        @staticmethod
        def queryDatasetTypes(_pattern):
            class _DT:
                name = "preliminary_visit_image"

            return [_DT()]


class FakeButlerVisitImage(FakeButler):
    class registry:
        @staticmethod
        def queryDatasetTypes(_pattern):
            class _DT:
                name = "visit_image"

            return [_DT()]


def test_factory_uses_provided_butler():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", butler=butler)
    assert isinstance(svc, nt.ButlerCutoutService)


def test_visit_detector_cutout_calls_butler():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", butler=butler)

    out = svc.cutout(visit=123, detector=9, radius=4)

    assert len(out) == 1
    assert out[0].token in {"root", "cutout"}
    assert butler.calls == [("preliminary_visit_image", {"visit": 123, "detector": 9})]


def test_sky_cutout_requires_resolver():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", butler=butler)

    with pytest.raises(NotImplementedError):
        svc.cutout(ra=10.0, dec=-1.0, radius=3)


def test_sky_cutout_with_resolver():
    butler = FakeButler()
    resolver_calls = []

    def resolver(ra, dec, time):
        resolver_calls.append((ra, dec, time))
        return [{"visit": 1, "detector": 2}, {"visit": 1, "detector": 3}]

    svc = nt.cutouts_from_butler("dp1", butler=butler, sky_resolver=resolver, dataset_type="raw")
    out = svc.cutout(ra=1.2, dec=3.4, time="2025-01-01T00:00:00", radius=5)

    assert resolver_calls == [(1.2, 3.4, "2025-01-01T00:00:00")]
    assert out == [{"data": {"visit": 1, "detector": 2}}, {"data": {"visit": 1, "detector": 3}}]


def test_dataset_type_falls_back_to_visit_image():
    butler = FakeButlerVisitImage()
    svc = nt.cutouts_from_butler("dp1", butler=butler)
    svc.cutout(visit=11, detector=12, radius=4)
    assert butler.calls == [("visit_image", {"visit": 11, "detector": 12})]


def test_invalid_args():
    butler = FakeButler()
    svc = nt.cutouts_from_butler("dp1", butler=butler)

    with pytest.raises(ValueError):
        svc.cutout(radius=2)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, visit=1, detector=2, radius=2)

    with pytest.raises(ValueError):
        svc.cutout(visit=1, radius=2)

    with pytest.raises(ValueError):
        svc.cutout(ra=1, dec=2, radius=0)
