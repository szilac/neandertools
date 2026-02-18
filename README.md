# neandertools

`neandertools` provides a small API for generating image cutouts from Rubin Observatory LSST Butler repositories.

## Quick start

```python
import neandertools as nt

svc = nt.cutouts_from_butler("~/lsst/dp1_subset")
images = svc.cutout(visit=2024110800253, detector=5, radius=100)
```

For this repo, `dataset_type` is auto-selected as `visit_image`. If your repo has
`preliminary_visit_image`, it will be selected automatically instead.

By default, `cutout` returns a list of image-like objects returned by your Butler stack.

## Notes

- This package keeps LSST imports optional and raises a clear error if the LSST stack is not installed.
- The `visit+detector` cutout path is implemented and tested against `~/lsst/dp1_subset`.
- The sky-coordinate mode (`ra/dec/time`) is wired, and needs a `sky_resolver` callback.
