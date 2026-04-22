import numpy as np

from yt.data_objects.selection_objects.data_selection_objects import (
    YTSelectionContainer,
    YTSelectionContainer1D,
)
from yt.data_objects.static_output import Dataset
from yt.frontends.sph.data_structures import SPHDataset
from yt.funcs import (
    fix_axis,
    validate_3d_array,
    validate_axis,
    validate_float,
    validate_object,
    validate_sequence,
)
from yt.units import YTArray, YTQuantity
from yt.units._numpy_wrapper_functions import uconcatenate, udot, unorm
from yt.utilities.lib.pixelization_routines import SPHKernelInterpolationTable
from yt.utilities.logger import ytLogger as mylog


class YTOrthoRay(YTSelectionContainer1D):
    """
    This is an orthogonal ray cast through the entire domain, at a specific
    coordinate.

    This object is typically accessed through the `ortho_ray` object that
    hangs off of index objects.  The resulting arrays have their
    dimensionality reduced to one, and an ordered list of points at an
    (x,y) tuple along `axis` are available.

    Parameters
    ----------
    axis : int or char
        The axis along which to slice.  Can be 0, 1, or 2 for x, y, z.
    coords : tuple of floats
        The (plane_x, plane_y) coordinates at which to cast the ray.  Note
        that this is in the plane coordinates: so if you are casting along
        x, this will be (y, z).  If you are casting along y, this will be
        (z, x).  If you are casting along z, this will be (x, y).
    ds: ~yt.data_objects.static_output.Dataset, optional
        An optional dataset to use rather than self.ds
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.
    data_source: optional
        Draw the selection from the provided data source rather than
        all data associated with the data_set

    Examples
    --------

    >>> import yt
    >>> ds = yt.load("RedshiftOutput0005")
    >>> oray = ds.ortho_ray(0, (0.2, 0.74))
    >>> print(oray["gas", "density"])

    Note: The low-level data representation for rays are not guaranteed to be
    spatially ordered.  In particular, with AMR datasets, higher resolution
    data is tagged on to the end of the ray.  If you want this data
    represented in a spatially ordered manner, manually sort it by the "t"
    field, which is the value of the parametric variable that goes from 0 at
    the start of the ray to 1 at the end:

    >>> my_ray = ds.ortho_ray(...)
    >>> ray_sort = np.argsort(my_ray["t"])
    >>> density = my_ray["gas", "density"][ray_sort]
    """

    _key_fields = ["x", "y", "z", "dx", "dy", "dz"]
    _type_name = "ortho_ray"
    _con_args = ("axis", "coords")

    def __init__(self, axis, coords, ds=None, field_parameters=None, data_source=None):
        validate_axis(ds, axis)
        validate_sequence(coords)
        for c in coords:
            validate_float(c)
        validate_object(ds, Dataset)
        validate_object(field_parameters, dict)
        validate_object(data_source, YTSelectionContainer)
        super().__init__(ds, field_parameters, data_source)
        self.axis = fix_axis(axis, self.ds)
        xax = self.ds.coordinates.x_axis[self.axis]
        yax = self.ds.coordinates.y_axis[self.axis]
        self.px_ax = xax
        self.py_ax = yax
        # Even though we may not be using x,y,z we use them here.
        self.px_dx = f"d{'xyz'[self.px_ax]}"
        self.py_dx = f"d{'xyz'[self.py_ax]}"
        # Convert coordinates to code length.
        if isinstance(coords[0], YTQuantity):
            self.px = self.ds.quan(coords[0]).to("code_length")
        else:
            self.px = self.ds.quan(coords[0], "code_length")
        if isinstance(coords[1], YTQuantity):
            self.py = self.ds.quan(coords[1]).to("code_length")
        else:
            self.py = self.ds.quan(coords[1], "code_length")
        self.sort_by = "xyz"[self.axis]

    @property
    def coords(self):
        return (self.px, self.py)


class YTRay(YTSelectionContainer1D):
    """
    This is an arbitrarily-aligned ray cast through the entire domain, at a
    specific coordinate.

    This object is typically accessed through the `ray` object that hangs
    off of index objects.  The resulting arrays have their
    dimensionality reduced to one, and an ordered list of points at an
    (x,y) tuple along `axis` are available, as is the `t` field, which
    corresponds to a unitless measurement along the ray from start to
    end.

    Parameters
    ----------
    start_point : array-like set of 3 floats
        The place where the ray starts.
    end_point : array-like set of 3 floats
        The place where the ray ends.
    ds: ~yt.data_objects.static_output.Dataset, optional
        An optional dataset to use rather than self.ds
    field_parameters : dictionary
         A dictionary of field parameters than can be accessed by derived
         fields.
    data_source: optional
        Draw the selection from the provided data source rather than
        all data associated with the data_set

    Examples
    --------

    >>> import yt
    >>> ds = yt.load("RedshiftOutput0005")
    >>> ray = ds.ray((0.2, 0.74, 0.11), (0.4, 0.91, 0.31))
    >>> print(ray["gas", "density"], ray["t"], ray["dts"])

    Note: The low-level data representation for rays are not guaranteed to be
    spatially ordered.  In particular, with AMR datasets, higher resolution
    data is tagged on to the end of the ray.  If you want this data
    represented in a spatially ordered manner, manually sort it by the "t"
    field, which is the value of the parametric variable that goes from 0 at
    the start of the ray to 1 at the end:

    >>> my_ray = ds.ray(...)
    >>> ray_sort = np.argsort(my_ray["t"])
    >>> density = my_ray["gas", "density"][ray_sort]
    """

    _type_name = "ray"
    _con_args = ("start_point", "end_point")
    _container_fields = ("t", "dts")

    def __init__(
        self, start_point, end_point, ds=None, field_parameters=None, data_source=None
    ):
        validate_3d_array(start_point)
        validate_3d_array(end_point)
        validate_object(ds, Dataset)
        validate_object(field_parameters, dict)
        validate_object(data_source, YTSelectionContainer)
        super().__init__(ds, field_parameters, data_source)
        if isinstance(start_point, YTArray):
            self.start_point = self.ds.arr(start_point).to("code_length")
        else:
            self.start_point = self.ds.arr(start_point, "code_length", dtype="float64")
        if isinstance(end_point, YTArray):
            self.end_point = self.ds.arr(end_point).to("code_length")
        else:
            self.end_point = self.ds.arr(end_point, "code_length", dtype="float64")
        if (self.start_point < self.ds.domain_left_edge).any() or (
            self.end_point > self.ds.domain_right_edge
        ).any():
            mylog.warning(
                "Ray start or end is outside the domain. "
                "Returned data will only be for the ray section inside the domain."
            )
        self.vec = self.end_point - self.start_point
        self._set_center(self.start_point)
        self.set_field_parameter("center", self.start_point)
        self._dts, self._ts = None, None

    def _generate_container_field(self, field):
        # What should we do with `ParticleDataset`?
        if isinstance(self.ds, SPHDataset):
            return self._generate_container_field_sph(field)
        else:
            return self._generate_container_field_grid(field)

    def _generate_container_field_grid(self, field):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        if field == "dts":
            return self._current_chunk.dtcoords
        elif field == "t":
            return self._current_chunk.tcoords
        else:
            raise KeyError(field)

    def _generate_container_field_sph(self, field):
        if field not in ["dts", "t"]:
            raise KeyError(field)

        length = unorm(self.vec)
        pos = self[self.ds._sph_ptypes[0], "particle_position"]
        r = pos - self.start_point
        l = udot(r, self.vec / length)

        if field == "t":
            return l / length

        hsml = self[self.ds._sph_ptypes[0], "smoothing_length"]
        mass = self[self.ds._sph_ptypes[0], "particle_mass"]
        dens = self[self.ds._sph_ptypes[0], "density"]
        # impact parameter from particle to ray
        b = np.sqrt(np.sum(r**2, axis=1) - l**2)

        # Use an interpolation table to evaluate the integrated 2D
        # kernel from the dimensionless impact parameter b/hsml.
        # The table tabulates integrals for values of (b/hsml)**2
        itab = SPHKernelInterpolationTable(self.ds.kernel_name)
        dl = itab.interpolate_array((b / hsml) ** 2) * mass / dens / hsml**2
        return dl / length


class YTRayBundle(YTSelectionContainer1D):
    """
    A bundle of arbitrarily-aligned rays cast through the dataset.

    Each ray in the bundle is defined by a start point and an end point,
    exactly like :class:`YTRay`.  Every ray independently selects the cells
    it passes through, so a cell hit by multiple rays appears in the output
    **once per ray that crosses it**, each with its own ``t`` and ``dts``
    values.  The ``ray_index`` field (0-based integer) identifies which ray
    each data point belongs to.

    This object is typically accessed through the ``ray_bundle`` attribute
    of a dataset index object.

    Parameters
    ----------
    start_points : array-like, shape (N, 3)
        Starting coordinates of each ray.  If plain numbers are given they
        are assumed to be in *code* units; pass a
        :class:`~yt.units.yt_array.YTArray` to supply explicit units.
    end_points : array-like, shape (N, 3)
        Ending coordinates of each ray, in the same format.
    ds : ~yt.data_objects.static_output.Dataset, optional
        Dataset to use.
    field_parameters : dict, optional
        Field parameters accessible from derived fields.
    data_source : optional
        Restrict selection to a parent data source.

    Examples
    --------

    >>> import yt
    >>> ds = yt.load("RedshiftOutput0005")
    >>> # Three parallel rays along z, separated in x
    >>> starts = ds.arr([[0.2, 0.5, 0.0], [0.4, 0.5, 0.0], [0.6, 0.5, 0.0]], "code_length")
    >>> ends   = ds.arr([[0.2, 0.5, 1.0], [0.4, 0.5, 1.0], [0.6, 0.5, 1.0]], "code_length")
    >>> bundle = ds.ray_bundle(starts, ends)
    >>> print(bundle["gas", "density"])
    >>> print(bundle["ray_index"])   # 0, 0, …, 1, 1, …, 2, 2, …

    Note: data are not guaranteed to be spatially ordered.  Sort by ``t``
    (parametric position 0 → 1 along each ray) within each ``ray_index``
    group to recover spatial order.
    """

    _type_name = "ray_bundle"
    _con_args = ("start_points", "end_points")
    _container_fields = ("t", "dts", "ray_index")

    def __init__(
        self,
        start_points,
        end_points,
        ds=None,
        field_parameters=None,
        data_source=None,
    ):
        validate_object(ds, Dataset)
        validate_object(field_parameters, dict)
        validate_object(data_source, YTSelectionContainer)
        super().__init__(ds, field_parameters, data_source)

        start_points = np.asarray(start_points, dtype="float64")
        end_points = np.asarray(end_points, dtype="float64")

        if start_points.ndim != 2 or start_points.shape[1] != 3:
            raise TypeError(
                "start_points must be an array of shape (N, 3), "
                f"got shape {start_points.shape}"
            )
        if end_points.ndim != 2 or end_points.shape[1] != 3:
            raise TypeError(
                "end_points must be an array of shape (N, 3), "
                f"got shape {end_points.shape}"
            )
        if start_points.shape[0] != end_points.shape[0]:
            raise TypeError(
                "start_points and end_points must have the same number of rows, "
                f"got {start_points.shape[0]} and {end_points.shape[0]}"
            )

        # Convert to code_length, preserving any existing unit information.
        if isinstance(start_points, YTArray):
            self.start_points = self.ds.arr(start_points).to("code_length")
        else:
            self.start_points = self.ds.arr(
                start_points, "code_length", dtype="float64"
            )
        if isinstance(end_points, YTArray):
            self.end_points = self.ds.arr(end_points).to("code_length")
        else:
            self.end_points = self.ds.arr(end_points, "code_length", dtype="float64")

        self.vecs = self.end_points - self.start_points

        # Warn if any ray endpoint lies outside the domain.
        if (self.start_points < self.ds.domain_left_edge).any() or (
            self.end_points > self.ds.domain_right_edge
        ).any():
            mylog.warning(
                "One or more ray start/end points are outside the domain. "
                "Returned data will only cover the portions inside the domain."
            )

        # Use the centroid of all start/end points as the notional centre.
        center = 0.5 * (self.start_points.mean(axis=0) + self.end_points.mean(axis=0))
        self._set_center(center)
        self.set_field_parameter("center", center)

        self._rays_cache = None

    # ------------------------------------------------------------------
    # Individual-ray access
    # ------------------------------------------------------------------

    @property
    def _individual_rays(self):
        """Lazily build one YTRay per (start, end) pair."""
        if self._rays_cache is None:
            self._rays_cache = [
                self.ds.ray(sp, ep)
                for sp, ep in zip(self.start_points, self.end_points, strict=True)
            ]
        return self._rays_cache

    # ------------------------------------------------------------------
    # Data access
    #
    # Each ray is independent: a cell hit by ray 0 *and* ray 1 contributes
    # two separate data points.  We therefore bypass yt's per-cell mask
    # machinery entirely and delegate to individual YTRay objects, then
    # concatenate their results.
    # ------------------------------------------------------------------

    def __getitem__(self, key):
        # Container fields handled inline.
        if key in self._container_fields:
            if key not in self.field_data:
                self.field_data[key] = self._generate_container_field(key)
            return self.field_data[key]

        # All other fields: concatenate across individual rays.
        f = self._determine_fields(key)
        fkey = f[0]
        if fkey not in self.field_data:
            arrays = [r[key] for r in self._individual_rays]
            self.field_data[fkey] = (
                uconcatenate(arrays)
                if arrays
                else self.ds.arr(np.zeros(0), "dimensionless")
            )
        return self.field_data[fkey]

    def _generate_container_field(self, field):
        rays = self._individual_rays
        if field == "ray_index":
            return np.concatenate(
                [np.full(len(r["t"]), i, dtype="int64") for i, r in enumerate(rays)]
            )
        elif field == "t":
            return uconcatenate([r["t"] for r in rays])
        elif field == "dts":
            return uconcatenate([r["dts"] for r in rays])
        raise KeyError(field)
