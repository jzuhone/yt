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
from yt.units._numpy_wrapper_functions import udot, unorm
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
    exactly like :class:`YTRay`.  The bundle selects the union of all cells
    intersected by any ray and exposes the extra ``ray_index`` container
    field to identify which ray (0-based) owns each cell.  When multiple
    rays pass through the same cell the one with the lowest index is used.

    This object is typically accessed through the ``ray_bundle`` attribute
    of a dataset index object.

    Parameters
    ----------
    start_points : array-like, shape (N, 3)
        Starting coordinates of each ray.  If plain numbers are given they
        are assumed to be in *code* units; pass a :class:`~yt.units.yt_array.YTArray`
        to supply explicit units.
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
    >>> import numpy as np
    >>> ds = yt.load("RedshiftOutput0005")
    >>> # Three parallel rays along z separated in x
    >>> starts = ds.arr([[0.2, 0.5, 0.0], [0.4, 0.5, 0.0], [0.6, 0.5, 0.0]], "code_length")
    >>> ends   = ds.arr([[0.2, 0.5, 1.0], [0.4, 0.5, 1.0], [0.6, 0.5, 1.0]], "code_length")
    >>> bundle = ds.ray_bundle(starts, ends)
    >>> print(bundle["gas", "density"])
    >>> print(bundle["ray_index"])   # which ray owns each cell

    Note: Like :class:`YTRay`, the data are not guaranteed to be spatially
    ordered.  Sort by the ``t`` field (parametric position along each ray,
    0 at start → 1 at end) within each ``ray_index`` group if you need
    spatial order.
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

        self._dts, self._ts = None, None

    # ------------------------------------------------------------------
    # Container-field generation
    # ------------------------------------------------------------------

    def _generate_container_field(self, field):
        if isinstance(self.ds, SPHDataset):
            raise NotImplementedError("YTRayBundle does not yet support SPH datasets.")
        return self._generate_container_field_grid(field)

    def _generate_container_field_grid(self, field):
        if self._current_chunk is None:
            self.index._identify_base_chunk(self)
        chunk = self._current_chunk
        # Compute and cache all three arrays in one pass to avoid repeated
        # grid walks (which are O(N_rays × grid_cells)).
        if not hasattr(chunk, "_bundle_tcoords"):
            self._compute_bundle_coords(chunk)
        if field == "dts":
            return chunk._bundle_dtcoords
        elif field == "t":
            return chunk._bundle_tcoords
        elif field == "ray_index":
            return chunk._bundle_ray_index
        raise KeyError(field)

    def _compute_bundle_coords(self, chunk):
        """Walk each object in *chunk* once, collecting t, dts, and ray_index."""
        dts_list, ts_list, ri_list = [], [], []
        for obj in chunk._fast_index or chunk.objs:
            dtr, tr, ri = self.selector.get_dt_with_ray_index(obj)
            if tr.size == 0:
                continue
            dts_list.append(dtr)
            ts_list.append(tr)
            ri_list.append(ri)
        if dts_list:
            chunk._bundle_dtcoords = np.concatenate(dts_list)
            chunk._bundle_tcoords = np.concatenate(ts_list)
            chunk._bundle_ray_index = np.concatenate(ri_list)
        else:
            chunk._bundle_dtcoords = np.empty(0, dtype="float64")
            chunk._bundle_tcoords = np.empty(0, dtype="float64")
            chunk._bundle_ray_index = np.empty(0, dtype="int64")
