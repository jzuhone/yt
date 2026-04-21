cdef class RayBundleSelector(SelectorObject):
    """
    Selector for a bundle of arbitrarily-aligned rays.

    Each ray is defined by a start point (p1) and a direction vector (vec).
    Selection methods return True if *any* ray in the bundle intersects the
    object.  For grid traversal (fill_mask / get_dt), the first ray (lowest
    index) that intersects a given cell "owns" that cell's t / dts values.
    """

    cdef public int n_rays
    # Flattened C arrays [n_rays * 3] – safe to access without the GIL.
    cdef np.float64_t *p1_data
    cdef np.float64_t *p2_data
    cdef np.float64_t *vec_data
    # Python-visible arrays for hashing and state serialisation.
    cdef public object start_points   # ndarray (n_rays, 3)
    cdef public object end_points     # ndarray (n_rays, 3)
    cdef public object vecs           # ndarray (n_rays, 3)

    def __cinit__(self):
        self.p1_data  = NULL
        self.p2_data  = NULL
        self.vec_data = NULL
        self.n_rays   = 0

    def __dealloc__(self):
        if self.p1_data  != NULL: free(self.p1_data)
        if self.p2_data  != NULL: free(self.p2_data)
        if self.vec_data != NULL: free(self.vec_data)

    def __init__(self, dobj):
        cdef int i, j
        cdef np.ndarray[np.float64_t, ndim=2] sp, ep, vs

        _ensure_code(dobj.start_points)
        _ensure_code(dobj.end_points)

        self.n_rays = dobj.start_points.shape[0]

        self.p1_data  = <np.float64_t *> malloc(
            self.n_rays * 3 * sizeof(np.float64_t))
        self.p2_data  = <np.float64_t *> malloc(
            self.n_rays * 3 * sizeof(np.float64_t))
        self.vec_data = <np.float64_t *> malloc(
            self.n_rays * 3 * sizeof(np.float64_t))

        sp = np.ascontiguousarray(dobj.start_points, dtype="float64")
        ep = np.ascontiguousarray(dobj.end_points,   dtype="float64")
        vs = np.ascontiguousarray(dobj.vecs,         dtype="float64")

        for i in range(self.n_rays):
            for j in range(3):
                self.p1_data [i * 3 + j] = sp[i, j]
                self.p2_data [i * 3 + j] = ep[i, j]
                self.vec_data[i * 3 + j] = vs[i, j]

        # Keep Python references for hashing / serialisation.
        self.start_points = sp
        self.end_points   = ep
        self.vecs         = vs

    # ------------------------------------------------------------------
    # GIL-free geometric tests
    # ------------------------------------------------------------------

    cdef int select_point(self, np.float64_t pos[3]) noexcept nogil:
        # Two 0-volume constructs don't intersect.
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_sphere(self, np.float64_t pos[3],
                           np.float64_t radius) noexcept nogil:
        cdef int ax, ri
        cdef np.float64_t length, l, b_sqr
        cdef np.float64_t r[3], p1_ray[3], vec_ray[3]

        for ri in range(self.n_rays):
            for ax in range(3):
                p1_ray[ax]  = self.p1_data [ri * 3 + ax]
                vec_ray[ax] = self.vec_data[ri * 3 + ax]

            length = norm(vec_ray)
            for ax in range(3):
                r[ax] = pos[ax] - p1_ray[ax]
            l     = dot(r, vec_ray) / length
            b_sqr = dot(r, r) - l * l

            if -radius < l and l < (length + radius) and b_sqr < radius * radius:
                return 1

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox(self, np.float64_t left_edge[3],
                         np.float64_t right_edge[3]) noexcept nogil:
        cdef int ax, ri, rv
        cdef np.float64_t dt_val, t_val
        cdef np.uint8_t cm
        cdef np.float64_t p1_ray[3], vec_ray[3]
        cdef VolumeContainer vc
        cdef IntegrationAccumulator ia

        for ax in range(3):
            vc.left_edge[ax]  = left_edge[ax]
            vc.right_edge[ax] = right_edge[ax]
            vc.dds[ax]        = right_edge[ax] - left_edge[ax]
            vc.idds[ax]       = 1.0 / vc.dds[ax]
            vc.dims[ax]       = 1

        cm = 1
        ia.child_mask = &cm

        for ri in range(self.n_rays):
            for ax in range(3):
                p1_ray[ax]  = self.p1_data [ri * 3 + ax]
                vec_ray[ax] = self.vec_data[ri * 3 + ax]

            t_val  = 0.0
            dt_val = 0.0
            ia.t    = &t_val
            ia.dt   = &dt_val
            ia.hits = 0

            walk_volume(&vc, p1_ray, vec_ray, dt_sampler, <void *> &ia)

            if ia.hits > 0:
                return 1

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox_edge(self, np.float64_t left_edge[3],
                              np.float64_t right_edge[3]) noexcept nogil:
        # A box with non-zero volume can never be *inside* a (1-D) ray.
        if self.select_bbox(left_edge, right_edge):
            return 2
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_cell(self, np.float64_t pos[3],
                         np.float64_t dds[3]) noexcept nogil:
        cdef int ax
        cdef np.float64_t left_edge[3], right_edge[3]
        for ax in range(3):
            left_edge[ax]  = pos[ax] - dds[ax] * 0.5
            right_edge[ax] = pos[ax] + dds[ax] * 0.5
        return self.select_bbox(left_edge, right_edge)

    # ------------------------------------------------------------------
    # Grid mask / dt methods (called from Python; GIL held)
    # ------------------------------------------------------------------

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def fill_mask_regular_grid(self, gobj):
        cdef np.ndarray[np.float64_t, ndim=3] t_ray, dt_ray
        cdef np.ndarray[np.uint8_t,   ndim=3, cast=True] child_mask
        cdef np.ndarray[np.uint8_t,   ndim=3] mask
        cdef IntegrationAccumulator *ia
        cdef VolumeContainer vc
        cdef np.float64_t p1_ray[3], vec_ray[3]
        cdef int ax, ri, ni
        cdef int total = 0

        _ensure_code(gobj.LeftEdge)
        _ensure_code(gobj.RightEdge)
        _ensure_code(gobj.dds)

        mask       = np.zeros(gobj.ActiveDimensions, dtype="uint8")
        child_mask = gobj.child_mask
        t_ray      = np.zeros(gobj.ActiveDimensions, dtype="float64")
        dt_ray     = np.zeros(gobj.ActiveDimensions, dtype="float64")

        ia = <IntegrationAccumulator *> malloc(sizeof(IntegrationAccumulator))
        ia.t          = <np.float64_t *> t_ray.data
        ia.dt         = <np.float64_t *> dt_ray.data
        ia.child_mask = <np.uint8_t  *> child_mask.data

        for ax in range(3):
            vc.left_edge[ax]  = gobj.LeftEdge[ax]
            vc.right_edge[ax] = gobj.RightEdge[ax]
            vc.dds[ax]        = gobj.dds[ax]
            vc.idds[ax]       = 1.0 / gobj.dds[ax]
            vc.dims[ax]       = dt_ray.shape[ax]

        for ri in range(self.n_rays):
            for ax in range(3):
                p1_ray[ax]  = self.p1_data [ri * 3 + ax]
                vec_ray[ax] = self.vec_data[ri * 3 + ax]

            # Reset dt_ray before each ray walk.
            dt_ray[:] = -1.0
            ia.hits = 0

            walk_volume(&vc, p1_ray, vec_ray, dt_sampler, <void *> ia)

            # Mark any newly-hit cell (first ray per cell wins).
            for i in range(mask.shape[0]):
                for j in range(mask.shape[1]):
                    for k in range(mask.shape[2]):
                        if dt_ray[i, j, k] >= 0 and mask[i, j, k] == 0:
                            mask[i, j, k] = 1
                            total += 1

        free(ia)
        if total == 0:
            return None, 0
        return mask.astype("bool"), total

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_dt(self, gobj):
        """Return (dtr, tr) for cells hit by any ray (first-hit ownership)."""
        dtr, tr, _ = self.get_dt_with_ray_index(gobj)
        return dtr, tr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def get_dt_with_ray_index(self, gobj):
        """
        Return (dtr, tr, ray_idx) – one entry per selected grid cell.

        The cell is owned by the lowest-index ray that intersects it.
        All three arrays have the same length and share the same i,j,k
        traversal order as fill_mask_regular_grid, so they are consistent
        with the standard chunk machinery.
        """
        cdef np.ndarray[np.float64_t, ndim=3] t_all, dt_all
        cdef np.ndarray[np.float64_t, ndim=3] t_ray, dt_ray
        cdef np.ndarray[np.int64_t,   ndim=3] ridx_all
        cdef np.ndarray[np.float64_t, ndim=1] tr, dtr
        cdef np.ndarray[np.int64_t,   ndim=1] ray_idx_out
        cdef np.ndarray[np.uint8_t,   ndim=3, cast=True] child_mask
        cdef IntegrationAccumulator *ia
        cdef VolumeContainer vc
        cdef np.float64_t p1_ray[3], vec_ray[3]
        cdef int ax, ri, ni

        _ensure_code(gobj.LeftEdge)
        _ensure_code(gobj.RightEdge)
        _ensure_code(gobj.dds)

        shape = gobj.ActiveDimensions
        t_all    = np.zeros(shape, dtype="float64")
        dt_all   = np.full (shape, -1.0, dtype="float64")
        ridx_all = np.full (shape, -1,   dtype="int64")

        t_ray    = np.zeros(shape, dtype="float64")
        dt_ray   = np.full (shape, -1.0, dtype="float64")
        child_mask = gobj.child_mask

        ia = <IntegrationAccumulator *> malloc(sizeof(IntegrationAccumulator))
        ia.t          = <np.float64_t *> t_ray.data
        ia.dt         = <np.float64_t *> dt_ray.data
        ia.child_mask = <np.uint8_t  *> child_mask.data

        for ax in range(3):
            vc.left_edge[ax]  = gobj.LeftEdge[ax]
            vc.right_edge[ax] = gobj.RightEdge[ax]
            vc.dds[ax]        = gobj.dds[ax]
            vc.idds[ax]       = 1.0 / gobj.dds[ax]
            vc.dims[ax]       = dt_ray.shape[ax]

        ni = 0
        for ri in range(self.n_rays):
            for ax in range(3):
                p1_ray[ax]  = self.p1_data [ri * 3 + ax]
                vec_ray[ax] = self.vec_data[ri * 3 + ax]

            dt_ray[:] = -1.0
            ia.hits = 0

            walk_volume(&vc, p1_ray, vec_ray, dt_sampler, <void *> ia)

            for i in range(shape[0]):
                for j in range(shape[1]):
                    for k in range(shape[2]):
                        if dt_ray[i, j, k] >= 0 and dt_all[i, j, k] < 0:
                            t_all   [i, j, k] = t_ray [i, j, k]
                            dt_all  [i, j, k] = dt_ray[i, j, k]
                            ridx_all[i, j, k] = ri
                            ni += 1

        free(ia)

        tr          = np.zeros(ni, dtype="float64")
        dtr         = np.zeros(ni, dtype="float64")
        ray_idx_out = np.zeros(ni, dtype="int64")
        ni = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if dt_all[i, j, k] >= 0:
                        tr         [ni] = t_all   [i, j, k]
                        dtr        [ni] = dt_all  [i, j, k]
                        ray_idx_out[ni] = ridx_all[i, j, k]
                        ni += 1

        return dtr, tr, ray_idx_out

    # ------------------------------------------------------------------
    # Hashing / serialisation
    # ------------------------------------------------------------------

    def _hash_vals(self):
        return (
            ("n_rays",       self.n_rays),
            ("start_points", self.start_points.tobytes()),
            ("end_points",   self.end_points.tobytes()),
        )

    def _get_state_attnames(self):
        return ("n_rays", "start_points", "end_points", "vecs")


ray_bundle_selector = RayBundleSelector
