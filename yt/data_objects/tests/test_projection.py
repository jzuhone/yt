import os
import tempfile
from unittest import mock

import numpy as np
from numpy.testing import assert_equal

from yt.testing import assert_rel_equal, fake_amr_ds, fake_random_ds
from yt.units.unit_object import Unit

LENGTH_UNIT = 2.0


def setup_module():
    from yt.config import ytcfg

    ytcfg["yt", "internals", "within_testing"] = True


def teardown_func(fns):
    for fn in fns:
        try:
            os.remove(fn)
        except OSError:
            pass


@mock.patch("matplotlib.backends.backend_agg.FigureCanvasAgg.print_figure")
def test_projection(pf):
    fns = []
    for nprocs in [8, 1]:
        # We want to test both 1 proc and 8 procs, to make sure that
        # parallelism isn't broken
        fields = ("density", "temperature", "velocity_x", "velocity_y", "velocity_z")
        units = ("g/cm**3", "K", "cm/s", "cm/s", "cm/s")
        ds = fake_random_ds(
            64, fields=fields, units=units, nprocs=nprocs, length_unit=LENGTH_UNIT
        )
        dims = ds.domain_dimensions
        xn, yn, zn = ds.domain_dimensions
        xi, yi, zi = ds.domain_left_edge.to_ndarray() + 1.0 / (ds.domain_dimensions * 2)
        xf, yf, zf = ds.domain_right_edge.to_ndarray() - 1.0 / (
            ds.domain_dimensions * 2
        )
        dd = ds.all_data()
        coords = np.mgrid[xi : xf : xn * 1j, yi : yf : yn * 1j, zi : zf : zn * 1j]
        uc = [np.unique(c) for c in coords]
        # test if projections inherit the field parameters of their data sources
        dd.set_field_parameter("bulk_velocity", np.array([0, 1, 2]))
        proj = ds.proj(("gas", "density"), 0, data_source=dd)
        assert_equal(
            dd.field_parameters["bulk_velocity"], proj.field_parameters["bulk_velocity"]
        )

        # Some simple projection tests with single grids
        for ax, an in enumerate("xyz"):
            xax = ds.coordinates.x_axis[ax]
            yax = ds.coordinates.y_axis[ax]
            for wf in [("gas", "density"), None]:
                proj = ds.proj(
                    [("index", "ones"), ("gas", "density")], ax, weight_field=wf
                )
                if wf is None:
                    assert_equal(
                        proj["index", "ones"].sum(),
                        LENGTH_UNIT * proj["index", "ones"].size,
                    )
                    assert_equal(proj["index", "ones"].min(), LENGTH_UNIT)
                    assert_equal(proj["index", "ones"].max(), LENGTH_UNIT)
                else:
                    assert_equal(
                        proj["index", "ones"].sum(), proj["index", "ones"].size
                    )
                    assert_equal(proj["index", "ones"].min(), 1.0)
                    assert_equal(proj["index", "ones"].max(), 1.0)
                assert_equal(np.unique(proj["px"]), uc[xax])
                assert_equal(np.unique(proj["py"]), uc[yax])
                assert_equal(np.unique(proj["pdx"]), 1.0 / (dims[xax] * 2.0))
                assert_equal(np.unique(proj["pdy"]), 1.0 / (dims[yax] * 2.0))
                plots = [proj.to_pw(fields=("gas", "density")), proj.to_pw()]
                for pw in plots:
                    for p in pw.plots.values():
                        tmpfd, tmpname = tempfile.mkstemp(suffix=".png")
                        os.close(tmpfd)
                        p.save(name=tmpname)
                        fns.append(tmpname)
                frb = proj.to_frb((1.0, "unitary"), 64)
                for proj_field in [
                    ("index", "ones"),
                    ("gas", "density"),
                    ("gas", "temperature"),
                ]:
                    fi = ds._get_field_info(proj_field)
                    assert_equal(frb[proj_field].info["data_source"], proj.__str__())
                    assert_equal(frb[proj_field].info["axis"], ax)
                    assert_equal(frb[proj_field].info["field"], str(proj_field))
                    field_unit = Unit(fi.units)
                    if wf is not None:
                        assert_equal(
                            frb[proj_field].units,
                            Unit(field_unit, registry=ds.unit_registry),
                        )
                    else:
                        if frb[proj_field].units.is_code_unit:
                            proj_unit = "code_length"
                        else:
                            proj_unit = "cm"
                        if field_unit != "" and field_unit != Unit():
                            proj_unit = f"({field_unit}) * {proj_unit}"
                        assert_equal(
                            frb[proj_field].units,
                            Unit(proj_unit, registry=ds.unit_registry),
                        )
                    assert_equal(frb[proj_field].info["xlim"], frb.bounds[:2])
                    assert_equal(frb[proj_field].info["ylim"], frb.bounds[2:])
                    assert_equal(frb[proj_field].info["center"], proj.center)
                    if wf is None:
                        assert_equal(frb[proj_field].info["weight_field"], wf)
                    else:
                        assert_equal(
                            frb[proj_field].info["weight_field"],
                            proj.data_source._determine_fields(wf)[0],
                        )
            # wf == None
            assert_equal(wf, None)
            v1 = proj["gas", "density"].sum()
            v2 = (dd["gas", "density"] * dd["index", f"d{an}"]).sum()
            assert_rel_equal(v1, v2.in_units(v1.units), 10)

        # Test moment projections
        def make_vsq_field(aname):
            def _vsquared(data):
                return data["gas", f"velocity_{aname}"] ** 2

            return _vsquared

        for ax, an in enumerate("xyz"):
            ds.add_field(
                ("gas", f"velocity_{an}_squared"),
                make_vsq_field(an),
                sampling_type="local",
                units="cm**2/s**2",
            )
            proj1 = ds.proj(
                [("gas", f"velocity_{an}"), ("gas", f"velocity_{an}_squared")],
                ax,
                weight_field=("gas", "density"),
                moment=1,
            )
            proj2 = ds.proj(
                ("gas", f"velocity_{an}"), ax, weight_field=("gas", "density"), moment=2
            )
            assert_rel_equal(
                np.sqrt(
                    proj1["gas", f"velocity_{an}_squared"]
                    - proj1["gas", f"velocity_{an}"] ** 2
                ),
                proj2["gas", f"velocity_{an}"],
                10,
            )
    teardown_func(fns)


def _add_velocity_vector_field(ds):
    # a vector-valued field: one access returns an (N, 3) array stacking the
    # three velocity components along a trailing axis
    def _velocity_vector(field, data):
        return data.ds.arr(
            np.stack(
                [
                    data["gas", "velocity_x"],
                    data["gas", "velocity_y"],
                    data["gas", "velocity_z"],
                ],
                axis=-1,
            ),
            "cm/s",
        )

    ds.add_field(
        ("gas", "velocity_vector"),
        _velocity_vector,
        sampling_type="local",
        units="cm/s",
    )


def test_proj_vector():
    # projecting a vector-valued field with proj_vector must match stacking the
    # per-component scalar projections, for both weighted and unweighted, and
    # for moment=1 and moment=2
    fields = ("density", "velocity_x", "velocity_y", "velocity_z")
    units = ("g/cm**3", "cm/s", "cm/s", "cm/s")
    components = [("gas", "velocity_x"), ("gas", "velocity_y"), ("gas", "velocity_z")]
    for nprocs in [8, 1]:
        ds = fake_random_ds(32, fields=fields, units=units, nprocs=nprocs)
        _add_velocity_vector_field(ds)
        assert hasattr(ds, "proj_vector")
        for ax in range(3):
            for wf in [("gas", "density"), None]:
                vproj = ds.proj_vector(("gas", "velocity_vector"), ax, weight_field=wf)
                vv = vproj["gas", "velocity_vector"]
                assert vv.shape == (vproj["px"].size, 3)
                sproj = ds.proj(components, ax, weight_field=wf)
                ref = np.stack([sproj[c] for c in components], axis=-1)
                assert_equal(vv, ref)
                assert_equal(vv.units, sproj[components[0]].units)
        # standard deviation (moment=2) along the components
        vproj2 = ds.proj_vector(
            ("gas", "velocity_vector"), 0, weight_field=("gas", "density"), moment=2
        )
        for i, c in enumerate(components):
            sproj2 = ds.proj(c, 0, weight_field=("gas", "density"), moment=2)
            assert_rel_equal(vproj2["gas", "velocity_vector"][:, i], sproj2[c], 10)


def test_proj_vector_sph():
    # the SPH path projects all components in a single pass through
    # pixelize_sph_kernel_projection_vector; the resulting (ny, nx, ncomp) image
    # must match stacking the per-component scalar SPH projections
    from yt.testing import fake_random_sph_ds

    bbox = np.array([[0.0, 10.0], [0.0, 10.0], [0.0, 10.0]])
    ds = fake_random_sph_ds(2000, bbox)

    def _make_component(factor):
        def _component(field, data):
            return factor * data["gas", "density"]

        return _component

    for i, factor in enumerate((1.0, 2.0, 3.0)):
        ds.add_field(
            ("gas", f"c{i}"),
            _make_component(factor),
            sampling_type="local",
            units="g/cm**3",
        )

    def _vector(field, data):
        return data.ds.arr(
            np.stack(
                [data["gas", "c0"], data["gas", "c1"], data["gas", "c2"]], axis=-1
            ),
            "g/cm**3",
        )

    ds.add_field(("gas", "vector"), _vector, sampling_type="local", units="g/cm**3")

    width = (10.0, "cm")
    npix = 32
    for wf in [None, ("gas", "density")]:
        vproj = ds.proj_vector(("gas", "vector"), "z", weight_field=wf)
        img = np.asarray(vproj.to_frb(width, npix)["gas", "vector"])
        assert img.shape == (npix, npix, 3)
        ref = np.stack(
            [
                np.asarray(
                    ds.proj(("gas", f"c{i}"), "z", weight_field=wf).to_frb(width, npix)[
                        "gas", f"c{i}"
                    ]
                )
                for i in range(3)
            ],
            axis=-1,
        )
        np.testing.assert_allclose(img, ref, rtol=1e-12, equal_nan=True)


def test_max_level():
    ds = fake_amr_ds(fields=[("gas", "density")], units=["mp/cm**3"])
    proj = ds.proj(("gas", "density"), 2, method="max", max_level=2)
    assert proj["index", "grid_level"].max() == 2

    proj = ds.proj(("gas", "density"), 2, method="max")
    assert proj["index", "grid_level"].max() == ds.index.max_level


def test_min_level():
    ds = fake_amr_ds(fields=[("gas", "density")], units=["mp/cm**3"])
    proj = ds.proj(("gas", "density"), 2, method="min")
    assert proj["index", "grid_level"].min() == 0

    proj = ds.proj(("gas", "density"), 2, method="max")
    assert proj["index", "grid_level"].min() == ds.index.min_level
