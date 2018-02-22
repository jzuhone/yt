import sys

import numpy as np

from yt._typing import FieldType
from yt.fields.derived_field import ValidateParameter
from yt.fields.field_info_container import FieldInfoContainer
from yt.geometry.api import Geometry
from yt.units import dimensions

from .field_plugin_registry import register_field_plugin
from .vector_operations import \
    create_magnitude_field

from yt.funcs import \
    just_one

from yt.fields.derived_field import \
    ValidateParameter, \
    ValidateSpatial

if sys.version_info >= (3, 11):
    from typing import assert_never
else:
    from typing_extensions import assert_never

cgs_normalizations = {"gaussian": 4.0 * np.pi, "lorentz_heaviside": 1.0}

def get_magnetic_normalization(key: str) -> float:
    if key not in cgs_normalizations:
        raise ValueError(
            "Unknown magnetic normalization convention. "
            f"Got {key!r}, expected one of {tuple(cgs_normalizations)}"
        )
    return cgs_normalizations[key]


@register_field_plugin
def setup_magnetic_field_fields(
    registry: FieldInfoContainer, ftype: FieldType = "gas", slice_info=None
) -> None:
    ds = registry.ds

    unit_system = ds.unit_system
    pc = registry.ds.units.physical_constants

    axis_names = registry.ds.coordinates.axis_order

    if (ftype, f"magnetic_field_{axis_names[0]}") not in registry:
        return

    u = registry[ftype, f"magnetic_field_{axis_names[0]}"].units

    def mag_factors(dims):
        if dims == dimensions.magnetic_field_cgs:
            return getattr(ds, "_magnetic_factor", 4.0 * np.pi)
        elif dims == dimensions.magnetic_field_mks:
            return ds.units.physical_constants.mu_0

    def _magnetic_field_strength(field, data):
        xm = f"relative_magnetic_field_{axis_names[0]}"
        ym = f"relative_magnetic_field_{axis_names[1]}"
        zm = f"relative_magnetic_field_{axis_names[2]}"

        B2 = (data[ftype, xm]) ** 2 + (data[ftype, ym]) ** 2 + (data[ftype, zm]) ** 2

        return np.sqrt(B2)
    
    registry.add_field(
        (ftype, "magnetic_field_strength"),
        sampling_type="local",
        function=_magnetic_field_strength,
        validators=[ValidateParameter("bulk_magnetic_field")],
        units=u,
    )

    def _magnetic_energy_density(field, data):
        B = data[ftype, "magnetic_field_strength"]
        return 0.5 * B * B / mag_factors(B.units.dimensions)

    registry.add_field(
        (ftype, "magnetic_energy_density"),
        sampling_type="local",
        function=_magnetic_energy_density,
        units=unit_system["pressure"],
    )

    def _plasma_beta(field, data):
        return data[ftype, "pressure"] / data[ftype, "magnetic_energy_density"]

    registry.add_field(
        (ftype, "plasma_beta"), sampling_type="local", function=_plasma_beta, units=""
    )

    def _magnetic_pressure(field, data):
        return data[ftype, "magnetic_energy_density"]

    registry.add_field(
        (ftype, "magnetic_pressure"),
        sampling_type="local",
        function=_magnetic_pressure,
        units=unit_system["pressure"],
    )

    _magnetic_field_poloidal_magnitude = None
    _magnetic_field_toroidal_magnitude = None

    geometry: Geometry = registry.ds.geometry

    if geometry is Geometry.CARTESIAN:

        def _magnetic_field_poloidal_magnitude(field, data):
            B2 = (
                data[ftype, "relative_magnetic_field_x"]
                * data[ftype, "relative_magnetic_field_x"]
                + data[ftype, "relative_magnetic_field_y"]
                * data[ftype, "relative_magnetic_field_y"]
                + data[ftype, "relative_magnetic_field_z"]
                * data[ftype, "relative_magnetic_field_z"]
            )
            Bt2 = (
                data[ftype, "magnetic_field_spherical_phi"]
                * data[ftype, "magnetic_field_spherical_phi"]
            )
            return np.sqrt(B2 - Bt2)

    elif geometry is Geometry.CYLINDRICAL or geometry is Geometry.POLAR:

        def _magnetic_field_poloidal_magnitude(field, data):
            bm = data.get_field_parameter("bulk_magnetic_field")
            rax = axis_names.index("r")
            zax = axis_names.index("z")

            return np.sqrt(
                (data[ftype, "magnetic_field_r"] - bm[rax]) ** 2
                + (data[ftype, "magnetic_field_z"] - bm[zax]) ** 2
            )

        def _magnetic_field_toroidal_magnitude(field, data):
            ax = axis_names.find("theta")
            bm = data.get_field_parameter("bulk_magnetic_field")
            return data[ftype, "magnetic_field_theta"] - bm[ax]

    elif geometry is Geometry.SPHERICAL:

        def _magnetic_field_poloidal_magnitude(field, data):
            bm = data.get_field_parameter("bulk_magnetic_field")
            rax = axis_names.index("r")
            tax = axis_names.index("theta")

            return np.sqrt(
                (data[ftype, "magnetic_field_r"] - bm[rax]) ** 2
                + (data[ftype, "magnetic_field_theta"] - bm[tax]) ** 2
            )

        def _magnetic_field_toroidal_magnitude(field, data):
            ax = axis_names.find("phi")
            bm = data.get_field_parameter("bulk_magnetic_field")
            return data[ftype, "magnetic_field_phi"] - bm[ax]

    elif geometry is Geometry.GEOGRAPHIC or geometry is Geometry.INTERNAL_GEOGRAPHIC:
        # not implemented
        pass
    elif geometry is Geometry.SPECTRAL_CUBE:
        # nothing to be done
        pass
    else:
        assert_never(geometry)

    if _magnetic_field_poloidal_magnitude is not None:
        registry.add_field(
            (ftype, "magnetic_field_poloidal_magnitude"),
            sampling_type="local",
            function=_magnetic_field_poloidal_magnitude,
            units=u,
            validators=[
                ValidateParameter("normal"),
                ValidateParameter("bulk_magnetic_field"),
            ],
        )

    if _magnetic_field_toroidal_magnitude is not None:
        registry.add_field(
            (ftype, "magnetic_field_toroidal_magnitude"),
            sampling_type="local",
            function=_magnetic_field_toroidal_magnitude,
            units=u,
            validators=[
                ValidateParameter("normal"),
                ValidateParameter("bulk_magnetic_field"),
            ],
        )

    if geometry is Geometry.CARTESIAN:
        registry.alias(
            (ftype, "magnetic_field_toroidal_magnitude"),
            (ftype, "magnetic_field_spherical_phi"),
            units=u,
        )
        registry.alias(
            (ftype, "magnetic_field_toroidal"),
            (ftype, "magnetic_field_spherical_phi"),
            units=u,
            deprecate=("4.1.0", None),
        )
        registry.alias(
            (ftype, "magnetic_field_poloidal"),
            (ftype, "magnetic_field_spherical_theta"),
            units=u,
            deprecate=("4.1.0", None),
        )
    elif (
        geometry is Geometry.CYLINDRICAL
        or geometry is Geometry.POLAR
        or geometry is Geometry.SPHERICAL
    ):
        # These cases should be covered already, just check that they are
        assert (ftype, "magnetic_field_toroidal_magnitude") in registry
        assert (ftype, "magnetic_field_poloidal_magnitude") in registry
    elif geometry is Geometry.GEOGRAPHIC or geometry is Geometry.INTERNAL_GEOGRAPHIC:
        # not implemented
        pass
    elif geometry is Geometry.SPECTRAL_CUBE:
        # nothing to be done
        pass
    else:
        assert_never(Geometry)

    def _alfven_speed(field, data):
        B = data[ftype, "magnetic_field_strength"]
        return B / np.sqrt(mag_factors(B.units.dimensions) * data[ftype, "density"])

    registry.add_field(
        (ftype, "alfven_speed"),
        sampling_type="local",
        function=_alfven_speed,
        units=unit_system["velocity"],
    )

    def _mach_alfven(field, data):
        return data[ftype, "velocity_magnitude"] / data[ftype, "alfven_speed"]

    registry.add_field(
        (ftype, "mach_alfven"),
        sampling_type="local",
        function=_mach_alfven,
        units="dimensionless",
    )

    b_units = registry.ds.quan(1.0, u).units
    if dimensions.current_mks in b_units.dimensions.free_symbols:
        rm_scale = pc.qp.to("C", "SI") ** 3 / (4.0 * np.pi * pc.eps_0)
    else:
        rm_scale = pc.qp**3 / pc.clight
    rm_scale *= registry.ds.quan(1.0, "rad") / (2.0 * np.pi * pc.me**2 * pc.clight**3)
    rm_units = registry.ds.quan(1.0, "rad/m**2").units / unit_system["length"]

    def _rotation_measure(field, data):
        return (
            rm_scale
            * data[ftype, "magnetic_field_los"]
            * data[ftype, "El_number_density"]
        )

    registry.add_field(
        (ftype, "rotation_measure"),
        sampling_type="local",
        function=_rotation_measure,
        units=rm_units,
        validators=[ValidateParameter("axis", {"axis": [0, 1, 2]})],
    )


def setup_magnetic_field_aliases(registry, ds_ftype, ds_fields, ftype="gas"):
    r"""
    This routine sets up special aliases between dataset-specific magnetic
    fields and the default magnetic fields in yt so that unit conversions
    between different unit systems can be handled properly. This is only called
    from the `setup_fluid_fields` method (for grid dataset) or the
    `setup_gas_particle_fields` method (for particle dataset) of a frontend's
    :class:`FieldInfoContainer` instance.

    Parameters
    ----------
    registry : :class:`FieldInfoContainer`
        The field registry that these definitions will be installed into.
    ds_ftype : string
        The field type for the fields we're going to alias, e.g. "flash",
        "enzo", "athena", "PartType0", etc.
    ds_fields : list of strings or string
        The fields that will be aliased. For grid dataset, this should be a
        list of strings corresponding to the components of magnetic field. For
        particle dataset, this should be a single string corresponding to the
        vector magnetic field.
    ftype : string, optional
        The resulting field type of the fields. Default "gas".

    Examples
    --------
    >>> from yt.fields.magnetic_field import setup_magnetic_field_aliases
    >>> class PlutoFieldInfo(ChomboFieldInfo):
    ...     def setup_fluid_fields(self):
    ...         setup_magnetic_field_aliases(
    ...             self, "chombo", ["bx%s" % ax for ax in [1, 2, 3]]
    ...         )
    >>> class GizmoFieldInfo(GadgetFieldInfo):
    ...     def setup_gas_particle_fields(self):
    ...         setup_magnetic_field_aliases(
    ...             self, "PartType0", "MagneticField", ftype="PartType0"
    ...         )
    """
    unit_system = registry.ds.unit_system
    if isinstance(ds_fields, list):
        # If ds_fields is a list, we assume a grid dataset
        sampling_type = "local"
        ds_fields = [(ds_ftype, fd) for fd in ds_fields]
        ds_field = ds_fields[0]
    else:
        # Otherwise, we assume a particle dataset
        sampling_type = "particle"
        ds_field = (ds_ftype, ds_fields)
    if ds_field not in registry:
        return

    # Figure out the unit conversion to use
    if unit_system.base_units[dimensions.current_mks] is not None:
        to_units = unit_system["magnetic_field_mks"]
    else:
        to_units = unit_system["magnetic_field_cgs"]
    units = unit_system[to_units.dimensions]

    # Add fields
    if sampling_type in ["cell", "local"]:
        # Grid dataset case
        def mag_field_from_field(fd):
            def _mag_field(field, data):
                return data[fd].to(field.units)

            return _mag_field

        for ax, fd in zip(registry.ds.coordinates.axis_order, ds_fields):
            registry.add_field(
                (ftype, f"magnetic_field_{ax}"),
                sampling_type=sampling_type,
                function=mag_field_from_field(fd),
                units=units,
            )
    else:
        # Particle dataset case
        def mag_field_from_ax(ax):
            def _mag_field(field, data):
                return data[ds_field][:, "xyz".index(ax)]

            return _mag_field

        for ax in registry.ds.coordinates.axis_order:
            fname = f"particle_magnetic_field_{ax}"
            registry.add_field(
                (ds_ftype, fname),
                sampling_type=sampling_type,
                function=mag_field_from_ax(ax),
                units=units,
            )
            sph_ptypes = getattr(registry.ds, "_sph_ptypes", ())
            if ds_ftype in sph_ptypes:
                registry.alias((ftype, f"magnetic_field_{ax}"), (ds_ftype, fname))


@register_field_plugin
def setup_current_density_vector_fields(registry, ftype = "gas", slice_info = None):
    """
    Derived from vorticity calculation in fluid_vector_fields.py
    This function sets up the current density vector fields calculated by
    Ampere's law: "curl B = j_factor * J" assuming E = 0.
    """
    unit_system = registry.ds.unit_system
    #
    # slice_info would be the left, the right, and the factor.
    # For example, with the old Enzo-ZEUS fields, this would be:
    # slice(None, -2, None)
    # slice(1, -1, None)
    # 1.0
    # Otherwise, we default to a centered difference.
    if slice_info is None:
        sl_left = slice(None, -2, None)
        sl_right = slice(2, None, None)
        div_fac = 2.0
    else:
        sl_left, sl_right, div_fac = slice_info
    sl_center = slice(1, -1, None)

    # Constant factor in Ampere's law depends on the unit system
    # curl B = j_factors * J
    j_factors = {dimensions.magnetic_field_cgs/dimensions.length: 4.0*np.pi/c,
                 dimensions.magnetic_field_mks/dimensions.length: mu_0}

    def _current_density_x(field, data):
        f  = (data[ftype, "magnetic_field_z"][sl_center,sl_right,sl_center] -
              data[ftype, "magnetic_field_z"][sl_center,sl_left,sl_center]) \
              / (div_fac*just_one(data["index", "dy"]))
        f -= (data[ftype, "magnetic_field_y"][sl_center,sl_center,sl_right] -
              data[ftype, "magnetic_field_y"][sl_center,sl_center,sl_left]) \
              / (div_fac*just_one(data["index", "dz"]))
        new_field = data.ds.arr(np.zeros_like(data[ftype, "magnetic_field_z"],
                                              dtype=np.float64), f.units)
        new_field[sl_center, sl_center, sl_center] = f
        return new_field/j_factors[new_field.units.dimensions]

    def _current_density_y(field, data):
        f  = (data[ftype, "magnetic_field_x"][sl_center,sl_center,sl_right] -
              data[ftype, "magnetic_field_x"][sl_center,sl_center,sl_left]) \
              / (div_fac*just_one(data["index", "dz"]))
        f -= (data[ftype, "magnetic_field_z"][sl_right,sl_center,sl_center] -
              data[ftype, "magnetic_field_z"][sl_left,sl_center,sl_center]) \
              / (div_fac*just_one(data["index", "dx"]))
        new_field = data.ds.arr(np.zeros_like(data[ftype, "magnetic_field_z"],
                                              dtype=np.float64), f.units)
        new_field[sl_center, sl_center, sl_center] = f
        return new_field/j_factors[new_field.units.dimensions]

    def _current_density_z(field, data):
        f  = (data[ftype, "magnetic_field_y"][sl_right,sl_center,sl_center] -
              data[ftype, "magnetic_field_y"][sl_left,sl_center,sl_center]) \
              / (div_fac*just_one(data["index", "dx"]))
        f -= (data[ftype, "magnetic_field_x"][sl_center,sl_right,sl_center] -
              data[ftype, "magnetic_field_x"][sl_center,sl_left,sl_center]) \
              / (div_fac*just_one(data["index", "dy"]))
        new_field = data.ds.arr(np.zeros_like(data[ftype, "magnetic_field_z"],
                                              dtype=np.float64), f.units)
        new_field[sl_center, sl_center, sl_center] = f
        return new_field/j_factors[new_field.units.dimensions]

    curl_validators = [ValidateSpatial(1,
                            [(ftype, "magnetic_field_x"),
                             (ftype, "magnetic_field_y"),
                             (ftype, "magnetic_field_z")])]
    # Determine the correct unit for the current density
    if dimensions.current_mks in unit_system.base_units:
        current_density_unit = unit_system["current_mks"]/unit_system["length"]**2
    else:
        current_density_unit = unit_system["current_cgs"]/unit_system["length"]**2
    for ax in 'xyz':
        n = "current_density_%s" % ax
        registry.add_field((ftype, n), sampling_type="cell",
                           function=eval("_%s" % n),
                           units=current_density_unit,
                           validators=curl_validators)
    create_magnitude_field(registry, "current_density", current_density_unit,
                           ftype=ftype, slice_info=slice_info,
                           validators=curl_validators)
