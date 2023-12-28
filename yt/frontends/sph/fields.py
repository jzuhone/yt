from yt._typing import KnownFieldsT
from yt.fields.field_info_container import FieldInfoContainer
from yt.fields.species_fields import setup_species_fields


class SPHFieldInfo(FieldInfoContainer):
    known_particle_fields: KnownFieldsT = (
        ("Mass", ("code_mass", ["particle_mass"], None)),
        ("Masses", ("code_mass", ["particle_mass"], None)),
        ("ParticleIDs", ("", ["particle_index"], None)),
        ("InternalEnergy", ("code_specific_energy", ["specific_thermal_energy"], None)),
        ("SmoothingLength", ("code_length", ["smoothing_length"], None)),
        ("Density", ("code_mass / code_length**3", ["density"], None)),
        ("MaximumTemperature", ("K", [], None)),
        ("Temperature", ("K", ["temperature"], None)),
        ("Epsilon", ("code_length", [], None)),
        ("Metals", ("code_metallicity", ["metallicity"], None)),
        ("Metallicity", ("code_metallicity", ["metallicity"], None)),
        ("Phi", ("code_length", [], None)),
        ("Potential", ("code_velocity**2", ["gravitational_potential"], None)),
        ("StarFormationRate", ("Msun / yr", ["star_formation_rate"], None)),
        ("FormationTime", ("code_time", ["creation_time"], None)),
        ("Metallicity_00", ("", ["metallicity"], None)),
        ("InitialMass", ("code_mass", [], None)),
        ("TrueMass", ("code_mass", [], None)),
        ("ElevenMetalMasses", ("code_mass", [], None)),
        ("ColdFraction", ("", ["cold_fraction"], None)),
        ("HotTemperature", ("code_temperature", ["hot_temperature"], None)),
        ("CloudFraction", ("", ["cold_fraction"], None)),
        ("HotPhaseTemperature", ("code_temperature", ["hot_temperature"], None)),
    )

    def __init__(self, ds, field_list, slice_info=None):
        super().__init__(ds, field_list, slice_info=slice_info)
        vfields = (
            (
                self.ds._particle_coordinates_name,
                ("code_length", ["particle_position"], None),
            ),
            (
                self.ds._particle_velocity_name,
                ("code_velocity", ["particle_velocity"], None),
            ),
        )
        if self.ds._particle_magnetic_name is not None:
            vfields += (
                (
                    self.ds._particle_magnetic_name,
                    ("code_magnetic", ["particle_magnetic_field"], None),
                ),
            )
        self.known_particle_fields += vfields

    def setup_particle_fields(self, ptype, *args, **kwargs):
        super().setup_particle_fields(ptype, *args, **kwargs)
        setup_species_fields(self, ptype)

    def setup_fluid_index_fields(self):
        pass
