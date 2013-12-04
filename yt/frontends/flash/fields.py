"""
FLASH-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import numpy as np
from yt.fields.field_info_container import \
    FieldInfoContainer
from yt.utilities.physical_constants import \
    kboltz, mh, Na
from yt.data_objects.yt_array import \
    YTArray

# Common fields in FLASH: (Thanks to John ZuHone for this list)
#
# dens gas mass density (g/cc) --
# eint internal energy (ergs/g) --
# ener total energy (ergs/g), with 0.5*v^2 --
# gamc gamma defined as ratio of specific heats, no units
# game gamma defined as in , no units
# gpol gravitational potential from the last timestep (ergs/g)
# gpot gravitational potential from the current timestep (ergs/g)
# grac gravitational acceleration from the current timestep (cm s^-2)
# pden particle mass density (usually dark matter) (g/cc)
# pres pressure (erg/cc)
# temp temperature (K) --
# velx velocity x (cm/s) --
# vely velocity y (cm/s) --
# velz velocity z (cm/s) --

b_units = "code_magnetic"
pres_units = "code_mass/(code_length*code_time**2)"
erg_units = "code_mass * (code_length/code_time)**2"
rho_units = "code_mass / code_length**3"

class FLASHFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("velx", ("code_length/code_time", ["velocity_x"], None)),
        ("vely", ("code_length/code_time", ["velocity_y"], None)),
        ("velz", ("code_length/code_time", ["velocity_z"], None)),
        ("dens", ("code_mass/code_length**3", ["density"], None)),
        ("temp", ("code_temperature", ["temperature"], None)),
        ("pres", (pres_units, ["pressure"], None)),
        ("gpot", ("code_length**2/code_time**2", ["gravitational_potential"], None)),
        ("gpol", ("code_length**2/code_time**2", [], None)),
        ("eint", ("code_length**2/code_time**2", ["thermal_energy"], None)),
        ("ener", ("code_length**2/code_time**2", ["total_energy"], None)),
        ("tion", ("code_temperature", [], None)),
        ("tele", ("code_temperature", [], None)),
        ("trad", ("code_temperature", [], None)),
        ("pion", (pres_units, [], None)),
        ("pele", (pres_units, [], "Electron Pressure, P_e")),
        ("prad", (pres_units, [], "Radiation Pressure")),
        ("eion", (erg_units, [], "Ion Internal Energy")),
        ("eele", (erg_units, [], "Electron Internal Energy")),
        ("erad", (erg_units, [], "Radiation Internal Energy")),
        ("pden", (rho_units, [], None)),
        ("depo", ("code_length**2/code_time**2", [], None)),
        ("ye", ("code_length**2/code_time**2", [], None)),
        ("magp", (pres_units, [], None)),
        ("divb", ("code_magnetic*code_length", [], None)),
        ("game", ("", [], "\gamma_e\/\rm{(ratio\/of\/specific\/heats)}")),
        ("gamc", ("", [], "\gamma_c\/\rm{(ratio\/of\/specific\/heats)}")),
        ("flam", ("", [], None)),
        ("absr", ("", [], "Absorption Coefficient")),
        ("emis", ("", [], "Emissivity")),
        ("cond", ("", [], "Conductivity")),
        ("dfcf", ("", [], "Diffusion Equation Scalar")),
        ("fllm", ("", [], "Flux Limit")),
        ("pipe", ("", [], "P_i/P_e")),
        ("tite", ("", [], "T_i/T_e")),
        ("dbgs", ("", [], "Debug for Shocks")),
        ("cham", ("", [], "Chamber Material Fraction")),
        ("targ", ("", [], "Target Material Fraction")),
        ("sumy", ("", [], None)),
        ("mgdc", ("", [], "Emission Minus Absorption Diffusion Terms")),
        ("magx", (b_units, ["magnetic_field_x"], "B_x")),
        ("magy", (b_units, ["magnetic_field_y"], "B_y")),
        ("magz", (b_units, ["magnetic_field_z"], "B_z")),
    )

    known_particle_fields = (
        ("particle_posx", ("code_length", ["particle_position_x"], None)),
        ("particle_posy", ("code_length", ["particle_position_y"], None)),
        ("particle_posz", ("code_length", ["particle_position_z"], None)),
        ("particle_velx", ("code_length/code_time", ["particle_velocity_x"], None)),
        ("particle_vely", ("code_length/code_time", ["particle_velocity_y"], None)),
        ("particle_velz", ("code_length/code_time", ["particle_velocity_z"], None)),
        ("particle_tag", ("", ["particle_index"], None)),
        ("particle_mass", ("code_mass", ["particle_mass"], None)),
    )

    def setup_fluid_fields(self):
        # Now we conditionally load a few other things.
        #if self.pf.parameters["MultiSpecies"] > 0:
        #    self.setup_species_fields()
        #self.setup_energy_field()
        for i in range(1, 1000):
            self.add_output_field(("flash", "r{0:03}".format(i)), 
                units = "",
                display_name="Energy Group {0}".format(i))

def _ThermalEnergy(fields, data) :
    try:
        return data["eint"]
    except:
        pass
    try:
        return data["Pressure"] / (data.pf.gamma - 1.0) / data["Density"]
    except:
        pass
    if data.has_field_parameter("mu") :
        mu = data.get_field_parameter("mu")
    else:
        mu = 0.6
    return kboltz*data["Density"]*data["Temperature"]/(mu*mh) / (data.pf.gamma - 1.0)

def _TotalEnergy(fields, data) :
    try:
        etot = data["ener"]
    except:
        etot = data["ThermalEnergy"] + 0.5 * (
            data["x-velocity"]**2.0 +
            data["y-velocity"]**2.0 +
            data["z-velocity"]**2.0)
    try:
        etot += data['magp']/data["Density"]
    except:
        pass
    return etot

# See http://flash.uchicago.edu/pipermail/flash-users/2012-October/001180.html
# along with the attachment to that e-mail for details
def GetMagRescalingFactor(pf):
    if pf['unitsystem'].lower() == "cgs":
         factor = 1
    elif pf['unitsystem'].lower() == "si":
         factor = np.sqrt(4*np.pi/1e7)
    elif pf['unitsystem'].lower() == "none":
         factor = np.sqrt(4*np.pi)
    else:
        raise RuntimeError("Runtime parameter unitsystem with "
                           "value %s is unrecognized" % pf['unitsystem'])
    return factor

## Derived FLASH Fields
def _nele(field, data):
    return data['dens'] * data['ye'] * Na
#add_field('nele', function=_nele, take_log=True, units="cm**-3")
#add_field('edens', function=_nele, take_log=True, units="cm**-3")

def _nion(field, data):
    return data['dens'] * data['sumy'] * Na
#add_field('nion', function=_nion, take_log=True, units="cm**-3")


def _abar(field, data):
    try:
        return 1.0 / data['sumy']
    except:
        pass
    return data['dens']*Na*kboltz*data['temp']/data['pres']
#add_field('abar', function=_abar, take_log=False)
	

def _NumberDensity(fields,data) :
    try:
        return data["nele"]+data["nion"]
    except:
        pass
    return data['pres']/(data['temp']*kboltz)
#add_field("NumberDensity", function=_NumberDensity,
#        units=r'\rm{cm}^{-3}')


