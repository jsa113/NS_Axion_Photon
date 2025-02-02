# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:43:05 2022

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz

The purpose of this file is to consolidate all the choices and variables for 
the ray tracing and power in one spot.

Includes:
    - Simulation choices for dispersion method
    - Importing of simulation parameter file, giving the parameters for the simulation
    - Unit conversion from SI to natural units
"""

from numpy import pi, sqrt, loadtxt, linspace, tile, repeat, stack # Importing useful functions

# =============================================================================
# Simulation function choices
# =============================================================================
dispersion_method = 'unmagnetised' # Chose from 'vacuum', 'unmagnetised' or 'magnetised'
metric_choice = 'schwarzschild' # Chose from 'flat' or 'schwarzschild'
mag_field_choice = 'GJ' # Chose the magnetic field from 'GJ' or 'GR'

save_photon_paths = True # Use to save the photon data
save_plots = True # use to save the plots

# =============================================================================
# Observers theta and phi angle from the neutron star's rotation axis
# =============================================================================

phase_array = linspace(0., 0., 1) # Choose the phase fractions to run the simulation at

Obs_theta_array = linspace(36 + 1e-6, 90 + 1e-6, 1) * pi / 180. # Polar angle that the detector makes to the rotation axis in radians
Obs_phi_array = linspace(0., 0., 1) # Azimuthal anglethat the detector makes to the rotation axis in radians

Obs_r0_n = 500. # Distance of dectector as a multiple of the neutron star radius

# Converts the above data to distinct simulations which are iterated through by the main code
Detector_data_array = stack((tile(phase_array, Obs_theta_array.size * Obs_phi_array.size), 
                        tile(repeat(Obs_theta_array, phase_array.size), Obs_phi_array.size), 
                        repeat(Obs_phi_array, phase_array.size * Obs_theta_array.size))).T

# Detector resolution
coarse_resolution = 5 # Size of coarse search mesh
fine_resolution = 3 # Size of fine meshes within the coarse mesh pixels

total_resolution = coarse_resolution * fine_resolution # Total number of fine pixels of the detector
num_coarse_pixels = coarse_resolution ** 2 # Total number of corase pixels of the detector

# Sets the relative region size for the coarse search termination event. 
# =1 will be only an intercept with conversion surface, set >1 for a better coarse search
coarse_search_rel_tol = 1.0075

# parameter_set = 'battye_2021_0.5ueV.txt'
parameter_set = 'mcdonald_2023_10ueV_2_2_ns.txt' # Used to choose the parameter file to use.
# =============================================================================
# Load the parameter set for the simulation from a file in this following order 
# and units

# Axion parameters:

    # m_a: Mass of axion in eV/c^2
    
    # g_ayy: Axion-photon coupling constant in GeV^-1

# Physical properties of the Neutron Star (NS)

    # Period: Period of rotation for the NS in seconds
    
    # R_ns: Neutron star radius in metres
    # M_ns: Neutron star mass as a ratio with solar mass
    # B_0: Magnetic field strength at surface in Tesla
    
    # Theta_m: Misalignment angle of magnetic feild and spin axis in degrees
    
    # DM_v_0: Dark matter dispersion velocity in m/s
    
    # DM_pho_inf: Density of dark matter infinity far from the neutron star (kg/m^3)
# =============================================================================

parameters = loadtxt('./parameter_sets/' + parameter_set) # Load the simulation parameters

m_a, g_ayy, Period, R_ns, M_ns_ratio, B_0, Theta_m, DM_v_0, DM_pho_inf = parameters # Assign parameters to appriate variable names

# m_a = 10e-6
# M_ns_ratio = 2.2
# Predefine some useful quanities for the simulations
m_a_sq = m_a ** 2 # Axion mass squared
Rot_ns = 2. * pi / Period # Period converted to angular velocity of the NS
Theta_m = Theta_m / 180. * pi # Misalignment angle converted from degrees to radians

# =============================================================================
# The following code defines universal constants and converts the parameters
# =============================================================================
# =============================================================================
# Universal constants in SI units from scipy imports
# =============================================================================
from scipy.constants import physical_constants
from scipy.constants import c # Speed of light in m/s
from scipy.constants import hbar # Reduced Planck's constant in Js
from scipy.constants import e as q_e # Charge of an electron in C
from scipy.constants import m_e # Mass of an electron in kg
from scipy.constants import epsilon_0 # Permittivity of free space
from scipy.constants import G # Newtonians gravitional constant
from scipy.constants import eV # Electron volt in joules
from scipy.constants import alpha # Fine structure constant
from astropy.constants import M_sun # Solar mass

# =============================================================================
# Conversion factors from SI to natural units (c = hbar = e0 = 1)
# Factors taken from multiple sources 
#   Jaffe MIT Notes 2007
#   http://ilan.schnell-web.net/physics/natural.pdf
# =============================================================================
m_to_per_eV = eV / (hbar * c) # Metres converted to per eV 
s_to_per_eV = eV / hbar # Seconds converted to per eV 
kg_to_eV = c** 2 / eV # kg converted to eV
Watts_to_eV_sq = 1. / (eV * s_to_per_eV) # Joules per second converted to eV^2
q_e = sqrt(4. * pi * alpha) # Charge of electron in natural units
M_sol = M_sun.value * kg_to_eV # Solar mass converted from kg to eV

# Gauss converted to eV^2 from Introduction to Elementary Particle Physics and Jaffe MIT Notes 2007
# Using the SI constants imported above and converting them to cgs to work with gauss unit
# gauss_to_eV_sq = (hbar * 1000. * 1e2**2 * c * 1e2) ** (3/2) / (eV * 1000. * 1e2**2)**2 / sqrt(4. * pi) 
# gauss_to_eV_sq = (hbar * 1000. * 1e2**2 * c * 1e2) ** (3/2) / (eV * 1000. * 1e2**2)**2 
# B_0 = B_0 * 1e4 # Convert from Tesla to Gauss first, going from SI to CGS
# B_0 = B_0 * gauss_to_eV_sq # Convert from Gauss to eV^2

A_to_eV = q_e * hbar / eV**2 # Ampere (C/s) converted to eV
tesla_to_eV_sq = kg_to_eV / A_to_eV / s_to_per_eV**2 # Tesla converted to eV^2

# =============================================================================
# Convert the simulation parameters to natural units using above factors.
# =============================================================================
Rot_ns = Rot_ns / s_to_per_eV
Period = Period * s_to_per_eV
R_ns = R_ns * m_to_per_eV
M_ns = M_sol * M_ns_ratio
B_0 = parameters[5] * tesla_to_eV_sq

Obs_r0 = R_ns * Obs_r0_n # Initial radial distance of the centre of the detector

Int_time = Obs_r0 / m_a # Total integration time of solve_ivp

DM_v_0 = DM_v_0 / c # Convert to unitless
DM_pho_inf = DM_pho_inf * kg_to_eV / (m_to_per_eV ** 3) # Convert to eV^4

# =============================================================================
# Update the Universal constants in natural units measured in varying powers of 
# eV where c = hbar = e0 = 1
# =============================================================================
c = 1.
c_sq = 1.
hbar = 1.
hbar_ev = 1.
epsilon_0 = 1 # permittivity of free space 
m_e = physical_constants['electron mass energy equivalent in MeV'][0] * 1e6 # Mass of an electron in eV
m_pl = physical_constants['Planck mass energy equivalent in GeV'][0] * 1e9 # Planck mass in eV
G = 1. / m_pl**2 # Universal grativational constant in per eV^2


# The Schwarzschild radius of the neutron star in natural units
if metric_choice == 'flat': # Makes the Schwarzschild metric Minkowski instead
    r_s_val = 0. 
else:
    r_s_val = 2. * G * M_ns / c ** 2 # In eV^-1



















