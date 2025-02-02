# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:29:21 2023

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz
"""

from parameters import *
from functions import *
import autograd.numpy as np
from autograd import grad, jacobian


# =============================================================================
# From Witte ET AL. Plasma frequency relationship
# =============================================================================
def Magnetic_Field_flatspace(t, r, theta, phi):
    """Return the GJ dipole magnetic field vector at (r, theta, phi) and time t for
    a Minkowski metric.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    returns:
        A 4-vector containing the components of the magnetic field at the given
        point in space. First (0th) component is zero.
    """
    
    B_r = B_0 * (R_ns / r) ** 3 * (np.cos(Theta_m) * np.cos(theta) + np.sin(Theta_m) * np.sin(theta) * np.cos(phi - Rot_ns * t))
    B_theta = (B_0 / 2.) * (R_ns / r) ** 3 * (np.cos(Theta_m) * np.sin(theta) - np.sin(Theta_m) * np.cos(theta) * np.cos(phi - Rot_ns * t))
    B_phi = (B_0 / 2.) * (R_ns / r) ** 3 * np.sin(Theta_m) * np.sin(phi - Rot_ns * t)
    try:
        if B_r.size > 1:
            return np.array([np.full(B_r.size, 0), B_r, B_theta, B_phi])
        else:
            return np.array([0., B_r, B_theta, B_phi])
    except:
        return np.array([0., B_r, B_theta, B_phi])

def Magnetic_Field(t, r, theta, phi, r_s=r_s_val):
    """Modify the newtionian derived GJ magnetic field to account for the Schwarzschild metric
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    returns:
        A 4 vector containing the components of the magnetic field at the given
        point in space. First (0th) component is zero.
    """
    _, B_r, B_theta, B_phi = Magnetic_Field_flatspace(t, r, theta, phi)
    B_r = np.sqrt(1. - r_s / r) * B_r
    B_theta = (1. / r) * B_theta # Changed with 1/r term
    B_phi = (1. / (r * np.sin(theta))) * B_phi # Changed with 1/(r*sin(theta)) term
    return np.array([0, B_r, B_theta, B_phi])
    # return np.matmul(np.power(np.abs(inv_metric(r, theta, phi)), 1/2), B_flatspace) # Accounting for metric in GR

def Number_Density(t, r, theta, phi, r_s=r_s_val):
    """The number density in SI units of the GJ pulsar model.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        r_s = Schwarzschild radius
        
    Returns:
        num_density = The charged particle number density at the given point in
        space. Negative number density indicates negative charged particels 
        (i.e. electrons).
    """
    _, B_r, B_theta, B_phi = Magnetic_Field_flatspace(t, r, theta, phi) #Dipole Magnetic field
    
    B_z = np.cos(theta) * B_r - np.sin(theta) * B_theta
    w = np.array([0, 0, Rot_ns]) #Angular velocity vector in z-hat direction
    num_density = 2. *epsilon_0 * w[2] * B_z / (q_e * (1. - (Rot_ns*r*np.sin(theta))**2 / c_sq)) # Number density of electrons and positrons
    ## Other methods or equations for number density
    # num_density = B_0 * Rot_ns / (2 * q_e) * (R_ns / r) ** 3 * (np.cos(Theta_m) + 3 * np.cos(Theta_m) * np.cos(2 * theta) + 3 * np.sin(Theta_m) * np.cos(phi - Rot_ns * t) * np.sin(2 * theta))
    # m_dot_r = np.cos(Theta_m) * np.cos(theta) + np.sin(Theta_m) * np.sin(theta)  * np.cos(phi - Rot_ns * t)
    # num_density = B_0 * Rot_ns / (q_e) * (R_ns / r) ** 3 * (3 * np.cos(theta) * m_dot_r - np.cos(Theta_m)) # From McDonald and Witte 2023
    return num_density

def Omega_p_sq(t, r, theta, phi, r_s=r_s_val):
    """Compute the square of the plasma frequency. Take the absolute values as 
    number density can be both negative or positive, but plasma frequency 
    (and its square) is always positive.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        r_s = Schwarzschild radius
        
    Return:
        w_p_sq = The plasma fequency squared at the given point in space.
    """
    w_p_sq = np.abs(Number_Density(t, r, theta, phi, r_s) * q_e**2 / (m_e * epsilon_0))
    return w_p_sq

def Omega_p(t, r, theta, phi, r_s=r_s_val):
    """Compute the plasma frequency.
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        r_s = Schwarzschild radius
        
    Return:
        w_p = The plasma fequency at the given point in space.
    """
    w_p = np.power(Omega_p_sq(t, r, theta, phi, r_s), 1/2)
    return w_p

# =============================================================================
# Plasma frequency partial derivatives. To be passed onto main.py
# =============================================================================
dw_p_dr = grad(Omega_p, argnum=1) #partial omega / partial r
dw_p_dtheta = grad(Omega_p, argnum=2) #partial omega / partial theta
dw_p_dphi = grad(Omega_p, argnum=3) #partial omega / partial phi

# =============================================================================
# Analytical solution of the axion-photon conversion surface
# =============================================================================
def Conversion_Surface(theta, phi, t):
    """Computes the radius at which photon axion conversion occurs m_a/hbar = w_p 
    when m_a is given in eV. Can take arguments as vectors.
    
    Arguments:
        theta = polar angle from the NS norht pole
        phi = azimuthal angle from the positive x-axis
        t = current time
        
    Returns:
        r_conversion = the distance from the centre of the NS to the edge of 
                       the conversion surface at the given angles and time
    
    """
    w_p_sq = Omega_p_sq(t, 1, theta, phi) 
    r_conversion = np.power(hbar_ev ** 2 * w_p_sq / m_a ** 2, 1/3)
    return r_conversion

def Conversion_Surface_scalar(theta, phi, t):
    """Same as Conversion_Surface but needed for code compatibility with the GR
    magnetic field functions."""
    w_p_sq = Omega_p_sq(t, 1, theta, phi)
    r_conversion = np.power(hbar_ev ** 2 * w_p_sq / m_a ** 2, 1/3)
    return r_conversion

# =============================================================================
# Extra functions
# =============================================================================

def magnetic_field_unit_vec(t, r, theta, phi):
    """Returns the magnetic field 4-unit vector in the assoicated metric space-time
    
    Arguments:
        t = time
        r = radius
        theta = polar angle 
        phi = azimuthal angle
        
    Returns:
        b = the unit vector pointing in the direction of the magnetic field
    """
    B = Magnetic_Field(t, r, theta, phi) # (B^t, B^r, B^theta, B^phi)
    
    g_ab = metric(r, theta, phi) # Metric
    
    B_sq = np.matmul(np.matmul(g_ab, B), B) # Magnitude of the spatial part of the magnetic field (B_u*B^u)
    
    b =  B / np.sqrt(B_sq) # b^a
    return b

def magnetic_field_angle(x, k):
    """Returns the angle between the magnetic field and the photon's momentum.
    
    Arguments:
        x = photon 4-position
        k = photon 4-momentum
        
    Returns:
        theta_k = angle between magnetic field and the photon's momentum measured
    """
    
    g_ab = metric(x[1], x[2], x[3])
    inv_g_ab = inv_metric(x[1], x[2], x[3])
    
    B = Magnetic_Field(x[0], x[1], x[2], x[3])
    
    k_spac_norm = np.sqrt(np.einsum('ij,i,j', inv_g_ab[1:, 1:], k[1:], k[1:]))
    B_sq = np.einsum('ij,i,j', g_ab, B, B)
    dot_product = np.dot(B, k) / (np.sqrt(B_sq) * k_spac_norm)
    
    theta_k = np.arccos(dot_product)
    return theta_k

# =============================================================================
# Partial derivaties for the ray tracing equations
# =============================================================================

dw_p_sq_dt = grad(Omega_p_sq, argnum=0) #partial omega / partial t
dw_p_sq_dr = grad(Omega_p_sq, argnum=1) #partial omega / partial r
dw_p_sq_dtheta = grad(Omega_p_sq, argnum=2) #partial omega / partial theta
dw_p_sq_dphi = grad(Omega_p_sq, argnum=3) #partial omega / partial phi

dConversion_Surface_scalar_dtheta = grad(Conversion_Surface_scalar, argnum=0)
dConversion_Surface_scalar_dphi = grad(Conversion_Surface_scalar, argnum=1)

db_dt = jacobian(magnetic_field_unit_vec, argnum=0) #partial b / partial t
db_dr = jacobian(magnetic_field_unit_vec, argnum=1) #partial b / partial r
db_dtheta = jacobian(magnetic_field_unit_vec, argnum=2) #partial b / partial theta
db_dphi = jacobian(magnetic_field_unit_vec, argnum=3) #partial b / partial phi


