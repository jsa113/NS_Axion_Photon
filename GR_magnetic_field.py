# -*- coding: utf-8 -*-
"""
Created on Thu May 18 09:29:32 2023

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz
"""

from parameters import *
from functions import *
from sympy import LeviCivita
import autograd.numpy as np
from autograd import grad, jacobian
from scipy import optimize

# =============================================================================
# From Gralla et. al. https://iopscience.iop.org/article/10.3847/1538-4357/833/2/258/pdf
# Magnetic field relationships
# =============================================================================
# Make the 4D levi civita tensor
LeviCivita_tensor = np.zeros((4,4,4,4))
for i in range(4):
    for j in range(4):
        for k in range(4):
            for l in range(4):
                LeviCivita_tensor[i, j, k, l] = int(LeviCivita(i, j, k, l)) # As defined in flatspace det(g) = 1
                
def theta_inclined(theta, phi, t):
    "Returns the polar angle in the magnetic field symmetry axis"
    azi = phi - Rot_ns * t
    return np.arccos(np.cos(theta) * np.cos(Theta_m) - np.sin(theta) * np.cos(azi) * np.sin(Theta_m))

def phi_inclined(theta, phi, t):
    "Returns the azimuth angle in the magnetic field symmetry axis"
    azi = phi - Rot_ns * t
    return np.arctan2(np.sin(theta) * np.sin(azi), np.cos(theta) * np.sin(Theta_m) + np.sin(theta) * np.cos(azi) * np.cos(Theta_m))

def Radial_eigenfunction_1(r, r_s=r_s_val):
    "Returns the first radial eigenfunction"
    f = 1. - r_s / r
    return -1. * (3. / (2.*r)) * (3. - 4.*f + np.power(f, 2) + 2.* np.log(f)) / np.power(1. -f, 3)

def mag_alpha(t, r, theta, phi, r_s=r_s_val):
    "Eqn (16) of Gralla 2019"
    B_1 = B_0 / 2 # Below equation (66) of Gralla et. al. 2016
    # mu = B_1 * R_ns ** 2  * Radial_eigenfunction_1(r, r_s) / Radial_eigenfunction_1(R_ns, r_s)
    f_r = 1. - r_s / r
    f_R = 1. - r_s / R_ns
    c = (R_ns / r) * ((3. - 4.*f_r + np.power(f_r, 2) + 2.* np.log(f_r)) / (3. - 4.*f_R + np.power(f_R, 2) + 2.* np.log(f_R))) * (np.power(1. - f_R, 3) / np.power(1. -f_r, 3))
    mu = B_1 * R_ns **2 * c
    theta_prime = theta_inclined(theta, phi, t)
    return mu * np.power(np.sin(theta_prime), 2)
    
def mag_beta(t, r, theta, phi):
    'Eqn (16) of Gralla 2019'
    return phi_inclined(theta, phi, t)

dalphadt = grad(mag_alpha, argnum=0) #partial alpha / partial t
dalphadr = grad(mag_alpha, argnum=1) #partial alpha / partial r
dalphadtheta = grad(mag_alpha, argnum=2) #partial alpha / partial theta
dalphadphi = grad(mag_alpha, argnum=3) #partial alpha / partial phi

dbetadt = grad(mag_beta, argnum=0) #partial beta / partial t
dbetadr = grad(mag_beta, argnum=1) #partial beta / partial r
dbetadtheta = grad(mag_beta, argnum=2) #partial beta / partial theta
dbetadphi = grad(mag_beta, argnum=3) #partial beta / partial phi

def Number_Density(t, r, theta, phi, r_s=r_s_val):
    """
    equations from http://arxiv.org/abs/1904.11534
    """
    omega = Rot_ns
    I_hat = 2. / 5. * M_ns * R_ns ** 2 # Assuming uniform solid sphere
    omega_z = 2. * I_hat * G / (r**3) * omega # eqn(2) frame drag frequency (times by G to fix units)
    f = 1. - r_s / r
    
    first_term = dalphadtheta(t, r, theta, phi, r_s) * grad(dbetadphi, argnum=2)(t, r, theta, phi)
    second_term = dbetadtheta(t, r, theta, phi) * grad(dalphadphi, argnum=2)(t, r, theta, phi, r_s)
    
    f3 = lambda t, r, theta, phi: np.sin(theta) * dbetadtheta(t, r, theta, phi)
    third_term = dalphadphi(t, r, theta, phi, r_s) * grad(f3, argnum=2)(t, r, theta, phi) / np.sin(theta)
    
    f4 = lambda t, r, theta, phi: np.power(r, 2) * dalphadr(t, r, theta, phi, r_s)
    f5 = lambda t, r, theta, phi: np.sin(theta) * dalphadtheta(t, r, theta, phi, r_s)
    fourth_term = dbetadphi(t, r, theta, phi) * (f * grad(f4, argnum=1)(t, r, theta, phi) + grad(f5, argnum=2)(t, r, theta, phi) / np.sin(theta))
    
    # From eqn. (4a)
    J_up_t = -1. * (omega - omega_z) / (np.power(f, 1/2) * np.power(r, 2)) * (first_term - second_term - third_term + fourth_term)

    # From the result below eqn. (12)
    num_density = J_up_t / q_e # Charge density divided by charge
    
    return num_density

def electromagnetic_tensor(t, r, theta, phi):
    """Returns the electromagnetic tensor using the relation 
    F_uv = d_u alpha d_v beta - d_v alpha d_u beta
    
    Arguments:
        t = time
        r = radius
        theta = polar angle
        phi = azimuthal angle
        
    Returns:
        F_ab = The electromangetic tensor at the given point.
    """
    # Collecting the partial derivatives into vectors
    duda = np.array([dalphadt(t, r, theta, phi), dalphadr(t, r, theta, phi), dalphadtheta(t, r, theta, phi), dalphadphi(t, r, theta, phi)])
    dudb = np.array([dbetadt(t, r, theta, phi), dbetadr(t, r, theta, phi), dbetadtheta(t, r, theta, phi), dbetadphi(t, r, theta, phi)])
    
    # Taking the sum of the previously made vectors
    F_ab = np.einsum('i,j->ij',duda, dudb) - np.einsum('i,j->ji',duda, dudb)
            
    return F_ab

def Magnetic_Field(t, r, theta, phi, r_s=r_s_val):
    "Return the magnetic field vector at (r, theta, phi) and time t"
    
    f = 1. - r_s / r #Schwarzschild function
    
    F_ab = electromagnetic_tensor(t, r, theta, phi) # F_ab
    
    U = np.array([np.sqrt(f * c_sq), 0, 0, 0]) # Global velocity in the pulsars rotating frame U_a
    
    # Det of metric due to not being in a flatspace time on the Levi-Civita tensor with raised indicies https://en.wikipedia.org/wiki/Levi-Civita_symbol Sec: Levi-Civita tensors
    # B = 1/2 * np.einsum('ijkl,jk,l', LeviCivita_tensor, F_ab, U) * (-1. / np.sqrt(np.abs(np.linalg.det(metric(r, theta, phi)))))
    B = 1/2 * np.einsum('ijkl,ij,k', LeviCivita_tensor, F_ab, U) * (-1. / np.sqrt(np.abs(np.linalg.det(metric(r, theta, phi, r_s)))))
    return B

def Omega_p_sq(t, r, theta, phi, r_s=r_s_val):
    "Compute the square of wp. Take absolute as we consider regions of electrons and positrons"
    w_p_sq = np.abs(Number_Density(t, r, theta, phi, r_s) * q_e ** 2 / (m_e * epsilon_0))
    return w_p_sq

def Omega_p(t, r, theta, phi, r_s=r_s_val):
    "Compute the square of wp. Take absolute as we consider regions of electrons and positrons"
    w_p = np.power(Omega_p_sq(t, r, theta, phi, r_s), 1/2)
    return w_p

# Plasma frequency partial derivatives
dw_p_dr = grad(Omega_p, argnum=1) #partial omega / partial r
dw_p_dtheta = grad(Omega_p, argnum=2) #partial omega / partial theta
dw_p_dphi = grad(Omega_p, argnum=3) #partial omega / partial phi

# =============================================================================
# Solution of the axion-photon conversion surface
# =============================================================================
def Conversion_Surface_scalar(theta, phi, t, r_s=r_s_val):    
    """
    Computes the radius at which photon axion conversion occurs m_a/hbar = w_p when m_a is given in eV
    
    Use for scalar inputs
    """
    f1 = lambda r: Omega_p_sq(t, r, theta, phi) - m_a**2 
    try:
        # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
        r_conversion = optimize.root_scalar(f1, bracket=[r_s *1.1, 100 * R_ns]).root
    except ValueError:
        r_conversion = None
    return r_conversion

def Conversion_Surface(theta_array, phi_array, t_array, r_s=r_s_val):
    """
    Computes the radius at which photon axion conversion occurs m_a/hbar = w_p when m_a is given in eV
    
    Use for array inputs
    """
    try:
        r_conversion = np.zeros(theta_array.size)
        for i, theta in enumerate(theta_array):
            phi = phi_array[i]
            t = t_array[i]
            f1 = lambda r: Omega_p_sq(t, r * R_ns, theta, phi, r_s) - m_a**2
            try:
                # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
                r_conversion[i] = optimize.root_scalar(f1, bracket=[r_s / R_ns *1.1, 100]).root * R_ns
            except ValueError:
                r_conversion = None
    # In the case that the inputs are not arrays and are instead floats
    except:
        f1 = lambda r: Omega_p_sq(t_array, r, theta_array, phi_array, r_s) - m_a**2 
        try:
            # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
            r_conversion = optimize.root_scalar(f1, bracket=[r_s *1.1, 100 * R_ns]).root
        except ValueError:
            r_conversion = None
    return r_conversion

# def Conversion_Surface(theta_array, phi_array, t_array, r_s=r_s_val):
#     """
#     Computes the radius at which photon axion conversion occurs m_a/hbar = w_p when m_a is given in eV
    
#     Use for array inputs
#     """
#     try:
#         r_conversion = np.zeros(theta_array.size)
#         for i, theta in enumerate(theta_array):
#             phi = phi_array[i]
#             t = t_array[i]
#             f1 = lambda r: Omega_p_sq(t, r * R_ns, theta, phi, r_s) - m_a**2
#             try:
#                 # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces 
#                 r_conversion[i] = optimize.root_scalar(f1, bracket=[r_s / R_ns *1.1, 100]).root * R_ns # Find the r when w_p^2=m_a^2
#             except ValueError:
#                 r_conversion[i] = None # Otherwise conversion does not occur
#     # In the case that the inputs are not arrays and are instead floats
#     except:
#         f1 = lambda r: Omega_p_sq(t_array, r, theta_array, phi_array, r_s) - m_a**2 
#         try:
#             # The upper limit of the bracket may need to be changed to accommodate larger conversion surfaces
#             r_conversion = optimize.root_scalar(f1, bracket=[r_s *1.1, 100 * R_ns]).root # Find the r when w_p^2=m_a^2
#         except ValueError:
#             r_conversion = None # Otherwise conversion does not occur
#     return r_conversion

# =============================================================================
# Extra functions
# =============================================================================

def magnetic_field_unit_vec(t, r, theta, phi):
    "Returns the magnetic field 4-unit vector in curved space-time"
    B = Magnetic_Field(t, r, theta, phi) # (B^t, B^r, B^theta, B^phi)
    
    g_ab = metric(r, theta, phi) # Metric
    
    B_sq = np.matmul(np.matmul(g_ab, B), B) # Magnitude of the spatial part of the magnetic field (B_u*B^u)
    
    b =  B / np.sqrt(B_sq) # b^a
    return b

# =============================================================================
# Partial derivaties for the ray tracing equations
# =============================================================================

dw_p_sq_dt = grad(Omega_p_sq, argnum=0) #partial omega / partial t
dw_p_sq_dr = grad(Omega_p_sq, argnum=1) #partial omega / partial r
dw_p_sq_dtheta = grad(Omega_p_sq, argnum=2) #partial omega / partial theta
dw_p_sq_dphi = grad(Omega_p_sq, argnum=3) #partial omega / partial phi

db_dt = jacobian(magnetic_field_unit_vec, argnum=0) #partial b / partial t
db_dr = jacobian(magnetic_field_unit_vec, argnum=1) #partial b / partial r
db_dtheta = jacobian(magnetic_field_unit_vec, argnum=2) #partial b / partial theta
db_dphi = jacobian(magnetic_field_unit_vec, argnum=3) #partial b / partial phi




