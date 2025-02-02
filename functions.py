# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 12:43:05 2022

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz

Functions used by a varity of scripts written to produce axion to photon simulation
conversion data around a Neutron star.

The purpose of this file is to consolidate additional functions in one spot.
Decreasing the bulk of the main simulation file and allowing other following
data processing to use the functions contained within here.

Functions:
    Rot_mat(theta, phi)
    Rot_mat_inv(theta, phi)
    xyz_proj(theta, phi)
    Rot_x(angle)
    Rot_y(angle)
    Rot_z(angle)
    cart_to_spher(x, y, z)
    spher_to_cart(x, y, z)
    metric(r, theta, phi)
    inv_metric(r, theta, phi)
    Global_velocity_vector(r, theta, phi)
    make_save_dir(path, simulation_name)
"""

from parameters import *
import autograd.numpy as np
import os
from numpy import char

# =============================================================================
# Rotation functions to move between coordinate systems
# =============================================================================

def Rot_mat(theta, phi): 
    """
    Takes an input of polar angles and returns the 3x3 rotation matrix to go 
    from cartesian basis vectors to spherical basis vectors.
    
    Arguments:
        theta = polar angle from positive z-axis,
        phi = azimuthal angle from positive x-axis increasing anticlockwise (towards pos. y-axis)
        
    Returns:
        Rot_mat = 3x3 array which will do a rotation of theta and phi
    """
    return np.array([[np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], 
                     [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)], 
                     [ -np.sin(phi), np.cos(phi), 0]])

def Rot_mat_inv(theta, phi): 
    """
    Takes an input of polar angles and returns the 3x3 rotation matrix to go 
    from spherical basis vectors to cartesian basis vectors.
    
    Arguments:
        theta = polar angle from positive z-axis,
        phi = azimuthal angle from positive x-axis increasing anticlockwise (towards pos. y-axis)
        
    Returns:
        Rot_mat = 3x3 array which will undo the rotation of theta and phi
    """
    return np.array([[np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi),], 
                     [np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)], 
                     [np.cos(theta), -np.sin(theta), 0]])

def xyz_proj(theta, phi): 
    """
    Takes the angles theta and phi and returns the x, y and z projections of a 
    unit vector pointing in the theta and phi direction in the cartesian basis 
    vectors x, y and z.
    
    Multiply the result of this function onto a matrix.
    
    In essence, it is an alternative for the Rot_mat(theta, phi) useful for 
    arrays of data.
      
    Parameters
    ----------
    theta: array_like
        polar angle from positive z-axis.
        
    phi: array_like
        azimuthal angle from positive x-axis increasing anticlockwise (towards pos. y-axis).
        
    Returns
    -------
    x_proj: ndarray
        the x projection of the unit vector pointing in theta and phi direction.
    y_proj: ndarray
        the y projection of the unit vector pointing in theta and phi direction.
    z_proj: ndarray
        the z projection of the unit vector pointing in theta and phi direction.
    """
    if type(theta) == type(np.array([])): # Checks if input is an array of angles
        x_proj = np.array([np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)]).T
        y_proj = np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)]).T
        z_proj = np.array([np.cos(theta), -np.sin(theta), np.zeros(theta.shape)]).T
    else:
        x_proj = np.array([np.sin(theta)*np.cos(phi), np.cos(theta)*np.cos(phi), -np.sin(phi)]).T
        y_proj = np.array([np.sin(theta)*np.sin(phi), np.cos(theta)*np.sin(phi), np.cos(phi)]).T
        z_proj = np.array([np.cos(theta), -np.sin(theta), 0]).T
    return x_proj, y_proj, z_proj

def Rot_x(angle):
    "Rotation matrix about the global x-axis"
    return np.array([[1, 0, 0], 
                     [0, np.cos(angle), -np.sin(angle)], 
                     [0, np.sin(angle), np.cos(angle)]])

def Rot_y(angle):
    "Rotation matrix about the global y-axis"
    return np.array([[np.cos(angle), 0, np.sin(angle)], 
                     [0, 1, 0], 
                     [-np.sin(angle), 0, np.cos(angle)]])

def Rot_z(angle):
    "Rotation matrix about the global z-axis"
    return np.array([[np.cos(angle), -np.sin(angle), 0], 
                     [np.sin(angle), np.cos(angle), 0], 
                     [0, 0, 1]])

def cart_to_spher(x, y, z):
    "Convert coordinates from cartesian to spherical"
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    try: 
        if phi < 0.:
            phi += 2.*np.pi
    except:
        mask = phi < 0.
        phi[mask] += 2.*np.pi
    return r, theta, phi

def spher_to_cart(r, theta, phi):
    "Convert coordinates from spherical to cartesian"
    x = r*np.sin(theta)*np.cos(phi)
    y = r*np.sin(theta)*np.sin(phi)
    z = r*np.cos(theta)
    return x, y, z

# =============================================================================
# General Relativity Relationships
# =============================================================================
# =============================================================================
# The Schwarzschild metric
# =============================================================================
def metric(r, theta, phi, r_s=r_s_val):
    "Define the schwarschild metric in spherical coordinates with (-+++)"
    f = 1. - r_s / r #Schwarzschild function
    g_diag = np.array([-f, 1. / f, np.power(r, 2), np.power(r * np.sin(theta), 2)]) # Diagonal of the metric
    g = np.diag(g_diag) # Convert to a 4x4 matrix (tensor)
    return g

def inv_metric(r, theta, phi, r_s=r_s_val):
    "Define the inverse schwarschild metric in spherical coordinates with (-+++)"
    f = 1. - r_s / r #Schwarzschild function
    g_diag = np.array([-1. / f, f, 1. / np.power(r, 2), 1. / (np.power(r * np.sin(theta), 2))]) # Diagonal of the metric
    g = np.diag(g_diag) # Convert to a 4x4 matrix (tensor)
    return g

def Global_velocity_vector(r, theta, phi, r_s=r_s_val):
    "The Global unit velocity vector of the plasma, in this case a static plasma"
    g_tt = metric(r, theta, phi, r_s)[0,0] # metric
    return np.array([1. / np.sqrt(-g_tt), 0., 0., 0.]) # U^a

# =============================================================================
# The Hartle-Thorne metric
# =============================================================================

# def metric(r, theta, phi):
#     "Define the schwarschild metric in spherical coordinates with (-+++)"
#     f = 1. - r_s_val / r #Schwarzschild function
    
#     omega = Rot_ns
#     I_hat = 2. / 5. * M_ns * R_ns ** 2 # Assuming uniform solid sphere
#     omega_z = 2. * I_hat * G / np.power(r, 3) * omega # eqn(2) frame drag frequency (times by G to fix units)
#     frag_drag = omega_z * np.power(r * np.sin(theta), 2)
    
#     g = np.array([[-1. * f + omega_z * frag_drag, 0, 0, -1 * frag_drag],
#                   [0, 1. / f, 0, 0],
#                   [0, 0, np.power(r, 2), 0],
#                   [-1. * frag_drag, 0, 0, np.power(r * np.sin(theta), 2)]])
#     return g

# def inv_metric(r, theta, phi):
#     "Define the inverse schwarschild metric in spherical coordinates with (-+++)"
#     f = 1. - r_s_val / r #Schwarzschild function
    
#     omega = Rot_ns
#     I_hat = 2. / 5. * M_ns * R_ns ** 2 # Assuming uniform solid sphere
#     omega_z = 2. * I_hat * G / np.power(r, 3) * omega # eqn(2) frame drag frequency (times by G to fix units)

#     g = np.array([[-1. / f, 0, 0, -1 * omega_z / f],
#                   [0, f, 0, 0],
#                   [0, 0, 1. / np.power(r, 2), 0],
#                   [-1. * omega_z /f, 0, 0, 1. / np.power(r * np.sin(theta), 2) - omega_z **2 / f]])
#     return g


# =============================================================================
# Supplementry functions, not needed for simulation but useful code for other
# functions
# =============================================================================

def make_save_dir(simulation_path, simulation_name):
    """
    A function that creates a new directory to save the results in and then
    returns a string of that path to be used by the main file.
    
    Parameters
    ----------
    simulation_path: string
        States the path to the folder which will house the simulation_name folder.
    simulation_name: string
        The name of the folder that the simulation data is going to be saved in.
        
    Returns
    -------
    path: string
        The path to the folder created to save the simulation data.
    
    """
    if simulation_path[-1] != '/': # Checks if the last part of the string is a slash to denote the path
        simulation_path += '/'
    
    try: # Tries the simple save path
        os.makedirs('./data' + simulation_path + simulation_name) # Make the directory
        path = simulation_path + simulation_name +'/' # Gives the path to the created folder
        # path = '/{}_magnetic_field/{}/{}/{}/'.format(mag_field_choice, dispersion_method, metric_choice, simulation_name) # Update path variable
        
    except OSError as error: # If the above already exists then create a slightly different one
        print(error) # Lets user know that the simple file creation failed
        
        folders = [x[0] for x in os.walk('./data' + simulation_path)] # Gets the name of all simulations
        mask = char.find(folders, simulation_name + '_(') != -1 # Checks whether any share the same simulation_name + '_(*)'
        folders = np.array(folders)[mask] # Creates a mask to filter out folders not sharing simulation_name
        mask = char.find(folders, 't_') == -1 # Checks whether any share the same simulation_name + '_(*)'
        folders = np.array(folders)[mask] # Creates a mask to filter out folders not sharing simulation_name
        
        if folders.size == 0: # If no folders with slightly different folder name then create the first one
            path = simulation_path + '{}_({})/'.format(simulation_name, 1) # Update path variable
            os.makedirs('./data' + path) 
            
        else:
            left_num_loc = char.find(folders, '(')
            right_num_loc = char.find(folders, ')')
            nums = np.zeros(folders.size, dtype=int)
            for i in range(folders.size):
                nums[i] = int(folders[i][left_num_loc[i] + 1:right_num_loc[i]])
            i = nums.max() + 1 # Assumes the folders are in numerical order
            path = simulation_path + '{}_({})/'.format(simulation_name, i) # Update path variable
            # path = '/{}_magnetic_field/{}/{}/{}_({})/'.format(mag_field_choice, dispersion_method, metric_choice, simulation_name, i) # Update path variable
            os.makedirs('./data' + path) 
        
    print('Directory created: ' + path)
    return path
    




























