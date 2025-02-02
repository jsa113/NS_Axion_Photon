# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:28:30 2022
Updated on Sun May 21 15:51:34 2024

@author: Jesse Satherley
@email: jsa113@uclive.ac.nz

This is code written to ray trace photons from a distant observer onto the 
resonant conversion surface of axions to photons around a neutron star with 
varying magnetic fields and plasmas.

The result of running this code is the points of resonant conversion, which
can then be used to find the expected power received from axion to photon 
conversions.

Using the complex relativitic GR relationship from DOI: 10.1103/PhysRevE.64.027401


Important notes:
    Units: Natural units (c=h_bar=epsilon_0=1)
    Metric: Schawarzschild

Parameters:
    These are all defined in the parameters.py and the parameter set text files.
    All of these quanities are given in natural units in this file.
    
    Note: in the parameter set files, they are given in a mixture of SI and natural units.
    
    - m_a: Mass of axion in eV/c^2
    - g_ayy: Axion-photon coupling constant in GeV^-1
    - Period: Period of rotation for the NS in eV^-1
    - R_ns: Neutron star radius in eV^-1
    - M_ns: Neutron star mass in eV
    - B_0: Magnetic field strength at surface in eV^2
    - Theta_m: Misalignment angle of magnetic feild and spin axis in radians
    - DM_v_0: Dark matter dispersion velocity in (unitless)
    - DM_pho_inf: Density of dark matter infinity far from the neutron star (eV^4)

Saves files:
    - conversion_point = The position 4-vector and the momentum 4-vector of all 
                         the intercepts that a photon makes with the conversion surface 
                         during the fine_search function saved as an array saves 
                         as a .npy file.
                         
    - conversion_index = The index of the associated photon conversion point 
                         from conversion_point as an array saved as a .npy file.
                         
    If save_photon_paths is True, then also saves the path of each photon as it
    is ray traced through the plasma. Can be useful for plotting the paths of 
    each photon or investigating the effects the plasma has on individual photons.
"""

# ntasks=4
# cpus-per-task=32
# srun --unbuffered python3.11 -O ...
from parameters import * # The file containing all the simulation parameters
from functions import * # The file containing all the extra functions
import autograd.numpy as np # Used for the derivatives
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import solve_ivp # Used to solve dispersion coupled ODEs
from multiprocessing import Pool, cpu_count # Use to run multithreaded processes
from mpi4py import MPI
import os
from datetime import datetime
from numpy import char

# Import the appropriate magnetic field functions
if mag_field_choice == 'GR':
    if Theta_m == 0:
        from GR_magnetic_field_aligned import *
        # print('Main: GR Aligned Magnetic Field Imported')
        
    else:
        from GR_magnetic_field import *
        # print('Main: GR Magnetic Field Imported')
        
elif mag_field_choice == 'GJ':
    from GJ_magnetic_field import *
    # print('Main: GJ Magnetic Field Imported')

# Import the dispersion relationships
from Dispersion_methods import *


# =============================================================================
# Termination event that finds when the conversion surface is crossed by the
# integration or the Schwarzschild radius or the Neutron star surface.
# =============================================================================
def coarse_terminate_event(t, y):
    """
    Defines the coarse search event that terminates the integration of solve_ivp.
    Checks when the radial distance of the photon is within the conversion 
    surface radius multiplied by a relative tolerance term to ensure edge coarse 
    search photons are included.
    
    Arguments:
        solve_ivp return variables
        
    returns:
        difference between current integration step and the conversion surface
    """
    
    if y[1] > max_r_conversion * 1.1: # This if statement saves unnesscary checks when far from the star. Most helpful in the GR case when it must root solve the conversion surface radius
        return 1.
    else:
        return y[1] - Conversion_Surface_scalar(y[2], y[3], y[0]) * coarse_search_rel_tol
    
def fine_search_event(t, y):
    """
    Defines the fine search event that terminates the integration of solve_ivp.
    
    The if statement saves unnesscary checks when far from the star.
    Most helpful in the GR case when it must root solve the conversion surface 
    radius.
    
    Arguments:
        solve_ivp return variables
        - t the time of the integration
        - y the concatenated four-vectors of position and momentum.
        
    returns:
        difference in the radius between current integration step and the 
        conversion surface found at the same angles.
    """
    
    if y[1] > max_r_conversion * 1.1: 
        return 1.
    else:
        # Compares the current photon's radius to the equivalent angle's conversion surface radius
        return y[1] - Conversion_Surface_scalar(y[2], y[3], y[0]) 

def Schwarzschild_terminate_event(t, y):
    """
    Defines the solve_ivp terimation event in the case that the integration 
    goes below the Schwarzschild radius.
    
    Arguments:
        solve_ivp return variables
        
    returns:
        difference between current integration step and the Schwarzschild radius
    """
    return y[1] - r_s_val


def R_ns_terminate_event(t, y):
    """
    Defines the solve_ivp terimation event in the case that the integration 
    goes below the Neutron star surface.
    
    Arguments:
        solve_ivp return variables
        
    returns:
        difference between current integration step and the Neutron star's radius
    """
    return y[1] - R_ns

coarse_terminate_event.terminal = True # Set true to terminate integration when above condition is satisfied
fine_search_event.terminal = False # Set true to terminate integration when above condition is satisfied
Schwarzschild_terminate_event.terminal = True # Set true to terminate integration when above condition is satisfied
R_ns_terminate_event.terminal = True # Set true to terminate integration when above condition is satisfied
    
# =============================================================================
# Find the approxiate maximum conversion radius by computing a fixed phi
# =============================================================================
conversion_surface_resolution = 180 # Resolution of the approximate maximum radius search
theta_array = np.linspace(0. + 1e-7, pi + 1e-7, conversion_surface_resolution) #Initial theta angle
max_r_conversion = np.max(Conversion_Surface(theta_array, np.zeros(theta_array.size), np.zeros(theta_array.size))) # Returns the max conversion radius
max_x_R_ns = np.ceil(max_r_conversion / R_ns) # Add 2 for extra spacing around detector # This is a bit ad-hoc

# =============================================================================
# Define the detector's mesh
# =============================================================================
max_x = max_x_R_ns * R_ns
max_y = max_x # A square mesh

coarse_step = 2. * max_x / coarse_resolution # Distance between adjacent coarse pixels
fine_step = 2. * max_x / total_resolution # Distance between adjacent fine pixels

# Defines the centers of the coarse pixels
coarse_x_array = np.linspace(-max_x, max_x, coarse_resolution, endpoint=False) + coarse_step / 2. 
coarse_y_array = np.linspace(-max_x, max_x, coarse_resolution, endpoint=False) + coarse_step / 2.

# Mesh containing the initial positions of the coarse-search photons
coarse_x_mesh, coarse_y_mesh = np.meshgrid(coarse_x_array, coarse_y_array)

# Defines the centers of the fine pixels
fine_x_array = np.linspace(-max_x, max_x, total_resolution, endpoint=False) + fine_step / 2.
fine_y_array = np.linspace(-max_x, max_x, total_resolution, endpoint=False) + fine_step / 2.

# Mesh containing the initial positions of the fine-search photons
fine_x_mesh, fine_y_mesh = np.meshgrid(fine_x_array, fine_y_array)

# for x in coarse_x_array:
#     print(np.isclose(fine_x_array, x, atol=coarse_step/2))

def Back_Propagate_Photon(x_i, y_i, t_0, Obs_theta, Obs_phi, search_event=fine_search_event, rtol=1e-6, atol=1e-6):
    """
    Back propagate a photon from the detector to the neutron star 
    (or until integration finishes) with a initial position on the detector of 
    x_i and y_i.
    
    This function is used to define all the physical parameters for the initial 
    coniditions of the ODE solver.
    
    Arguments:
        x_i = initial x position of the photon on the detector
        y_i = initial y position of the photon on the detector
        terminate_event = Which event function to use when terminating for intersection
    
    Returns:
        soln = ODE solver solution to the 
    """
    
    # Begin with a detector whose center is along the z-axis
    z_i = 0
    xyz_i = np.array([x_i, y_i, z_i])
    
    # Convert the detector mesh to positional data for the photons to "begin" at
    center_xyz = spher_to_cart(Obs_r0, Obs_theta, Obs_phi) # xyz-coordinate of the centre of the detector
    # Rotate detector to align with the center line of the observer
    rotated_xyz_i = center_xyz + np.dot(Rot_z(Obs_phi), np.dot(Rot_y(Obs_theta), xyz_i))
    
    # Convert carteisn coordinates to spherical polar
    r_0, theta_0, phi_0 = cart_to_spher(rotated_xyz_i[0], rotated_xyz_i[1], rotated_xyz_i[2])
    
    g_ab = metric(r_0, theta_0, phi_0)
    
    # Set this as the initial position of the photon for the ODE solver
    x_0 = np.array([t_0, r_0, theta_0, phi_0])
    
    # Define the initial velocity unit vector of the photon
    # =============================================================================
    # Return a vector in cartesian that represents the radial trajectory from the 
    # detector towards the centre of the star
    # =============================================================================
    v_hat = np.matmul(Rot_mat_inv(Obs_theta, Obs_phi), [1, 0, 0]) # Rotated from polar to cartesian # Unit vector length in GR
    rotated_v_hat = np.matmul(Rot_mat(theta_0, phi_0), v_hat) # Rotate from cartesian to polar so that all rays are parrallal
    
    # =========================
    # Photon initial conditions
    # =========================
    # Energy convservation for the photon's energy via the axion infalling plus asymptotic velocity
    # v_min_sq = 2* G* M_ns / r_0 + DM_v_0 ** 2 # Infalling velocity plus asymptotic velocity via energy conservation
    # w_0 = np.sqrt((1 + v_min_sq) * m_a**2) # E^2 = m^2 + p^2 or essentailly the red shift 
    
    # U = Global_velocity_vector(r_0, theta_0, phi_0)
    # k_up_t0 = w_0 * U[0] # From hbar * w = p^i * U_i => p^t = w * U^t
    k_up_t0 = np.sqrt(m_a**2) / ( -1. * g_ab[0, 0]) # Eqn (8) from A. Rogers 2015

    # The initial spatial momentum with rotations to point parallel
    k_0 = np.sqrt(k_up_t0 **2 * -1. * g_ab[0, 0] / g_ab[1, 1]) # Set refractive index to n = 1
    # Set the momentum of the unit vector to that of the photon
    kr_0 = k_0 * rotated_v_hat[0]
    ktheta_0 = k_0 * rotated_v_hat[1] / r_0 # https://en.wikipedia.org/wiki/Schwarzschild_geodesics Subheading:Local and delayed velocities; eqn. (2)
    kphi_0 = k_0 * rotated_v_hat[2] / (r_0 * np.sin(theta_0))
    k_0 = np.array([k_up_t0, kr_0, ktheta_0, kphi_0])
    k_0 = np.matmul(g_ab, k_0) # Shift the indicies downstairs for the Hamiltonian equations

    # Group the initial conditions
    y_0 = np.append(x_0, k_0)
    
    # Solve the initial value problem.
    soln = solve_ivp(F, [0, -Int_time], y_0, method='Radau', events=[search_event, R_ns_terminate_event], rtol=rtol, atol=atol)
    return soln

def coarse_search(index):
    """
    Function to complete the coarse search for photon intersections of the 
    conversion surface
    
    Arguments:
        Takes an array of the coarse pixel x and y indicies and the time to 
        evalulate the simulation at.
    
    Returns:
        Truth array of the entire detector with the intercepted photon coarse 
        pixel as True plus the surrounding coarse pixels
    """
    
    # extract the x and y index from the index array passed through
    i = int(index[0])
    j = int(index[1])
    # Extract the simulation parameters from the index array
    t_0 = index[2]
    Obs_theta = index[3]
    Obs_phi = index[4]
    
    # Define the initial x and y position of the photon on the detector
    coarse_x = coarse_x_mesh[i, j]
    coarse_y = coarse_y_mesh[i, j]
    coarse_mesh_intercept = np.full((coarse_resolution, coarse_resolution), False)
    # Do the ray tracing and return the solve_ivp solution.
    try: # Skips a ray if a numerical error occurs. A bit ad-hoc, but easiest method
        sol = Back_Propagate_Photon(coarse_x, coarse_y, t_0, Obs_theta, Obs_phi, coarse_terminate_event, rtol=1e-6, atol=1e-6)
        
        
        
        # If an coarse intercept occured, the set the pixels to True
        if sol.t_events[0].size > 0:
            # coarse_mesh_intercept[i, j] = True # Set just the pixel that the photon comes from
            coarse_mesh_intercept[i-1:i+2, j-1:j+2] = True # Set the search for the center plus all adjacent pixels, including diagonal
        return coarse_mesh_intercept
    except:
        return coarse_mesh_intercept
    
def fine_search(index):
    """
    Function to complete the fine search for photon intersections of the 
    conversion surface
    
    Arguments:
        Takes an array of the coarse pixel x and y indicies and the time to 
        evalulate the simulation at.
    
    Returns:
        If the photon intercepts the conversion surface then returns the photon
        - [np.array([i, j]), sol.y, sol.y_events[0]] 
        
        where i, j are the ray's origin pixel's indicies, sol.y is the ray path
        and sol.y_events[0] is the points at which the ray intercepts the 
        conversion surface
        
        Else if the photon does not intercept, then returns 
        - None
    """

    # extract the x and y index from the index array passed through
    i = int(index[0])
    j = int(index[1])
    t_0 = index[2]
    Obs_theta = index[3]
    Obs_phi = index[4]
    
    # Initial position data
    fine_x = fine_x_mesh[i, j]
    fine_y = fine_y_mesh[i, j]
    
    try: # Skips a ray if a numerical error occurs. A bit ad-hoc, but easiest method
        sol = Back_Propagate_Photon(fine_x, fine_y, t_0, Obs_theta, Obs_phi, fine_search_event, rtol=1e-8, atol=1e-8)
        
        # Set if the photon crosses the conversion surface
        if sol.t_events[0].size > 0:
            # return np.append(np.array([i, j]), sol.y[:,-1])
            # Returns the index of the detector, the ray path, and then the conversion surface intercepts.
            return [np.array([i, j]), sol.y, sol.y_events[0]] 
        else: # Else return an none element which can be removed.
            return None
    except:
        return None
    

if __name__ == "__main__": # Required to run Pool without Pool repeating itself
    comm = MPI.COMM_WORLD # Allows the spliting of processing across multiple CPUs
    size = comm.Get_size() # Number of CPUs
    rank = comm.Get_rank() # CPU number\
    # # Use these if not using MPI
    # size = 1
    # rank = 0
    
    # =============================================================================
    # Code to create the save directory of the resulting simulation data
    # =============================================================================
    if rank == 0:
        
        print('MPI Size = {}'.format(size))
        
        print("Beginning Axion-Photon Conversion Search in O2E Scheme\nPhase fraction(s) to evaluate: {}\nObs_Theta(s) to evaluate: {}".format(phase_array, Obs_theta_array))
        
        # Print the simulation parameters to the user
        sim_str = "dispersion_method = {}\nmetric_choice = {}\nmag_field_choice = {}\nmax_x = {} *R_ns\n".format(dispersion_method, metric_choice, mag_field_choice, max_x_R_ns)
        parameter_str_1 = "coarse_resolution = {}\nfine_resolution = {}\nObs_theta = {}\nObs_phi = {}\nObs_r0 = {}R_ns\n".format(coarse_resolution, fine_resolution, Obs_theta_array * 180 / pi, Obs_phi_array * 180 / pi, Obs_r0_n)
        parameter_str_2 = "m_a = {}\ng_ayy = {}\nPeriod = {}\nR_ns = {}\nM_ns = {}\nB_0 = {}\nTheta_m = {}\nDM_v_0 = {}\nDM_pho_inf = {}".format(*parameters)
        parameter_str = sim_str + parameter_str_1 + parameter_str_2
        # print('='*21 + "\nSimulation parameters\n" + '='*21 + "\n" + parameter_str + '\n' + '='*21)
        
        start = datetime.now() # Start time of the code
        date_str = start.strftime("%Y_%m_%d")
        
        # Create the path that will be the save location for the data
        path = '/{}_magnetic_field/{}/{}/'.format(mag_field_choice, dispersion_method, metric_choice) # Base path for results to be saved 
        # updates the path for the new one
        path = make_save_dir(path, date_str) 
        # Save the parameters of the simulation to a txt file
        np.savetxt('./data' + path + 'parameter_set.txt', [parameter_str], delimiter=" ", fmt="%s")
    else:
        path = None
        
    path = comm.bcast(path, root=0) # Updates path for all CPUs
   
    # =============================================================================
    # Carry out the simulation for a set of phase values
    # =============================================================================
    for detector_data in  Detector_data_array: # Iterate through each phase in the phase array
        phase = np.round(detector_data[0], 10) # Round so that the number is a nicer value when saving file
        t_0 = phase * Period # Convert phase to time
        detector_data[0] = t_0 # update this
        Obs_theta = detector_data[1]
        Obs_phi = detector_data[2]
        

        if rank == 0:
            print('Starting\nt_0 = {}*Period\nObs_theta = {}\n'.format(phase, Obs_theta * 180 / pi))
            print("Total number of coarse photon pixels to search: {}".format(num_coarse_pixels))
        
        # Gives the index of each coarse pixel to be passed through pool for coarse search function
        coarse_index_array =  np.flip(np.indices((coarse_resolution, coarse_resolution)).T.reshape(coarse_resolution**2, 2), axis=1)
        # Append the time of the simulation to the indicies to be passed through pool
        coarse_index_array = np.hstack([coarse_index_array, np.full((coarse_index_array.shape[0], 3),  detector_data)])
        
        coarse_index_array_proc = np.array_split(coarse_index_array, size, axis=0)[rank] # Data to search over for a single CPU
        
        # # Use this code to run without multiprocessing
        # coarse_result = list()
        # i = 0
        # for data in coarse_index_array:
        #     print(i)
        #     coarse_result.append(coarse_search(data))
        #     i += 1
        
        # Carry out the coarse search mesh refinement algrithm
        print("Starting: CPU {}".format(rank))
        
        with Pool() as p: # Change the number inside Pool() for number of threads
            coarse_result_proc = p.map(coarse_search, coarse_index_array_proc) # Same as below but without progess but apparently is MUCH faster
            # coarse_result = list(tqdm(p.imap(coarse_search, coarse_index_array_proc), total=num_coarse_pixels)) # Use this to see progress
           
        coarse_mesh_intercept_proc = np.any(np.stack([x for x in coarse_result_proc if x is not None]), axis=0) # Convert results of the coarse search into a single boolean array to act as a mask  
        print("Finished: CPU {}".format(rank))
        
        coarse_result_comm = comm.gather(coarse_mesh_intercept_proc, root=0)
        
        
        if rank == 0:
            
            # Indicies of the coarse-pixels that 'intersect' the conversion surface
            coarse_mesh_intercept = np.any(coarse_result_comm, axis=0)
            coarse_y_indicies, coarse_x_indicies = np.array(np.where(coarse_mesh_intercept == True)) * fine_resolution 
            print("Total number of coarse photon pixels intercepted: {}".format(coarse_y_indicies.size))
        
            # =============================================================================
            # Now construct the fine indices to search through from the coarse pixels that
            # reach the conversion surface
            # =============================================================================
            num_photons = coarse_x_indicies.size * fine_resolution ** 2 # Total number of photons to test in fine search
            
            # The fine pixel indicies that are to searched through
            fine_x_indicies = (np.tile(np.tile(np.arange(0, fine_resolution, 1), (fine_resolution,1)), (coarse_x_indicies.size, 1)) 
                               + np.vstack((np.repeat(coarse_x_indicies, fine_resolution),)*fine_resolution).T)
            fine_y_indicies = (np.tile(np.tile(np.arange(0, fine_resolution, 1), (fine_resolution,1)).T, (coarse_y_indicies.size, 1)) 
                               + np.vstack((np.repeat(coarse_y_indicies, fine_resolution),)*fine_resolution).T)
            
            # Take the indicies from above and append the time of the simulation
            # fine_search_data = np.hstack([np.reshape(fine_y_indicies, (num_photons, 1)), np.reshape(fine_x_indicies, (num_photons, 1))])
            fine_search_data = np.hstack([np.reshape(fine_y_indicies, (num_photons, 1)), np.reshape(fine_x_indicies, (num_photons, 1)), np.full((num_photons, 3), detector_data)])
            np.random.shuffle(fine_search_data) # Shuffle the data so that it is split randomly amongst tasks
            print("Total number of fine photon pixels to search: {}".format(num_photons))
        else:
            fine_search_data = None
            
        
        fine_search_data = comm.bcast(fine_search_data, root=0)
        fine_search_data_proc = np.array_split(fine_search_data, size, axis=0)[rank] # Data to search over for a single CPU
        
        print("Starting: CPU {}".format(rank))
        with Pool() as p: # Change the number inside Pool() for number of threads 
            fine_result = p.map(fine_search, fine_search_data_proc) # Same as below but without progess but apparently is MUCH faster
            # fine_result = list(tqdm(p.imap(fine_search, fine_search_data), total=num_photons)) # Use this to see progress
        
        # Removes nones from fine search results (fine pixel photons that missed the conversion surface)
        converted_photons = [x for x in fine_result if x is not None]
        print("Total number of fine photon pixels intercepted: {}".format(len(converted_photons)))
        
        # =============================================================================
        # Save the data photon path data from each CPU for later use
        # =============================================================================
        
        data_path = './data' + path + 'Obs_theta_{}_t_{}/'.format(np.round(Obs_theta * 180 / np.pi, 2), phase)
        try:
            os.makedirs(data_path) 
        except OSError as error: 
            None
        
        if save_photon_paths:
            data_str_list = ['photon_data_t', 'photon_data_r', 'photon_data_theta', 'photon_data_phi', 'photon_data_w', 'photon_data_kr', 'photon_data_ktheta', 'photon_data_kphi']
            for i, data_str in enumerate(data_str_list):
                data_proc = list()
                for photon in converted_photons:
                    data_proc.append(photon[1][i])
                    
                np.save(data_path + data_str + '_rank_{}'.format(rank), np.array(data_proc, dtype=object), allow_pickle=True)
        
        conversion_point_proc = np.array([], dtype=np.int64).reshape(0,8)
        conversion_index_proc = np.array([], dtype=np.int64).reshape(0,2)
        for photon in converted_photons:
            conversion_point_proc = np.vstack([conversion_point_proc, photon[2]])
            conversion_index_proc = np.vstack([conversion_index_proc, np.tile(photon[0], [photon[2].shape[0], 1])])
            
        np.save(data_path + 'conversion_point' + '_rank_{}'.format(rank), conversion_point_proc)
        np.save(data_path + 'conversion_index' + '_rank_{}'.format(rank), conversion_index_proc)
        
        print("Finished: CPU {}".format(rank))
        comm.Barrier()
        
        # =============================================================================
        # Collate the data from each CPU
        # =============================================================================
        
        if rank == 0:
            if save_photon_paths:
                for data_str in data_str_list:
                    data = np.array([])
                    for i in range(size):
                        data = np.append(data, np.load(data_path + data_str + '_rank_{}.npy'.format(i), allow_pickle=True))
                        os.remove(data_path + data_str + '_rank_{}.npy'.format(i))
                        
                    np.save(data_path + data_str, np.array(data, dtype=object), allow_pickle=True)
                
            conversion_point = np.array([])
            conversion_index = np.array([])
            for i in range(size):
                conversion_point = np.append(conversion_point, np.load(data_path + 'conversion_point' + '_rank_{}.npy'.format(i)))
                conversion_index = np.append(conversion_index, np.load(data_path + 'conversion_index' + '_rank_{}.npy'.format(i)))
                
                os.remove(data_path + 'conversion_point' + '_rank_{}.npy'.format(i))
                os.remove(data_path + 'conversion_index' + '_rank_{}.npy'.format(i))
                
            np.save(data_path + 'conversion_point', conversion_point)
            np.save(data_path + 'conversion_index', conversion_index)
            np.save(data_path + 'conversion_coarse_index', np.append(coarse_x_indicies, coarse_y_indicies))
            
            finish = datetime.now()
            print('\nFinished Axion-Photon conversion simulation\nRun time = {}'.format(finish - start))
        
        comm.Barrier()
    
    
    
    


















