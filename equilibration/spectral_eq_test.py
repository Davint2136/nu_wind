import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy
from os import system, name
from scipy.optimize import bisect
import pandas as pd

#TODO: Change rates to reflect different energy bins needed for computing spectral rates
class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1, time = 0.0, dU = None):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV cm-3]
        self.n_m1 = n_m1                                    # Gray neutrino number densities. Order: nue, anue, nux, anux [cm-3]
        self.J_m1 = J_m1                                    # Gray neutrino energy densities. Order: nue, anue, nux, anux [MeV cm-3]
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.edot_sources = None                                 # Neutrino energy density source terms for integration [MeV cm-3 s-1]
        self.ndot_sources = None                                 # Neutrino number density source terms for integration [cm-3 s-1]
        self.edot = None
        self.yedot = None
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None                                  # Dirac effective proton mass [MeV]
        self.mn_eff = None                                  # Dirac effective neutron mass [MeV]
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = dU
        self.time = time                                    # Time at which the state occurs during the simulation [s]


class Solver:
    c = 29979245800.0

    def __init__(self, table : Table, state : State, opacity_flags : dict, opacity_pars : dict):
        self.timestep = None
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars
        """self.k1_e = []
        self.k2_e = []
        self.k1_ye = []
        self.k2_ye = []"""


    def temperature_from_e(self, state : State):
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
                nb [np.NDArray]: A 1D array consisting of one baryon number density in cm^-3.
                ye [np.NDArray]: A 1D array consisting of one charge fraction.
                e_val [float]: The fluid's energy density in MeV/cm^3.

            Outputs:
                t [float]: The fluid temperature in MeV.
        """

        log_e = np.log((self.table.thermo["Q7"] + 1) * self.table.mn * state.nb * 1e-39)
        # Find difference between calculated e and actual e
        def f(t):
            e_interp = self.table.eval_given_rtx(log_e, np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([t]), method="linear")[0]
            return (state.fluid_e * 1e-39) - np.exp(e_interp)

        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")
        
        return t
    
    def get_potentials(self, state : State):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """


        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn 
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q  #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self, state : State):
        """
            Calulates quantities needed for BNS_NURATES corrections that are relevant at high density. 

            Outputs:
                mp_eff [float]: Proton Dirac effective mass [MeV]
                pn_eff [float]: Neutron Dirac effective mass [MeV]
                dm_eff [float]: The effective mass difference between neutrons and protons
        """

        #Use eos.qK to find dirac effective masses to calculate dm_eff, don't attempt dU for now

        interp = self.table.interpolate_3D(np.array([state.nb]) * 1e-39, np.array([state.ye]), np.array([state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff
        return mp_eff, mn_eff, dm_eff



    #TODO: do this using spectral rates. Make bins for different neutrino energies, find the rates for each, then add them up.
    def calculate_spectral_rates(self, state : State):
        """
            Calculates neutrino interaction rates based on the fluid's conditions in the current state.

            Outputs:
                gray_rates [dict]: A dictionary consisting of energy and number emissivities, energy and number absorptivities, and scattering absorptivities.
        """

        #Initialize eos_pars and populate
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = state.nb * 1e-21 # Convert baryon number density to nm^-3
        eos_pars.temp = state.t
        eos_pars.ye = state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = state.mu_n
        eos_pars.mu_e = state.mu_e
        eos_pars.mu_p = state.mu_p
        eos_pars.dm_eff = state.dm_eff

        #Create a quadrature, populate it with data for a Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 6
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars
        
        #Load M1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = state.chi_m1
        m1_pars.n = [x * 1e-21 for x in state.n_m1]
        m1_pars.J = [x * 1e-21 for x in state.J_m1]

        distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)

        #Initialize energy distribution - from 0 to 100 MeV in 5 MeV steps. Center of bin used for computation.
        nu_E_bins = np.linspace(2.5, 100, 5).tolist()
        
        #Calculate n_nu(E) by finding distribution
        ids = {0 : "nue", 1 : "anue", 2 : "nux", 3: "anux"}
        distribution = {"nue" : [], "anue" : [], "nux" : [], "anux" : []}

        for species_id in ids.keys():
            for E in nu_E_bins:
                f_point = bns.TotalNuF(E, distr_pars, species_id)
                distribution[ids[species_id]].append(f_point)

            distribution[ids[species_id]] = np.array(distribution[ids[species_id]])

        #Populate global structure using grey_pars
        grey_pars = bns.GreyOpacityParams()
        grey_pars.eos_pars = eos_pars
        grey_pars.opacity_flags = opacity_flags
        grey_pars.opacity_pars = opacity_pars
        grey_pars.distr_pars = distr_pars
        grey_pars.m1_pars = m1_pars

        return 

    #TODO: Make this work with spectral rates. source term calculations will change.
    def calculate_state(self, state : State):
        """
            Calculates state variables left blank at the time of state creation. Updates the inputted state in-place.
        """
        state.t = self.temperature_from_e(state)
        state.mu_p, state.mu_n, state.mu_e = self.get_potentials(state)
        state.mp_eff, state.mn_eff, state.dm_eff = self.calculate_corrector_quantities(state)
        state.rates = self.calculate_gray_rates(state)
        state.edot_sources = self.calculate_e_source_terms(state)
        state.ndot_sources = self.calculate_n_source_terms(state)
        return

    def generate_new_state(self, nb : float, ye : float, fluid_e : float, n_m1 : list, J_m1 : list, time : float, dU=None):
        """
            Given a number density, electron fraction, and energy density, creates a new State object.

            Inputs:
                nb [float]: The fluid's baryon number density [cm-3].
                ye [float]: The fluid's electron fraction.
                fluid_e [float]: The fluid's energy density [MeV cm-3]
                time [float]: The current time in the simulation for the new state.
            
            Outputs:
                [State]: A new state object containing the inputted values. n_m1, J_m1, and chi_m1 from the previous state persist into the new State object.
        """
        return State(nb, ye, fluid_e, n_m1, J_m1, [1/3, 1/3, 1/3, 1/3], time, dU)

    def integrate_step(self, state : State):
        """
            Integrates the simulation forward in time by one timestep. Finds the next fluid energy density and electron fraction using a variable step size RK2 integrator, where the timestep corresponds
            to a small fraction of the absorptivity corresponding to the shortest timescale.

            Outputs:
                next_e [float]: The fluid's energy density at the next timestep of the simulation.
                next_ye [float]: The fluid's electron fraction at the next timestep of the simulation.
                next_time [float]: The simulation time at the next timestep.
        """

        # Calculate timestep, largest absorptivity corresponds to shortest timescale
        self.calculate_state(state)

        timestep = 0.1 * ((1 / max(state.rates["kappa_a"] + state.rates["kappa_0_a"])) / self.c)
        
        # Find k1 for neutrino fields 
        k1_J = (timestep * np.array(state.edot_sources))
        k1_n = (timestep * np.array(state.ndot_sources))

        #Calculate k1 for the fluid energy density and electron fraction
        k1_e = -(np.sum(state.edot_sources)) * timestep
        k1_ye = ((state.ndot_sources[1] - state.ndot_sources[0]) / state.nb) * timestep

        state.edot = k1_e
        state.yedot = k1_ye

        #Calculate state at midpoint of timestep
        e_fluid_mid = state.fluid_e + (k1_e / 2)
        ye_mid = state.ye +  (k1_ye / 2)

        J_m1_mid = np.array(state.J_m1) + (k1_J / 2)
        n_m1_mid = np.array(state.n_m1) + (k1_n / 2)

        midpoint = self.generate_new_state(state.nb, ye_mid, e_fluid_mid, n_m1_mid, J_m1_mid, state.time + (timestep * 0.5))

        self.calculate_state(midpoint)

        #Find k2 for neutrino fields and matter fields
        k2_J = (timestep * np.array(midpoint.edot_sources))
        k2_n = (timestep * np.array(midpoint.ndot_sources))

        #Calculate k2 for the fluid energy density and electron fraction
        k2_e = -(np.sum(midpoint.edot_sources)) * timestep
        k2_ye = ((midpoint.ndot_sources[1] - midpoint.ndot_sources[0]) / midpoint.nb) * timestep

        #Update neutrino fields
        new_J_m1 = np.array(state.J_m1) + ((k1_J + k2_J) * 0.5)
        new_n_m1 = np.array(state.n_m1) + ((k1_n + k2_n) * 0.5)

        """print(f"\nTime: {midpoint.time}")
        print(f"k2_e / k1_e: {k2_e / k1_e}")
        print(f"k2_ye / k1_y1: {k2_ye / k1_ye}")
        self.k2_e.append(k2_e)
        self.k1_e.append(k1_e)
        self.k2_ye.append(k2_ye)
        self.k1_ye.append(k1_ye)"""
        #Update fluid energy density and electron fraction
        new_fluid_e = state.fluid_e + ((k1_e + k2_e) * 0.5)
        new_ye = state.ye + ((k1_ye + k2_ye) * 0.5)

        #Generate next state
        next_state = self.generate_new_state(state.nb, new_ye, new_fluid_e, new_n_m1, new_J_m1, state.time + timestep, state.dU)

        return next_state
    
    
    def integrate_to_eq(self, verbose=False):
        init_state = deepcopy(self.states[-1])
        self.calculate_state(init_state)
        t_stop = ((1 / min(init_state.rates["kappa_a"] + init_state.rates["kappa_0_a"])) / self.c) * 200
        init_e_tot = init_state.fluid_e + sum(init_state.J_m1)
        init_lnum = init_state.ye + ((init_state.n_m1[0] - init_state.n_m1[1]) / init_state.nb)

        eq_detected = False
        while not eq_detected:

            next_state = self.integrate_step(self.states[-1])
            current_e_tot = self.states[-1].fluid_e + sum(self.states[-1].J_m1)
            current_lnum = self.states[-1].ye + (((self.states[-1].n_m1[0] - self.states[-1].n_m1[1]) + (self.states[-1].n_m1[2] - self.states[-1].n_m1[3])) / self.states[-1].nb)
            current_nue_pot = self.states[-1].mu_p - self.states[-1].mu_n + self.states[-1].mu_e

            #rates = self.states[-1].rates
            #em_frac = max(abs(rates["eta"][i] - (self.c * rates["kappa_a"][i] * self.states[-1].J_m1[i])) / max(rates["eta"][i], self.c * rates["kappa_a"][i] * self.states[-1].J_m1[i]) for i in range(4))
            if verbose:
                print(f"\nTime: {self.states[-1].time}/{t_stop}"
                    f"\ne: {self.states[-1].fluid_e}"
                    f"\nYe: {self.states[-1].ye}"
                    f"\nT: {self.states[-1].t}"
                    f"\nnb: {self.states[-1].nb}"
                    f"\nmn_eff: {self.states[-1].mn_eff}"
                    f"\nmp_eff: {self.states[-1].mp_eff}"
                    f"\ndm_eff: {self.states[-1].dm_eff}"
                    f"\nedot_sources: {self.states[-1].edot_sources.tolist()}"
                    f"\nndot_sources: {self.states[-1].ndot_sources.tolist()}",
                    f"\nn_m1: {self.states[-1].n_m1}",
                    f"\nJ_m1: {self.states[-1].J_m1}",
                    f"\nedot: {self.states[-1].edot}"
                    f"\nyedot: {self.states[-1].yedot}",
                    f"\nEnergy conservation: {current_e_tot / init_e_tot}",
                    f"\nLepton fraction conservation: {current_lnum / init_lnum}")

            self.states.append(next_state)

            if self.states[-1].time > t_stop:
                eq_detected = True

        self.calculate_state(self.states[-1])
        self.save_to_csv()
        return


    def save_to_csv(self):
        nb_list = []
        e_list = []
        ye_list = []
        t_list = []
        edot_sources_list = []
        ndot_sources_list = []
        n_m1_list = []
        J_m1_list = []
        time_list = []
        
        for state in self.states:
            nb_list.append(state.nb)
            e_list.append(state.fluid_e)
            ye_list.append(state.ye)
            t_list.append(state.t)
            edot_sources_list.append(state.edot_sources)
            ndot_sources_list.append(state.ndot_sources)
            n_m1_list.append(state.n_m1)
            J_m1_list.append(state.J_m1)
            time_list.append(state.time)

        data = pd.DataFrame({"Time" : time_list, "nb" : nb_list, "e_fluid" : e_list, "ye" : ye_list, "t" : t_list, "edot_sources" : edot_sources_list, "ndot_sources" : ndot_sources_list, "n_m1" : n_m1_list, "J_m1" : J_m1_list})
        data.to_csv("single_zone.csv")
        return

        


#TESTING-------------------------------------------------------------
SCRIPTDIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(SCRIPTDIR, os.pardir))

md = Metadata(
    pairs = {
        0: ("e", "electron"),
        10: ("n", "neutro"),
        11: ("p", "proton"),
        4002: ("He4", "alpha particle"),
        3002: ("He3", "helium 3"),
        3001: ("H3", "tritium"),
        2001: ("H2", "deuteron")
    },
    quads = {
        999: ("N", "average nucleous")
    },
    micro={
        10041: ("mn_d", "neutron dirac effective mass divided by neutron mass"),
        11041: ("mp_d", "proton dirac effective mass divided by proton mass")
    }
)
eos = Table(md)
print("reading...")
eos.read(os.path.join(SCRIPTDIR, "DD2"), enforce_equal_spacing=True)

# %%
eos.compute_cs2(floor=1e-6)
eos.compute_abar()
eos.validate()
# Remove the highest temperature point
eos.restrict_idx(it1=-1)
eos.shrink_to_valid_nb()

## Input thermodynamic quantities
##  N.B.: chemical potentials include the rest mass contribution

nb = (9.81e13 / (eos.unit_dens * eos.mn)) * 1e39 # Baryon number density [cm-3]   
t = 16.63   # Temperature [MeV]
ye   =  0.06

e_nb = np.exp(eos.eval_given_rtx(np.log((eos.thermo["Q7"] + 1) * nb * 1e-39 * eos.mn), np.array([nb]) * 1e-39, np.array([ye]), np.array([t]), method="linear"))[0] * 1e39

n_m1 = [0.36e-3 * nb, 2.21e-3 * nb, 0.92e-3 * nb, 0.92e-3 * nb]  # Neutrino number densities [cm-3]
J_m1 = [0.18e-1 * nb, 1.23e-1 * nb, 0.48e-1 * nb, 0.48e-1 * nb]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1/3, 1/3, 1/3, 1/3]  # Eddington factor
dU = 3.48e1


init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1, dU=dU)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': False, 'use_WM_ab': False, 'use_WM_sc': False, 'use_decay': True, 'brem_implementation': 'GP19', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state, opacity_flags, opacity_pars)

try:   
    solver.integrate_to_eq(verbose=True)
except KeyboardInterrupt:
    solver.save_to_csv()