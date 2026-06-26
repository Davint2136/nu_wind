import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy


class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, fluid_e, n_m1, J_m1, chi_m1):
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
        edot_sources = None                                 # Neutrino energy density source terms for integration [MeV cm-3 s-1]
        ndot_sources = None                                 # Neutrino number density source terms for integration [cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.mp_eff = None                                  # Dirac effective proton mass [MeV]
        self.mn_eff = None                                  # Dirac effective neutron mass [MeV]
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.time = 0.0                                        # Time at which the state occurs during the simulation [s]

class Solver:
    def __init__(self, table, state, integrate_tolerance, opacity_flags, opacity_pars):
        self.timestep = None
        self.integrate_tolerance = integrate_tolerance
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars


    def temperature_from_e(self, nb : float, ye : float, e_val : float):
        """
            Given an eos table, nb, ye, and energy density e, calculates the temperature of the fluid. Uses linear interpolation.

            Inputs:
                table [compose.eos.Table]: An EOS table whose nb, yq, and t arrays are not empty. A non-empty thermo["Q7"] table must exist in the EOS Table object as well.
                nb [float]: Baryon number density in fm^-3.
                ye [float]: Unitless charge fraction.
                e_val [float]: The fluid's energy density in MeV/fm^3.

            Outputs:
                t [float]: The fluid temperature in MeV.
        """
        from scipy.optimize import bisect

        # Find difference between calculated e and actual e. Ensure lengths are in fm for PyCompOSE compatibility.
        def f(t):
            interp = self.table.interpolate(np.array([nb]), np.array([ye]), np.array([t]), method='linear')
            e_table = (interp.thermo["Q7"] + 1) * interp.mn * nb
            return e_val - e_table[0, 0, 0]


        # Use a bisection method to find root of f. 
        try:
            t = bisect(f, self.table.t[0], self.table.t[-1], disp=True)
        except RuntimeError:
            print("temperature_from_e could not converge to a temperature.")
        
        return t
    
    def get_potentials(self):
        """
            Calculates the proton, neutron, and electron chemical potentials for the current state.

            Outputs:
                mu_p [float]: The proton chemical potential [MeV]
                mu_n [float]: The neutron chemical potential [MeV]
                mu_e [float]: The electron chemical potential [MeV]
        """
        current_state = self.states[-1]

        interp = self.table.interpolate_3D(np.array([current_state.nb]) * 1e-39, np.array([current_state.ye]), np.array([current_state.t]), method='linear')
        mu_b = (interp.thermo["Q3"][0, 0, 0] + 1) * interp.mn
        mu_q = interp.thermo["Q4"][0, 0, 0] * interp.mn
        mu_l = interp.thermo["Q5"][0, 0, 0] * interp.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self):
        """
            Calulates quantities needed for BNS_NURATES corrections that are relevant at high density. 

            Outputs:
                mp_eff [float]: Proton Dirac effective mass [MeV]
                pn_eff [float]: Neutron Dirac effective mass [MeV]
                dm_eff [float]: The effective mass difference between neutrons and protons
        """

        #Use eos.qK to find dirac effective masses to calculate dm_eff, don't attempt dU for now
        current_state = self.states[-1]

        interp = self.table.interpolate_3D(np.array([current_state.nb]) * 1e-39, np.array([current_state.ye]), np.array([current_state.t]), method='linear')
        mn_eff = interp.qK["mn_d"][0, 0, 0] * interp.mn
        mp_eff = interp.qK["mp_d"][0, 0, 0] * interp.mp
        dm_eff = mn_eff - mp_eff
        return mp_eff, mn_eff, dm_eff

    def calculate_gray_rates(self):
        """
            Calculates neutrino interaction rates based on the fluid's conditions in the current state.

            Outputs:
                gray_rates [dict]: A dictionary consisting of energy and number emissivities, energy and number absorptivities, and scattering absorptivities.
        """

        #Initialize eos_pars and populate
        current_state = self.states[-1]
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = current_state.nb * 1e-21 # Convert baryon number density to nm^-3
        eos_pars.temp = current_state.t
        eos_pars.ye = current_state.ye
        eos_pars.yn = 1 - eos_pars.ye
        eos_pars.yp = eos_pars.ye
        eos_pars.mu_n = current_state.mu_n
        eos_pars.mu_e = current_state.mu_e
        eos_pars.mu_p = current_state.mu_p
        eos_pars.dm_eff = current_state.dm_eff

        #Create a quadrature, populate it with data for a Gauss-Legendre quadrature
        quad = bns.cvar.quadrature_default
        quad.nx = 6
        bns.GaussLegendre(quad)

        #Load reactions and corrections
        opacity_flags = self.opacity_flags
        opacity_pars = self.opacity_pars

        #Load M1 quantities
        m1_pars = bns.M1Quantities()
        m1_pars.chi = current_state.chi_m1
        m1_pars.n = [x * 1e-21 for x in current_state.n_m1]
        m1_pars.J = [x * 1e-21 for x in current_state.J_m1]

        distr_pars = bns.CalculateDistrParamsFromM1(m1_pars, eos_pars)

        #Populate global structure using grey_pars
        grey_pars = bns.GreyOpacityParams()
        grey_pars.eos_pars = eos_pars
        grey_pars.opacity_flags = opacity_flags
        grey_pars.opacity_pars = opacity_pars
        grey_pars.distr_pars = distr_pars
        grey_pars.m1_pars = m1_pars

        #Calculate rates
        gray_rates = bns.ComputeM1Opacities(quad, quad, grey_pars)
        gray_rates['eta']       = [x * 1e21 for x in gray_rates['eta']]
        gray_rates['eta_0']     = [x * 1e21 for x in gray_rates['eta_0']]
        gray_rates['kappa_a']   = [x * 1e7  for x in gray_rates['kappa_a']]
        gray_rates['kappa_0_a'] = [x * 1e7  for x in gray_rates['kappa_0_a']]
        gray_rates['kappa_s']   = [x * 1e7  for x in gray_rates['kappa_s']]
        return gray_rates

    def calculate_e_source_terms(self, J_m1):
        """
            Calculates the neutrino energy density source terms needed for numerical integration based on the conditions in the current simulation state.

            Inputs:
                J_m1 [list]: list of neutrino energy densities sorted by species in the format [nue, anue, nux, anux]
            
            Outputs:
                e_terms [numpy.ndarray]: A numpy array containing the energy density source terms in the format [nue, anue, nux, anux]
        """

        rates = self.states[-1].rates
        e_terms = [None, None, None, None]

        for i in range(0, 4):
            e_terms[i] = rates["eta"][i] - (rates["kappa_a"][i] * J_m1[i])

        return np.array(e_terms)
    
    def calculate_n_source_terms(self, n_m1):
        """
            Calculates the neutrino number density source terms needed for numerical integration based on the conditions in the current simulation state.

            Inputs:
                J_m1 [list]: list of neutrino number densities sorted by species in the format [nue, anue, nux, anux]
            
            Outputs:
                e_terms [numpy.ndarray]: A numpy array containing the energy density source terms in the format [nue, anue, nux, anux]
        """
        rates = self.states[-1].rates
        n_terms = [None, None, None, None]
        
        for i in range(0, 4):
            n_terms[i] = rates["eta_0"][i] - (rates["kappa_0_a"][i] * n_m1[i])

        return np.array(n_terms)

    def integrate_step(self):
        """
            Integrates the simulation forward in time by one timestep. Finds the next fluid energy density and electron fraction using a variable step size RK2 integrator, where the timestep corresponds
            to a small fraction of the absorptivity corresponding to the shortest timescale.

            Outputs:
                next_e [float]: The fluid's energy density at the next timestep of the simulation.
                next_ye [float]: The fluid's electron fraction at the next timestep of the simulation.
                next_time [float]: The simulation time at the next timestep.
        """

        # Calculate timestep
        c = 29979245800.0
        timestep = 0.01 * ((1 / max(self.states[-1].rates["kappa_a"] + self.states[-1].rates["kappa_0_a"])) / c)

        #Calculate source terms for current state
        self.states[-1].edot_sources = self.calculate_e_source_terms(self.states[-1].J_m1)
        self.states[-1].ndot_sources = self.calculate_n_source_terms(self.states[-1].n_m1)

        #Calculate k1 for RK2
        k1_e = timestep * self.states[-1].edot_sources
        k1_n = timestep * self.states[-1].ndot_sources

        #Calculate source terms at midpoint of timestep
        mid_edot_sources = self.calculate_e_source_terms((np.array(self.states[-1].J_m1) + (k1_e / 2)).tolist())
        mid_ndot_sources = self.calculate_n_source_terms((np.array(self.states[-1].n_m1) + (k1_n / 2)).tolist())

        #Calculate k2 for RK2
        k2_e = timestep * mid_edot_sources
        k2_n = timestep * mid_ndot_sources

        #Calculate fluid energy density and electron fraction at the next timestep
        next_e = self.states[-1].fluid_e + (float(-np.sum(k2_e)))
        next_ye = self.states[-1].ye + float((k2_n[1] - k2_n[0]) / self.states[-1].nb)

        next_time = self.states[-1].time + timestep
        return next_e, next_ye, next_time



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

## Input thermodynamic quantities (corresponding to point A in Chiesa+25 PRD)
##  N.B.: chemical potentials include the rest mass contribution
nb   = 4.208366627847035e+38  # Baryon number density [cm-3]      
e_nb = 4.3441769719367175e+41 # Baryon energy density [MeV cm-3]
temp = 12.406403541564941     # Temperature [MeV]
ye   = 0.07158458232879639    # Electron fraction
xn   = 1. - ye                # Neutron fraction
xp   = ye                     # Proton fraction
mu_e = 187.1814489            # Electron chemical potential [MeV]
mu_p = 1011.01797737          # Proton chemical potential [MeV]
mu_n = 1221.59013681          # Neutron chemical potential [MeV]
dU = 18.92714728              # Nucleon interaction potential difference (Un-Up) [MeV]
mp_eff = 278.87162217         # Proton effective mass [MeV]
mn_eff = 280.16495513         # Neutron effective mass [MeV]
dm = mn_eff - mp_eff          # Nucleon effective mass difference [MeV]

n_m1 = [3.739749408027436e+33, 1.2174961961689319e+35, 2.2438496448164613e+34, 2.2438496448164613e+34]  # Neutrino number densities [cm-3]
J_m1 = [1.246583136009145e+35,  5.360307484839323e+36,  8.726081952064015e+35,  8.726081952064015e+35]  # Neutrino energy densities [MeV cm-3]
chi_m1  = [1./3., 1./3., 1./3., 1./3.]  # Eddington factor

init_state = State(nb, ye, e_nb, n_m1, J_m1, chi_m1)

opacity_flags = {'use_abs_em': True, 'use_pair': True, 'use_brem': True, 'use_inelastic_scatt': True, 'use_iso': True}
opacity_pars = {'use_dU': False, 'use_dm_eff': True, 'use_WM_ab': True, 'use_WM_sc': True, 'use_decay': True, 'brem_implementation': 'HR98', 'neglect_blocking': False, 'use_NN_medium_corr': True}

solver = Solver(eos, init_state, None, opacity_flags, opacity_pars)



