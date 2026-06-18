import numpy as np
import bnsnurates as bns
import os
import sys
from compose.eos import Metadata, Table
from copy import deepcopy


class State:
    """Class showing the state of the simulation."""
    def __init__(self, nb, ye, e_nb, fluid_e, n_m1, J_m1, chi_m1):
        self.nb = nb                                        # Baryon number density [cm-3]
        self.e_nb = e_nb                                    # Baryon energy density [MeV/cm^-3]
        self.ye = ye                                        # Electron fraction
        self.xn = 1. - ye                                   # Neutron fraction
        self.xp = ye                                        # Proton fraction
        self.fluid_e  = fluid_e                             # Fluid energy density [MeV]
        self.n_m1 = n_m1                                    # Gray neutrino number densities. Order: nue, anue, nux, anux [cm-3]
        self.J_m1 = J_m1                                    # Gray neutrino energy densities. Order: nue, anue, nux, anux [MeV cm-3]
        self.chi_m1 = chi_m1                                # Eddington factor
        self.t = None                                       # Fluid temperature [MeV]
        self.mu_e = None                                    # Electron chemical potential [MeV]
        self.mu_p = None                                    # Proton chemical potential [MeV]
        self.mu_n = None                                    # Neutron Chemical Potential [MeV]
        self.source_terms = {"edot" : None, "ndot" : None}  # Neutrino transport source terms [MeV cm-3 s-1, cm-3 s-1]
        self.rates = None                                   # Neutrino reaction rates 
        self.dm_eff = None                                  # Nucleon effective mass difference [MeV]
        self.dU = None                                      # Nucleon interaction potential difference [MeV]


class Solver:
    def __init__(self, table, state, integrate_tolerance, opacity_flags, opacity_pars):
        self.timestep = None
        self.integrate_tolerance = integrate_tolerance
        self.table = table
        self.states = [state]
        self.opacity_flags = opacity_flags
        self.opacity_pars = opacity_pars


    def temperature_from_e(self, nb : np.ndarray, ye : np.ndarray, e_val : float):
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
        from scipy.optimize import bisect

        # Find difference between calculated e and actual e. Ensure lengths are in fm for PyCompOSE compatibility.
        def f(t):
            interp = self.table.interpolate(nb * 1e-39, ye, np.array([t]), method='linear')
            e_table = (interp.thermo["Q7"] + 1) * interp.mn * nb
            return e_val - (e_table[0, 0, 0] * 1e-39)


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
        interp = self.table.interpolate3D(self.states[-1].nb * 1e-39, self.states[-1].ye, self.states[-1].t, method='linear')
        mu_b = (self.table.thermo["Q3"][0, 0, 0] + 1) * self.table.mn
        mu_q = self.table.thermo["Q4"][0, 0, 0] * self.table.mn
        mu_l = self.table.thermo["Q5"][0, 0, 0] * self.table.mn

        mu_p = mu_b + mu_q #(Baryon number: 1, Charge number: +1, Lepton number: 0)
        mu_n = mu_b #(Baryon number: 1, Charge number: 0, Lepton number: 0)
        mu_e = mu_l - mu_q #(Baryon number: 0, Charge number: -1, Lepton number: 1)

        return mu_p, mu_n, mu_e

    def calculate_corrector_quantities(self):
        # TODO: Use eos.micro to find dirac effective masses to calculate dm_eff, don't attempt dU for now
        interp = self.table.interpolate3D(self.states[-1].nb * 1e-39, self.states[-1].ye, self.states[-1].t, method='linear')
        mn_eff = interp.micro["mn_d"][0, 0, 0]
        mp_eff = interp.micro["mp_d"][0, 0, 0]
        dm_eff = mn_eff - mp_eff
        return dm_eff

    def calculate_gray_rates(self):
        # TODO: Extremely similar to bns_nurates' test_bindings.py. Use the same code structure
        
        #Initialize eos_pars and populate
        current_state = self.states[-1]
        eos_pars = bns.MyEOSParams()
        eos_pars.nb = current_state.nb[0] * 1e-21 # Convert baryon number density to nm^-3
        eos_pars.temp = current_state.t[0]
        eos_pars.ye = current_state.ye[0]
        eos_pars.xn = 1 - eos_pars.ye
        eos_pars.xp = eos_pars.ye
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
        grey_rates = bns.ComputeM1Opacities(quad, quad, grey_pars)

        return grey_rates

    def calculate_source_terms(self):
        pass

    def integrate_step(self):
        # TODO: Build and test RK2 integrator before implementing anything here. Best to do this in a separate file.
        pass


