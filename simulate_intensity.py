import numpy as np
import numpy.random as random
from collections import namedtuple
from fast_cef import CEF 

def cylindrical_r(x, y):
    return np.sqrt(x**2 + y**2)


Fluorophore = namedtuple('Fluorophore', ['D', 'sigma_abs', 'q_f', 'wavelen_em'])
'''
Attributes:
-----------
D : float
    diffusion constant
sigma_abs : float
    absorption cross section
qf : float
    fluorescence quantum yield
wavelen_em : float
    emission wavelength
'''

class SimParams():
    def __init__(self, dt, T, L_box):
        '''
        Inputs
        ------
        dt : float
            Time step size
        T : float
            Total length of simulation
        L_box : float
            Side length of simulation PBC box
        '''
        self.dt = dt
        self.T = T
        self.L_box = L_box

    @property
    def n_steps(self):
        return int(np.ceil(self.T / self.dt))
        

class Optics():
    def __init__(self, wavelen, n_med, n_immersion, NA, mag, d_pinhole,
                 q_d):
        '''
        Describes experimental hardware setup.

        Inputs
        ------
        wavelen (float)
            Incident vacuum wavelength
        n_med (float)
            Sample medium refractive index
        n_immersion (float)
            Objective immersion medium refractive index
        NA (float)
            Objective numerical aperture
        mag (float)
            Objective magnification
        d_pinhole (float)
            Pinhole physical diameter
        q_d (float)
            Detector quantum efficiency
        '''
        self.wavelen = wavelen
        self.n_med = n_med
        self.n_immersion = n_immersion
        self.NA = NA
        self.mag = mag
        self.d_pinhole = d_pinhole
        self.q_d = q_d

    @property
    def wavelen_med(self):
        '''
        Wavelength in the medium
        '''
        return self.wavelen / self.n_med

    @property
    def incident_w0(self):
        '''
        Calculate diffraction-limited incident laser minimum spot size
        '''
        return 1.22 * self.wavelen / (2. * self.NA)

    @property
    def obj_half_angle(self):
        '''
        Objective acceptance cone half-angle in radians
        '''
        return np.arcsin(self.NA / self.n_immersion)

    @property
    def s0(self):
        '''
        Projected pinhole radius
        '''
        return self.d_pinhole / self.mag / 2.
    
    def intensity_profile(self, r, z, P = 1.):
        '''
        Calculate Gaussian beam profile of incident laser in the sample.
        See Wohland eq. 9.

        Inputs
        ------
        r, z (float)
            Cylindrical coordinates of location where intensity is calculated
        P (optional, float)
            Incident laser power at the sample. Defaults to 1, returning
            intensity / incident power.
        '''
        denom = np.pi * self.incident_w0**2 * \
                (1 + (z * self.wavelen_med / (np.pi * self.incident_w0**2))**2)
        I = 2. * P / denom * np.exp(-2 * r**2 * np.pi / denom)
        return I

class IntensityTrace():
    def __init__(self, sim_params, optics, fluorophore):
        '''
        Inputs
        ------
        sim_params 
            SimParams object
        optics
            Optics object describing experimental setup
        fluorophore (namedtuple)
            Describes molecular properties of fluorophore being used
        '''
        self.sim_params = sim_params
        self.optics = optics
        self.fluorophore = fluorophore
        
    @property
    def emission_R0(self):
        '''
        Calculate minimum emission spot radius.
        Assume diffraction-limited performance and use Rayleigh criterion.
        '''
        return 0.61 * self.fluorophore.wavelen_em / (2. * self.optics.NA)
        
    def update_pos(self, coords, dr = None):
        '''
        Update x, y, and z coordinates of a particle undergoing Brownian
        motion.

        Parameters
        ----------
        coords : ndarray(3)
            x, y, z coordinates of current particle location
        dr : ndarray(3), optional
            If provided, the random step to give the particle

        Notes
        -----
        Applies periodic boundary conditions to keep particle within a cubical
        box of side length L.
        '''
        # Delta x, y, and z each Gaussian random variables with mean 0 and
        # variance sigma^2 = 2 D \delta t.
        if dr is None:
            stddev = np.sqrt(2. * self.fluorophore.D * self.sim_params.dt)
            dr = random.randn(3) * stddev

        # box centered at 0, but math easier if centered at (L/2, L/2, L/2)
        # acceptable coordinates range from 0 to L
        shifted_coords = coords + self.sim_params.L_box / 2.
        new_coords = (shifted_coords + dr) % (self.sim_params.L_box) # PBC

        # re-center at 0, 0, 0
        return new_coords - self.sim_params.L_box / 2.

    def random_particle_distribution(self, N):
        '''
        Randomly distribute N particles in a cubical box
        '''
        coords = (np.random.rand(N * 3) - 0.5) * self.sim_params.L_box
        return coords.reshape((-1, 3))
        
    def simulate_trace(self, P, c_fluor, kappa, e_photon, save_coords = False,
                       update_val = 1e6):
        '''
        Do the simulation.

        Inputs
        ------
        P (float)
            Incident laser power at the sample
        c_fluor (float)
            Fluorophore concentration (# particles/volume)
        kappa (float)
            Fraction of emission photons captured by imaging system.
            Includes geometric factors due to NA, losses, etc.
        e_photon (float)
            Single-photon energy in units consistent with P.
        save_coords (Boolean)
            If True, return fluorophore coordinates. Not recommended for 
            long runs.
        update_val (numeric)
            If n_steps > update_val, display an update every update_val 
            timesteps

        Returns
        -------
        intensity : ndarray
        avg_countrate : float
        pos : ndarray(n_particles, n_steps, 3)        
        '''

        # Initialize
        # simulation volume * concentration
        N_particles = int(np.rint(self.sim_params.L_box**3 * c_fluor))
        intensity = np.zeros(self.sim_params.n_steps)

        if save_coords:
            pos = np.zeros((N_particles, self.sim_params.n_steps, 3))
            pos[:, 0, :] = self.random_particle_distribution(N_particles)
        else:
            pos = np.zeros((N_particles, 3))
            pos = self.random_particle_distribution(N_particles)
            
        # Loop over time steps
        for step in np.arange(0, self.sim_params.n_steps):
            # calculate random steps
            stddev = np.sqrt(2. * self.fluorophore.D * self.sim_params.dt)
            dr = np.random.randn(3 * N_particles).reshape((-1, 3)) * stddev
            
            # Loop over fluorophores
            for i in np.arange(N_particles):
                # get coords
                if save_coords:
                    x, y = pos[i, step, 0:2]
                    z = pos[i, step, 2]
                else:
                    x, y = pos[i, 0:2]
                    z = pos[i, 2]
                r = cylindrical_r(x, y)
                
                # calculate incident intensity
                I = self.optics.intensity_profile(r, z, P)

                # absorbed photons
                # Wohland eq. 11
                N_abs = I / e_photon * self.fluorophore.sigma_abs * \
                        self.sim_params.dt
                #print(N_abs)
                # emitted photons, Wohland eq. 12
                N_e = N_abs * self.fluorophore.q_f

                # CEF
                cef = CEF(x, y, z, self.optics.s0, self.emission_R0,
                          self.optics.obj_half_angle)

                # avg detected photon #
                # See Wohland eq. 13
                N_d_avg = kappa * N_e * cef * self.optics.q_d
                
                # calculate number of detected photons for this time step
                # in edge cases N_d_avg might be numerically close to 0
                # but negative -- take absolute value
                intensity[step] = intensity[step] + random.poisson(abs(N_d_avg))

                # update position
                if save_coords:
                    try:
                        pos[i, step + 1, :] = self.update_pos(pos[i, step, :],
                                                              dr[i])
                    except IndexError: # sloppy about last step
                        pass
                else:
                    pos[i, :] = self.update_pos(pos[i, :], dr[i])

            # provide a crude progress update
            if self.sim_params.n_steps > update_val:
                if step % update_val == 0:
                    print('Timestep: ', step)
            
        avg_countrate = intensity.sum() / (self.sim_params.n_steps *
                                           self.sim_params.dt)

        if save_coords:
            return intensity, avg_countrate, pos
        else:
            return intensity, avg_countrate
