import numpy as np
import numpy.random as random
from scipy.integrate import dblquad
from collections import namedtuple

def integrate_circle(func, R, args = None):
    if args is None:
        result, _ = dblquad(func, -R, R,
                            lambda x: -np.sqrt(R**2 - x**2),
                            lambda x: np.sqrt(R**2 - x**2))
    else:
        result, _ = dblquad(func, -R, R,
                            lambda x: -np.sqrt(R**2 - x**2),
                            lambda x: np.sqrt(R**2 - x**2), args = args)
    return result

def PSF_rad(z, R0, alpha):
    return np.sqrt(R0**2 + z**2 * np.tan(alpha))

def CEF(xprime, yprime, z, s0, R0, alpha):
    def integrand(y, x):
        R = PSF_rad(z, R0, alpha)
        rminusrp = np.sqrt((x - xprime)**2 + (y - yprime)**2)
        if rminusrp <= R:
            return 1 / (np.pi * R**2)
        else:
            return 0

    return integrate_circle(integrand, s0)

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
        
    def update_pos(self, coords):
        '''
        Update x, y, and z coordinates of a particle undergoing Brownian
        motion.

        Parameters
        ----------
        coords : ndarray(3)
            x, y, z coordinates of current particle location

        Notes
        -----
        Applies periodic boundary conditions to keep particle within a cubical
        box of side length L.
        '''
        # Delta x, y, and z each Gaussian random variables with mean 0 and
        # variance sigma^2 = 2 D \delta t.
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
        
    def simulate_trace(self, P, c_fluor, kappa, e_photon):
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

        Returns
        -------
        pos : ndarray(n_particles, n_steps, 3)
        intensity : ndarray
        avg_countrate : float
        '''

        # Initialize
        # simulation volume * concentration
        N_particles = int(np.rint(self.sim_params.L_box**3 * c_fluor))
        pos = np.zeros((N_particles, self.sim_params.n_steps, 3))
        intensity = np.zeros(self.sim_params.n_steps)

        pos[:, 0, :] = self.random_particle_distribution(N_particles)

        # Loop over time steps
        for step in np.arange(0, self.sim_params.n_steps):
            # Loop over fluorophores
            for i in np.arange(N_particles):
                # calculate incident intensity
                r = cylindrical_r(*pos[i, step, 0:2])
                z = pos[i, step, 2]
                I = self.optics.intensity_profile(r, z, P)

                # absorbed photons
                # Wohland eq. 11
                N_abs = I / e_photon * self.fluorophore.sigma_abs * \
                        self.sim_params.dt
                #print(N_abs)
                # emitted photons, Wohland eq. 12
                N_e = N_abs * self.fluorophore.q_f

                # CEF
                cef = CEF(pos[i, step, 0], pos[i, step, 1], pos[i, step, 2],
                          self.optics.s0, self.emission_R0,
                          self.optics.obj_half_angle)

                # avg detected photon #
                # See Wohland eq. 13
                N_d_avg = kappa * N_e * cef * self.optics.q_d
                print(N_e, cef, N_d_avg)
                # calculate number of detected photons for this time step
                intensity[step] = intensity[step] + random.poisson(N_d_avg)

                # update position
                try:
                    pos[i, step + 1, :] = self.update_pos(pos[i, step, :])
                except IndexError:
                    pass
            print(step)

        avg_countrate = intensity.sum() / (self.sim_params.n_steps *
                                           self.sim_params.dt)

        return pos, intensity, avg_countrate
