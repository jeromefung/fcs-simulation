import numpy as np
import numpy.random as random





class IntensityTrace():
    def __init__(self, L_box, conc, dt, T, D):
        self.L_box = L_box
        self.conc = conc
        self.dt = dt
        self.T = T
        self.D = D
        
        # simulation volume * concentration
        self.N_particles = int(np.rint(self.L_box**3 * self.conc))

        self.n_steps = int(np.ceil(self.T / self.dt))

        # slices: particles
        # rows: timesteps
        # cols: xyz
        self.pos = np.zeros((self.N_particles, self.n_steps, 3))
        # intensity = # of detected photons per time step
        self.intensity = np.zeros(self.n_steps)

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
        stddev = np.sqrt(2. * self.D * self.dt)
        dr = random.randn(3) * stddev

        # box centered at 0, but math easier if centered at (L/2, L/2, L/2)
        # acceptable coordinates range from 0 to L
        shifted_coords = coords + self.L_box / 2.
        new_coords = (shifted_coords + dr) % (self.L_box) # apply PBC

        # re-center at 0, 0, 0
        return new_coords - self.L_box / 2.
