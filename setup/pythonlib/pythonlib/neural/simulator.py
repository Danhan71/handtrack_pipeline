
# NOTE: some stuff taken from https://elephant.readthedocs.io/en/latest/tutorials/gpfa.html



import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


class Simulator():
    """
    for simulating neural datasets
    """
    def __init__(self, params=None):
        
        self.setParams()


    def setParams(self, params=None, ver=None):
        """ generate params either by entry (paramsd) or
        default (params = None). Will save in self.Params
        - ver is string, this will add on self.Params[ver], which is
        another dict holding params for this version of dynamics. 
        e.g. self.Params["oscill"] will be dict params for 
        generating oscillation.
        NOTE:
        - all times are sec, unless using units objects.
        - self.X... are all (ndim, ntime), if no trials. if trials,
        then (ntrials, ndim, ntime). this is also true for lsits of lists,
        e.g., self.Xspikes[ntrials][ndim]

        """

        if params is None:
            print("applied default params")
            self.Params = {
                "ver":"oscill",
            }
            self.Params["projection"] = {
                "embedding_dimension":10,
                "loc":0,
                "scale":None
            }
            self.Params["makespikes"] = {
                "num_trials":20,
                "timestep":1 * pq.ms
            }
            self.Params["behavior"] = {
                "embedding_dimension":2,
                "loc":0,
                "scale":None
            }
        else:
            print('applied entered params')
            self.Params = params

        # == params specific for each dynamics verison
        if params is None and ver is not None:
            print("OK: default params, user dynanmics params")
        elif params is None and ver is None:
            print("OK: no dynamics params added")
        elif params is not None and ver is not None:
            print("WARNING: overwriting self.Params[ver] (from params) with input ver")
        elif params is not None and ver is None:
            print("OK: using input params")
        
        if ver is not None:
            if ver=="oscill":
                self.Params["oscill"] = {
                    "dt":0.001,
                    "num_steps":1000,
                }
            else:
                print(ver)
                assert False, "not coded, this ver dynamics type"


    def generate(self):
        """ generate instance of stimilation from latent
        dynamics
        """
        
        # == 1) Get latent dynamics
        if self.Params["ver"]=="lorenz":
            assert False, "not coded"
            self.Xlatent = self.integrated_lorenz()
        elif self.Params["ver"]=="oscill":
            self.T, self.Xlatent = self.oscillator()
        else:
            print(self.Params["ver"])
            assert False, "dont knwo thisd"

        # == 2) [optional] random projection
        if self.Params["projection"] is not None:
            print("doing random projection of latent dyanmics")
            self.XlatentProj = self.randomProjection(self.Xlatent, self.Params["projection"])
            # == 3) convert to firing rates
            self.Xactivity = self.convertInstantRate(self.XlatentProj)
        else:
            # == 3) convert to firing rates
            self.Xactivity = self.convertInstantRate(self.Xlatent)


        # == 3) [optional] copnvert to spikes
        if self.Params["makespikes"] is not None:
            self.Xspikes = self.makeSpikes(self.Xactivity, self.Params["makespikes"])

        print("DONE GENERATING!")

    def generateBehavior(self):
        """ projection of neural activity, fake "behavior"
        """
        print("Behavior to self.Ybeh")
        self.Ybeh = self.randomProjection(self.Xactivity, self.Params["behavior"])


    def oscillator(self):
        def integrated_oscillator(dt, num_steps, x0=0, y0=1, angular_frequency=2*np.pi*1e-3):
            """
            Parameters
            ----------
            dt : float
                Integration time step in ms.
            num_steps : int
                Number of integration steps -> max_time = dt*(num_steps-1).
            x0, y0 : float
                Initial values in three dimensional space.
            angular_frequency : float
                Angular frequency in 1/ms.

            Returns
            -------
            t : (num_steps) np.ndarray
                Array of timepoints
            (2, num_steps) np.ndarray
                Integrated two-dimensional trajectory (x, y, z) of the harmonic oscillator
            """

            assert isinstance(num_steps, int), "num_steps has to be integer"
            t = dt*1000*np.arange(num_steps)
            x = x0*np.cos(angular_frequency*t) + y0*np.sin(angular_frequency*t)
            y = -x0*np.sin(angular_frequency*t) + y0*np.cos(angular_frequency*t)
            return t, np.array((x, y))

        if "num_steps" not in self.Params["oscill"]:            
            num_steps = np.ceil(params["duration"]/dt)


        t, X = integrated_oscillator(**self.Params["oscill"])

        return t, X

    
    def integrated_lorenz(dt, num_steps, x0=0, y0=1, z0=1.05,
                          sigma=10, rho=28, beta=2.667, tau=1e3):
        """

        Parameters
        ----------
        dt :
            Integration time step in ms.
        num_steps : int
            Number of integration steps -> max_time = dt*(num_steps-1).
        x0, y0, z0 : float
            Initial values in three dimensional space
        sigma, rho, beta : float
            Parameters defining the lorenz attractor
        tau : characteristic timescale in ms

        Returns
        -------
        t : (num_steps) np.ndarray
            Array of timepoints
        (3, num_steps) np.ndarray
            Integrated three-dimensional trajectory (x, y, z) of the Lorenz attractor
        """
        def _lorenz_ode(point_of_interest, timepoint, sigma, rho, beta, tau):
            """
            Fit the model with `spiketrains` data and apply the dimensionality
            reduction on `spiketrains`.

            Parameters
            ----------
            point_of_interest : tuple
                Tupel containing coordinates (x,y,z) in three dimensional space.
            timepoint : a point of interest in time
            dt :
                Integration time step in ms.
            num_steps : int
                Number of integration steps -> max_time = dt*(num_steps-1).
            sigma, rho, beta : float
                Parameters defining the lorenz attractor
            tau : characteristic timescale in ms

            Returns
            -------
            x_dot, y_dot, z_dot : float
                Values of the lorenz attractor's partial derivatives
                at the point x, y, z.
            """

            x, y, z = point_of_interest

            x_dot = (sigma*(y - x)) / tau
            y_dot = (rho*x - y - x*z) / tau
            z_dot = (x*y - beta*z) / tau
            return x_dot, y_dot, z_dot

        assert isinstance(num_steps, int), "num_steps has to be integer"

        t = dt*np.arange(num_steps)
        poi = (x0, y0, z0)
        return t, odeint(_lorenz_ode, poi, t, args=(sigma, rho, beta, tau)).T

    def makeSpikes(self, data, params):
        """ get multiple trials of data, randomly sampling spikes
        - data, (ndim x time)
        - params["num_trials"], num trials
        RETURNS:
        - (ntrials x ndim x time)
        """
        import neo
        from elephant.spike_train_generation import inhomogeneous_poisson_process

        def spikes(instantaneous_rates, num_trials, timestep):
            """
            Parameters
            ----------
            instantaneous_rates : np.ndarray
                Array containing time series.
            timestep :
                Sample period.
            num_steps : int
                Number of timesteps -> max_time = timestep*(num_steps-1).

            Returns
            -------
            spiketrains : list of neo.SpikeTrains
                List containing spiketrains of inhomogeneous Poisson
                processes based on given instantaneous rates.

            """

            spiketrains = []
            for _ in range(num_trials):
                spiketrains_per_trial = []
                for inst_rate in instantaneous_rates:
                    # print(inst_rate.shape)
                    # print(inst_rate)
                    anasig_inst_rate = neo.AnalogSignal(inst_rate, sampling_rate=1/timestep, units=pq.Hz)
                    # print(inst_rate)
                    # print(np.mean(inst_rate))
                    # print(anasig_inst_rate)
                    spiketrains_per_trial.append(inhomogeneous_poisson_process(anasig_inst_rate, as_array=True))
                    # print(spiketrains_per_trial)
                    # assert False
                spiketrains.append(spiketrains_per_trial)

            return spiketrains

        spiketrains = spikes(data, **params)
        return spiketrains


    def randomProjection(self, data, params):
        """ random projection of data.
        - data, (n, time),
        - params, dict.
        """

        def proj(data, embedding_dimension, loc=0, scale=None):
            """
            Parameters
            ----------
            data : np.ndarray
                Data to embed, shape=(M, N)
            embedding_dimension : int
                Embedding dimension, dimensionality of the space to project to.
            loc : float or array_like of floats
                Mean (“centre”) of the distribution.
            scale : float or array_like of floats
                Standard deviation (spread or “width”) of the distribution.

            Returns
            -------
            np.ndarray
               Random (normal) projection of input data, shape=(dim, N)

            See Also
            --------
            np.random.normal()

            """
            if scale is None:
                scale = 1 / np.sqrt(data.shape[0])
            projection_matrix = np.random.normal(loc, scale, (embedding_dimension, data.shape[0]))
            return np.dot(projection_matrix, data)

        X = proj(data, **params)
        print(f"done: random projection to shape {X.shape}")
        return X

    def convertInstantRate(self, data, max_rate = 70):
        """ convert latent dynamics into instantaneous rates (positive)
        by norming and taking power, for poiosson process
        - max_rate, you decide, in hz
        """
        normed_traj = data / data.max()
        data_rates = np.power(max_rate, normed_traj)
        return data_rates



    def plot(self):
        print("PLOT: currently only working for oscillator.")
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

        oscillator_trajectory_2dim = self.Xlatent
        num_spiketrains = self.Xactivity.shape[0]
        oscillator_trajectory_Ndim = self.XlatentProj
        oscillator_trajectory_Ndim_activity = self.Xactivity
        spiketrains_oscillator = self.Xspikes
        times_oscillator = self.T
        beh = self.Ybeh

        ax1.set_title('2-dim Harmonic Oscillator')
        ax1.set_xlabel('time [s]')
        for i, y in enumerate(oscillator_trajectory_2dim):
            ax1.plot(times_oscillator, y, label=f'dimension {i}')
        ax1.legend()

        ax2.set_title('Trajectory in 2-dim space')
        ax2.set_xlabel('Dim 1')
        ax2.set_ylabel('Dim 2')
        ax2.set_aspect(1)
        ax2.plot(oscillator_trajectory_2dim[0], oscillator_trajectory_2dim[1])

        ax3.set_title(f'Projection to {num_spiketrains}-dim space')
        ax3.set_xlabel('time [s]')
        y_offset = oscillator_trajectory_Ndim.std() * 3
        for i, y in enumerate(oscillator_trajectory_Ndim):
            ax3.plot(times_oscillator, y + i*y_offset)

        ax5.set_title(f'Projection to {num_spiketrains}-dim space (and convert to hz)')
        ax5.set_xlabel('time [s]')
        y_offset = oscillator_trajectory_Ndim_activity.std() * 3
        for i, y in enumerate(oscillator_trajectory_Ndim_activity):
            ax5.plot(times_oscillator, y + i*y_offset)

        trial_to_plot = 0
        ax4.set_title(f'Raster plot of trial {trial_to_plot}')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Spike train index')
        for i, spiketrain in enumerate(spiketrains_oscillator[trial_to_plot]):
            ax4.plot(spiketrain, np.ones_like(spiketrain) * i, ls='', marker='|')

        ax6.set_title(f'behavior')
        ax6.set_xlabel('time [s]')
        y_offset = beh.std() * 3
        for i, y in enumerate(beh):
            ax6.plot(times_oscillator, y + i*y_offset)



        plt.tight_layout()
        # plt.show()        