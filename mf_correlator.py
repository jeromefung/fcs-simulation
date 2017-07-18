'''
mf_correlator.py

Jerome Fung

Implement Magatti & Ferri, Applied Optics (2001) multi-tau software correlator.

Use symmetric normalization scheme of Schatzel.
'''

import numpy as np

class LinearCorrelator():
    '''
    Implement a software linear correlator; cascade to get a multi-tau
    correlator
    '''
    def __init__(self, n_elements, update_source, prev_binning_ratio):
        # Number of elements in this correlator
        self.n_elements = n_elements
        # width of bins in units of bin widths in previous stage
        self.prev_binning_ratio = prev_binning_ratio

        # count the number of update operations
        self.counter = 0

        self.source = update_source
        
        # zero out elements
        self.accumulator = np.zeros(self.n_elements)
        self.direct_monitor = np.zeros(self.n_elements)
        self.delayed_monitor = np.zeros(self.n_elements)
        self.channel_indices = np.arange(self.n_elements)

        # shift register: implement as ndarray and use array methods
        self.register = np.zeros(self.n_elements)


    def update(self):
        # get newest element
        new_elt = self.source.push(self.prev_binning_ratio)
        # remove last element in register
        self.register[1:] = self.register[:-1]
        # write newest element in position 0
        self.register[0] = new_elt

        # update delayed monitor -- copy what is currently in register
        self.delayed_monitor += self.register

        # update direct monitor for each channel
        # add newest element if initialized
        self.direct_monitor[self.channel_indices <= self.counter] += new_elt

        # update accumulator
        self.accumulator += new_elt * self.register

        # increment counter
        self.counter += 1

    def push(self, n_elts = 1):
        return (self.register[-n_elts:]).sum()

    def finalize(self):
        # number of elements in each channel
        M_per_channel = self.counter - self.channel_indices
        raw_autocorr = self.accumulator / M_per_channel
        direct_monitor_avg = self.direct_monitor / M_per_channel
        delayed_monitor_avg = self.delayed_monitor / M_per_channel

        # symmetric normalization a la Schatzel
        return raw_autocorr / (direct_monitor_avg * delayed_monitor_avg)


def clockstarts(n_channels, n_max, binning):
    '''
    Calculate array of clock readings at which each multitau stage starts 
    updating
    '''
    # formula holds for all stages past 0
    n_stages_minus1 = np.arange(int(n_max - 1))
    term_by_term = (n_channels - 1) * binning**(n_stages_minus1)
    return np.concatenate((np.array([0]), np.cumsum(term_by_term)))

    
class MultiTauCorrelator():
    def __init__(self, N_stages = 8, N_channels_per_stage = 28,
                 binning_ratio = 4, data_source = None, min_gate_time = 1.):
        self.N_stages = int(N_stages)
        self.N_channels_per_stage = N_channels_per_stage
        self.binning_ratio = int(binning_ratio)
        self.data_source = data_source
        self.min_gate_time = min_gate_time
        self.master_ctr = 0
        self.lc_indices = np.arange(self.N_stages)
        # linear correlator gate times in units of min_gate_time
        self.lc_gates = self.binning_ratio**np.arange(N_stages)
        self.clockstarts = clockstarts(self.N_channels_per_stage, self.N_stages,
                                       self.binning_ratio)
        
        # create list of linear correlators
        self.correlators = []
        # first correlator: source is data
        self.correlators.append(LinearCorrelator(self.N_channels_per_stage,
                                                 self.data_source, 1))
        # use factory to set up subsequent correlators
        for i in np.arange(self.N_stages - 1) + 1:
            # source is the previous correlator added
            self.correlators.append(self.__initialize_cascaded_lc__(self.correlators[-1]))
    

    def __initialize_cascaded_lc__(self, source):
        '''
        Factory for making secondary linear correlators
        '''
        return LinearCorrelator(self.N_channels_per_stage, source,
                                self.binning_ratio)

            
    def update(self):
        # loop over each linear stage
        for correlator, gate, clockstart in zip(self.correlators,
                                                self.lc_gates,
                                                self.clockstarts):
            if self.master_ctr >= clockstart and \
               (self.master_ctr - clockstart) % gate == 0:
                correlator.update()

        # update ticks
        self.master_ctr += 1
                            

    def finalize(self):
        '''
        Merge linear correlators
        '''
        output_correlogram = np.array([]) # list of arrays
        output_times = np.array([])
        for idx, correlator in zip(np.arange(len(self.correlators)),
                                   self.correlators):
            output = correlator.finalize()
            times = np.arange(self.N_channels_per_stage) * \
                    self.binning_ratio**idx
            if idx == 0: # keep all points in first correlator
                output_correlogram = np.append(output_correlogram, output)
                output_times = np.append(output_times, times)
            else: # throw out first 1/binning points
                start_idx = int(self.N_channels_per_stage /
                                self.binning_ratio) 
                output_correlogram = np.append(output_correlogram,
                                               output[start_idx:])
                output_times= np.append(output_times, times[start_idx:])

        return output_times * self.min_gate_time, output_correlogram

    
    
class DataStream(np.ndarray):
    '''
    Subclass ndarray to allow for correlator testing
    '''
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        obj.cntr = 0
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.cntr = getattr(obj, 'cntr', None)

    def push(self, n_elts = 1):
        try:
            self.cntr += 1
            return self[self.cntr - 1] # get current index
        except IndexError:
            self.cntr = 1
            return self[0]
        
    def reset(self):
        self.cntr = 0
    
