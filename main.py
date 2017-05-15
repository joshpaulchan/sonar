#!/usr/bin/env python
"""
@name   main.py
@desc   Logic of single-dimension radar

@auth   Joshua Paul A. Chan
"""
from matplotlib import pyplot as plt
import numpy as np

############################### configuration #############################

FREQ_SIGNAL = 12000         # hz
FREQ_SAMPLE = 2*FREQ_SIGNAL # satisfy nyquist
MAX_LISTEN = 1

V_SND = 343        # speed of sound (m/s)
SIM_DIST = 20      # dist of object in s
TIME_DELAY = 2*SIM_DIST/V_SND  # round-trip (2x)
print("Simulated distance: ", SIM_DIST)
print("Simulated time delay: ", TIME_DELAY)

def extract_delay(delayed_fn, ideal_fn):
    """
    Calculates the time delay of a time-domain signal given the evaluation of
    both functions on the same time interval.
    
    @pre    the time interval they were both evaluated on must be greater than
    the delay 
    
    @param  {number[]}  delayed_fn  the output of the time-delayed function
    @param  {number[]}  ideal_fn    the output of the ideal function
    @return {number}    an approximation of the time delay b/w the two functions
    """
    # TODO: code here
    return 1

def delay_by(fn, t0=1):
    """
    Delays the output of a vectorized time-domain function `nf` by `t0` units.
    
    @param  {func}      fn  the vectorized time-domain function to delay
    @param  {number}    t0  the time delay to use (default 1)
    @return {func}      the delayed version of the function
    """
    def wrapper(t):
        """wrapper fn"""
        return np.piecewise(t, [t < t0, t >= t0], [0, lambda t: fn(t - t0)])
    
    return wrapper
    
def apply_noise(fn, snr=0.5):
    """
    Applies a specific amount of noise (quanitified by signal-to-noise ratio
    `snr`) to vectorized time-domain function `fn`.
    
    @pre    `snr` must be a float from in the range [0, 1]
    
    @param  {func}  fn  the vectorized time-domain function to noise
    @param  {float} snr the amount of noise to apply (default 0.5)
    @return {func}  the function, all noised up 
    """
    def wrapper(t):
        """wrapper fn"""
        return fn(t) + np.random.normal(0, snr, len(t))
    
    return wrapper

def ideal(t, freq=1):
    """
    Generates the ideal radar signal.
    
    @pre    `freq` must be a non-zero number
    
    @param  {number}    freq    the frequency to use
    @return {number[]}  array of numbers
    """
    return np.sin((2*np.pi/freq)*t)
    
def main():
    """Main"""
    
    t = np.arange(0, MAX_LISTEN, 1/FREQ_SAMPLE)
    
    ############################# signal transmitter ###########################
    
    # base & plot
    ideal_fn = ideal
    
    xi = ideal_fn(t, FREQ_SIGNAL)
    
    Xi = np.fft.fft(xi)
    freq = np.fft.fftfreq(t.shape[-1])
    
    # plt.figure()
    # plt.title("Transmitted signal")
    # 
    # plt.subplot(211)
    # plt.plot(t, xi)
    # plt.grid()
    # plt.xlabel('t')
    # plt.ylabel('xi(t)')
    # 
    # plt.subplot(212)
    # plt.xticks(freq, list(['{}𝝿'.format(n/np.pi) for n in freq]))
    # plt.plot(freq, Xi.real)
    # plt.grid()
    # plt.xlabel('theta')
    # plt.ylabel('Xi(theta)')
    
    # delay
    delayed_fn = delay_by(ideal_fn, TIME_DELAY)
    # for delay in freq
    # http://stackoverflow.com/questions/31586803/delay-a-signal-in-time-domain-with-a-phase-change-in-the-frequency-domain-after
    
    # noise
    tx_fn = apply_noise(delayed_fn, 0.025)
    
    # debug
    plt.figure()
    
    plt.subplot(311)
    plt.plot(ideal_fn(t))
    
    plt.subplot(312)
    plt.plot(delayed_fn(t))
    
    plt.subplot(313)
    plt.plot(tx_fn(t))
    plt.show()
    
    ############################## signal receiver ############################
    
    # sample (evaluate), fft & plot
    # y = fi(t)
    # 
    # Y = np.fft.fft(y)
    # freq = np.fft.fftfreq(t.shape[-1])
    # 
    # plt.figure()
    # plt.grid()
    # plt.title("Received signal")
    # 
    # plt.subplot(211)
    # plt.plot(t, y)
    # plt.grid()
    # plt.xlabel('t')
    # plt.ylabel('y(t)')
    # 
    # plt.subplot(212)
    # plt.plot(freq, Y.real, freq, Y.imag)
    # plt.grid()
    # plt.xlabel('theta')
    # plt.ylabel('Y(theta)')
    
    # filter
    
    ############################# spectrum analysis ############################
    
    # phase = np.arctan((Xi.imag/Xi.real))
    # deriv = np.gradient(phase)
    # tde = np.mean(deriv)
    # 
    # # print(phases)
    # # print("Phase shift: ", phase)
    # plt.figure()
    # plt.grid()
    # plt.title('Phase Shift')
    # plt.plot(phase)
    # plt.xlabel('theta')
    # plt.ylabel('phase')
    # 
    # print("Analyzed time delay: ", tde)
    # print("Analyzed distance: ", tde*v_sound)
    
    # show all plots
    plt.show()

if __name__ == '__main__':
    main()
