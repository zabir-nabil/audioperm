"""
Helper functions for audioperm.
"""
import numpy as np
import soundfile as sf 

def type_nested(iterable, tp):
    """ Finds if array is of type tp (homogenous).
    Args:
        iterable (list): a list
        tp (type): type of iterable
    Returns:
        bool: If all are of same type.
    """
    if iterable == []:
        return False
    return all(isinstance(item, tp) for item in iterable)

def type_chain(iterable, type_iterable):
    """ Compares the type chain of an iterable, checks with only the first element
    Args:
        iterable (list): a list
        type_iterable (list): type chain
    Returns:
        bool: If the type chain is true for the iterable.
    """
    print(iterable)
    print(type_iterable)
    for c_type in type_iterable:
        print(c_type)
        if type(iterable) != c_type:
            return False
        try:
            iterable = iterable[0]
        except:
            pass
    return True

    

def max_min_heuristics(sig, max_perc = 0.2, min_perc = 0.2):
    """ Calculates the avg max and avg min considering a percentage of sorted amplitudes.
    For audio signals finding a single peak or valley is not enough. So, we take the average of top perc percentage of the population.
    Args:
        sig (ndarray): a numpy array
        max_perc (float): Population percentage for taking max
        min_perc (float): Population percentage for taking max
    Returns:
        (tuple): tuple containing:
            max_p(float): population max for positive signal
            min_p(float): population min for positive signal
            max_n(float): population max for negative signal
            min_n(float): population min for negative signal
    """
    max_p = 0. 
    min_p = 0. 
    max_n = 0. 
    min_n = 0. 

    sig = np.sort(sig.flatten(), kind = 'heapsort')
    sig_p = sig[sig > 0.]
    sig_n = sig[sig < 0.]
    
    
    if len(sig_p) == 0:
        pass # should raise an error
    if len(sig_p) <= 10:
        # dummy test
        max_p, min_p = sig_p.max(), sig_p.min()
    else:
        end_ind = len(sig_p)
        start_ind = int(end_ind * (1 - max_perc))
        max_p = sig_p[start_ind:end_ind].mean()
        start_ind = 0
        end_ind = int(len(sig_p)* min_perc)
        min_p = sig_p[start_ind:end_ind].mean()

    if len(sig_n) == 0:
        pass # should raise an error
    if len(sig_n) <= 10:
        # dummy test
        max_n, min_n = sig_n.min(), sig_n.max()
    else:
        end_ind = len(sig_n)
        start_ind = int(end_ind * (1 - max_perc))
        min_n = sig_n[start_ind:end_ind].mean()
        start_ind = 0
        end_ind = int(len(sig_n)* min_perc)
        max_n = sig_n[start_ind:end_ind].mean()

    return max_p, min_p, max_n, min_n

def noise_boundaries(sig, max_perc = 0.2, min_perc = 0.2):
    """ Calculates maximum noise boundaries for a signal. 
    Args:
        sig (ndarray): a numpy array
        max_perc (float): Population percentage for taking max
        min_perc (float): Population percentage for taking max
    Returns:
        (tuple): tuple containing:
            max_n(float): maximum boundary for noise
            min_n(float): minimum boundary for noise
    """
    max_p, min_p, max_n, min_n = max_min_heuristics(sig, max_perc, min_perc)
    # do some more operations
    snr_p = max_p - min_p
    snr_n = min_n - max_n 

    return min_p + snr_p * max_perc, min_n - snr_n * min_perc

def save_audio(sig, filename, sr = 22050):
    """Takes a PCM 16 or float32 signal and saves the audio in pcm16 format.
    Args:
        sig (ndarray): a numpy array
        filename (str): Filepath and filename.
        sr (int): Sampling rate.
    """
    try:
        if type(sig) != np.ndarray:
            raise TypeError("Expected a numpy array (int16, float32).")
        if sig.dtype == np.int16:
            sf.write(filename, sig, sr, 'PCM_16')
        elif isinstance(sig, np.floating): # style break, will fix later
            sig = np.array(sig * (1<<15), dtype=np.int16)
            sf.write(filename, sig, sr, 'PCM_16')
        else:
            raise TypeError("Expected a numpy array.")
    except Exception as e:
        raise Exception(e)



