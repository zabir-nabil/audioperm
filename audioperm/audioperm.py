import itertools

import numpy as np
import librosa
import pydub
import soundfile as sf

from audioperm.utils import type_nested, noise_boundaries, type_chain

class AudioPerm:
    """
    The main class for audioperm. Takes an audio file (or a batch of files) path or numpy array (int16, float). Internal audio representation is pcm 16 (not same as librosa default).
    """
    def __init__(self, audio, sr = 22050, **kwargs):
        """ Reads audio files.
        Args:
            audio (Union[:obj:`list` of :obj:`str`, :obj:`list` of :obj:`ndarray`, ndarray, str]): A list of file paths (str) or A list of numpy array (PCM16, 32FP) 
            sr (int): Sampling rate of audio
        """
        # if everything is okay
        self.audio_type = type(audio)
        self.audio_files = read_audio(audio, sr = sr, return_as_array = True)
        self.sr = sr
        self.words = []


    def word_segments(self, silence_thresh = -60., min_silence_len = 5, return_words = True):
        """ Segments the audio files into multiple segments or words.
        TODO: Improve word segmentation. Add label wise segmentation (If given n words as labels, find n appropriate words).

        Args:
            silence_thresh (float): Silence threshold for segmenting the audio. Same as pydub.
            min_silence_len (int): Minimum silence lenth (in ms). Same as pydub.
        
        Returns:
            Union[:obj:`list` of :obj:`list` of :obj:`ndarray`, :obj:`list` of :obj:`ndarray`]

        """
        self.words = word_segments(self.audio_files, self.sr, silence_thresh=silence_thresh, min_silence_len=min_silence_len, return_as_array = True)
        
        if return_words == True:
            if self.audio_type == str:
                return self.words[0]
            elif self.audio_type == np.ndarray:
                return self.words[0]
            elif self.audio_type == list:
                return self.words
            else:
                raise TypeError(f"audio_type is {self.audio_type}, expected: str, np.ndarray, or list")

    def permutations(self, n_permutations = 1, interm_silence = 1000):
        """Get the permutation of words.
        TODO: Use yield.

        Args:
            n_permutations (int): Number of (max) permutations to return
            interm_silence (int): Intermediate silence between words (in ms).
        Returns:
            Union[:obj:`list` of :obj:`list` of :obj:`ndarray`, :obj:`list` of :obj:`ndarray`]
        """
        audio_perms = permutations(self.words, n_permutations = n_permutations, interm_silence = interm_silence, return_as_array = True)

        if self.audio_type == str:
            return audio_perms[0]
        elif self.audio_type == np.ndarray:
            return audio_perms[0]
        elif self.audio_type == list:
            return audio_perms
        else:
            raise TypeError(f"audio_type is {self.audio_type}, expected: str, np.ndarray, or list")


"""
audioperm functions
"""
def read_audio(audio, sr = 22050, return_as_array = False):
    """ Reads audio files.
    Args:
        audio (Union[:obj:`list` of :obj:`str`, :obj:`list` of :obj:`ndarray`, ndarray, str]): A list of file paths (str) or A list of numpy array (PCM16, 32FP) 
        sr (int): Sampling rate of audio
    """
    if type(audio) == list:
        if type_nested(audio, str):
            # read all the filepaths
            audio_files = [librosa.load(f, sr = sr)[0] for f in audio] # librosa
            audio_files = [np.array(y * (1<<15), dtype=np.int16) for y in audio_files]
        elif type_nested(audio, np.ndarray):
            if audio[0].dtype == np.int16:
                audio_files = audio
            elif audio[0].dtype == np.float32:
                audio_files = [np.array(y * (1<<15), dtype=np.int16) for y in audio]
            else:
                raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")

        else:
            raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")
    elif type(audio) == str:
        audio_files = [librosa.load(audio, sr = sr)[0]] # always use arrays for consistency
        audio_files = [np.array(y * (1<<15), dtype=np.int16) for y in audio_files]
    elif type(audio) == np.ndarray:
        if audio.dtype == np.int16:
            audio_files = [audio]
        elif audio[0].dtype == np.float32:
            audio_files = [np.array(y * (1<<15), dtype=np.int16) for y in audio]
        else:
            raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")

    else:
        raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")
    # if everything is okay
    if return_as_array:
        return audio_files
    else:
        if type(audio) == str:
            return audio_files[0]
        else:
            return audio_files


def word_segments(audio_files, sr = 22050, silence_thresh = -60., min_silence_len = 5, return_as_array = False):
    """ Segments the audio files into multiple segments or words.
    TODO: Improve word segmentation. Add label wise segmentation (If given n words as labels, find n appropriate words).

    Args:
        audio_files ()
        sr (int): Sampling rate for audio files
        silence_thresh (float): Silence threshold for segmenting the audio. Same as pydub.
        min_silence_len (int): Minimum silence lenth (in ms). Same as pydub.
    
    Returns:
        Union[:obj:`list` of :obj:`list` of :obj:`ndarray`, :obj:`list` of :obj:`ndarray`]

    """
    # conversion to AudioSegment
    words = []
    type_audio_files = type(audio_files)
    if type_audio_files is not list: # single np.ndarray
        audio_files = [audio_files]
    for y in audio_files: # can use mp later
        if len(y.shape) > 1:
            y = y[:,0] # single channel

        n_max, n_min = noise_boundaries(y)

        audio_segment = pydub.AudioSegment(
            y.tobytes(), 
            frame_rate=sr,
            sample_width=y.dtype.itemsize, 
            channels=1)
        aud_segs = pydub.silence.split_on_silence(audio_segment, silence_thresh=silence_thresh, min_silence_len=5)
        
        seg_words = []
        
        last_word = -1
        c_word = np.array([], dtype=np.int16)
        # adding one silence word before and after for avoiding abrupt start and ending
        for s in aud_segs:
            s_pcm16 = np.array(s.get_array_of_samples(), dtype = np.int16)
            sig_max = s_pcm16.max()
            sig_min = s_pcm16.min() # negative max

            if (sig_max <= n_max and sig_min >= n_min): # inside noise boundaries
                # if we don't want to add noise/ silence
                if last_word == -1:
                    c_word = s_pcm16
                else:
                    c_word = np.r_[c_word, s_pcm16] # closure
                    seg_words.append(c_word)
                last_word = -1
            else:
                if last_word == -1:
                    c_word = np.r_[c_word, s_pcm16]
                else:
                    seg_words.append(c_word)
                    c_word = s_pcm16
                last_word = 1
        if last_word == 1:
            if len(c_word) > 10: # it should be longer than 10 timepoints for sure
                seg_words.append(c_word)
            
        
        words.append(seg_words)
    # if everything is okay
    if return_as_array:
        return words
    else:
        if type_audio_files == np.ndarray:
            return words[0]
        elif type_audio_files == list:
            return words
        else:
            raise TypeError("Takes an audio file (or a list of files) in numpy array format (int16, float). Type mismatch!")


def permutations(words, sr = 22050, n_permutations = 1, interm_silence = 1000, return_as_array = False):
    """Get the permutation of words.
    TODO: Use yield.

    Args:
        n_permutations (int): Number of (max) permutations to return
        interm_silence (int): Intermediate silence between words (in ms).
    Returns:
        Union[:obj:`list` of :obj:`list` of :obj:`ndarray`, :obj:`list` of :obj:`ndarray`]
    """
    type_list_of_words = False
    if type_chain(words, [list, np.ndarray]):
        words = [words]
    elif type_chain(words, [list, list, np.ndarray]):
        type_list_of_words = True
    else:
        raise TypeError("Takes a list of np.ndarray or list of list of np.ndarray. Type mismatch!")


    audio_perms = []

    for audio in words:
        # audio = list of ndarray [word1, word2, word3]
        c_audio = []
        for ind, idxs in enumerate(itertools.permutations(range(len(audio)))):
            if ind == n_permutations:
                break
            x = np.append([], [np.r_[audio[i], np.zeros(int(sr * interm_silence / 1000.))] for i in idxs])
            x = np.hstack(x).astype(np.int16)
            c_audio.append(x)
        # c_audio = np.hstack(c_audio).astype(np.int16)
        audio_perms.append(c_audio)

    if return_as_array == True:
        return audio_perms
    elif type_list_of_words == False:
        return audio_perms[0]
    else:
        return audio_perms

def segment_aud_eq(audio_segment, k):
    # k denotes, seconds * 1000
    a_segs = [audio_segment[i*k:min((i+1)*k, len(audio_segment)-1)] for i in range(len(audio_segment)//k)]
    return a_segs

def silence_remove_segment(filename, silence_thresh=-60., segment_size = 5.0, save = False, save_path = "", ret = True):
    # takes an wav/sph file/anything that librosa supports
    # removes the silence with a threshold
    # makes a list of segment of size >= segment_size (in sec.) 
    # saves the wav file or returns a numpy array 16 bit PCM
    y, sr = librosa.load(filename)
    # convert from float to uint16
    y = np.array(y * (1<<15), dtype=np.int16)
    audio_segment = pydub.AudioSegment(
        y.tobytes(), 
        frame_rate=sr,
        sample_width=y.dtype.itemsize, 
        channels=1
    )
    aud_segs = pydub.silence.split_on_silence(audio_segment, silence_thresh=silence_thresh)
    # join all
    all_seg = sum(aud_segs)
    eq_segs = segment_aud_eq(all_seg, int(segment_size * 1000)) # 1000 because, in AudioSegment 1s is 1000 points

    if save:
        # save as wav
        bn = os.path.basename(filename)
        for i, s in enumerate(eq_segs):
            s.export(f"{os.path.join(save_path, bn.split('.')[0])}_{i}.wav", format="wav")

    if ret:
        return [np.array(s.get_array_of_samples(), dtype = np.int16) for s in eq_segs]