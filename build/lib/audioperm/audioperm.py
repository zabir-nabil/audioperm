import itertools

import numpy as np
import librosa
import pydub
import soundfile as sf

from audioperm.utils import type_nested, noise_boundaries

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
        self.audio_files = audio_files
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
        # conversion to AudioSegment
        words = []
        
        for y in self.audio_files: # can use mp later
            if len(y.shape) > 1:
                y = y[:,0] # single channel

            n_max, n_min = noise_boundaries(y)

            audio_segment = pydub.AudioSegment(
                y.tobytes(), 
                frame_rate=self.sr,
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
                if len(c_word) > 10: # it should be longer than 10 samples for sure
                    seg_words.append(c_word)
                
            
            words.append(seg_words)
        
        self.words = words # update
        
        if return_words == True:
            if len(self.audio_files) == 1:
                return words[0]
            else:
                return words

    def permute(self, n_permutations = 1, interm_silence = 1000):
        """Get the permutation of words.
        TODO: Use yield.

        Args:
            n_permutations (int): Number of (max) permutations to return
            interm_silence (int): Intermediate silence between words (in ms).
        Returns:
            Union[:obj:`list` of :obj:`list` of :obj:`ndarray`, :obj:`list` of :obj:`ndarray`]
        """
        audio_perms = []

        for audio in self.words:
            # audio = list of ndarray [word1, word2, word3]
            c_audio = []
            for ind, idxs in enumerate(itertools.permutations(range(len(audio)))):
                if ind == n_permutations:
                    break
                x = np.append([], [np.r_[audio[i], np.zeros(int(self.sr * interm_silence / 1000.))] for i in idxs])
                c_audio.append(x)
            audio_perms.append(c_audio)

        if len(self.audio_files) == 1:
            return audio_perms[0]
        else:
            return audio_perms




