import numpy as np
import librosa

from audioperm.utils import type_nested

class AudioPerm:
    """
    The main class for audioperm. Takes an audio file (or a batch of files) path or numpy array (int16, float). Internal audio representation is pcm 16 (not same as librosa default).
    """
    def __init__(self, audio, sr = 22050, **kwargs):
        """
        Args:
            audio (Union[:obj:`list` of :obj:`str`, :obj:`list` of :obj:`ndarray`, ndarray, str]): A list of file paths (str) or A list of numpy array (PCM16, 32FP) 
        """
        if type(audio) == list:
            if type_nested(audio, str):
                # read all the filepaths
                audio_files = [librosa.load(f, sr = sr)[0] for f in audio] #
            elif type_nested(audio, np.ndarray):
                pass
            else:
                raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")
        elif type(audio) == str:
            pass
        elif type(audio) == np.ndarray:
            pass
        else:
            raise TypeError("Takes an audio file (or a list of files) path or numpy array (int16, float). Type mismatch!")



