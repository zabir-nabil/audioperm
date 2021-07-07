import pytest

import numpy as np
from audioperm import AudioPerm

def test_init_not_list_str_ndarray():
    """
    Testing random type errors.
    """
    with pytest.raises(TypeError):
        ap = AudioPerm(12)
    with pytest.raises(TypeError):
        ap = AudioPerm(34.435345)
    with pytest.raises(TypeError):
        ap = AudioPerm([12, 34])
    with pytest.raises(TypeError):
        ap = AudioPerm(["any str", np.array([1,2,3])])
    with pytest.raises(TypeError):
        ap = AudioPerm([]) 
    with pytest.raises(TypeError):
        ap = AudioPerm(["any str", []]) 

def test_init_audio_load():
    """
    Audio loading with librosa for multiple formats.
    """
    ap = AudioPerm(["tests/test.m4a"])
    ap = AudioPerm(["tests/test.mp3"])
    ap = AudioPerm(["tests/test.wav"])
    ap = AudioPerm(["tests/test.aac"])
    ap = AudioPerm(["tests/test.flac"])
    ap = AudioPerm(["tests/test.sph"])


