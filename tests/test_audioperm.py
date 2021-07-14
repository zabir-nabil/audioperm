import pytest

import numpy as np
from audioperm import AudioPerm

"""
testing 
run: python -m pytest tests/
"""

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

def test_init_audio_load_dtype():
    """
    Audio loading with librosa for multiple structures.
    """
    ap = AudioPerm(["tests/test.m4a"])
    ap = AudioPerm("tests/test.mp3")
    ap = AudioPerm("tests/test.aac")
    ap = AudioPerm(["tests/test.flac", "tests/test.sph"])
    ap = AudioPerm(np.array([123, 1400, -1599, 8000], dtype=np.int16))

    ap = AudioPerm(np.array([.123, .1400, 0., .8000], dtype=np.float32))
    # error cases
    with pytest.raises(TypeError):
        ap = AudioPerm([1, 2, 3, 5, 6, 7, 8, 9])
    with pytest.raises(TypeError):
        ap = AudioPerm(np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype = np.uint8))
    with pytest.raises(TypeError):
        ap = AudioPerm(np.array([1, 2, 3, 5, 6, 7, 8, 9], dtype = np.double))
    with pytest.raises(TypeError):
        ap = AudioPerm(np.array([1., 2., 600., 700.], dtype=np.float))

def test_word_segments():
    ap = AudioPerm("tests/i_love_cats.m4a")
    out = ap.word_segments()

    assert len(out) == 3 # 3 words
    
    ap = AudioPerm("tests/test.wav")
    out = ap.word_segments()

    assert len(out) == 1 # 1 word

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    out = ap.word_segments()

    assert len(out) == 2 # 2 audio files
    assert len(out[0]) == 1 # 1 word for first audio
    assert len(out[1]) == 3 # 1 word for first audio

def test_permute():
    ap = AudioPerm("tests/i_love_cats.m4a")
    ap.word_segments(return_words = False)
    perms = ap.permute(4)

    assert len(perms) == 4 # 4 permutations
    
    ap = AudioPerm("tests/test.wav")
    ap.word_segments(return_words = False)
    perms = ap.permute(4)

    assert len(perms) == 1 # 1 permutation, even though I requested 4

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    ap.word_segments(return_words = False)
    perms = ap.permute(5)

    assert len(perms) == 2 # 2 audio files
    assert len(perms[0]) == 1 # 1 max perm possible
    assert len(perms[1]) == 5 # 5 perms


