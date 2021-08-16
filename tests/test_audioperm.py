import pytest
import os

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

def test_read_audio():
    """
    Reading audio files from filepath (str), array of filepath (list of str), numpy array (np.ndarray), array of numpy array (list of np.ndarray)
    """
    from audioperm import read_audio
    a = read_audio("tests/test.mp3")
    assert type(a) == np.ndarray
    assert len(a) > 10000

    a = read_audio(["tests/test.mp3"])
    assert type(a) == list
    assert len(a) == 1
    assert type(a[0]) == np.ndarray

    a = read_audio(["tests/test.flac", "tests/test.sph"])
    assert type(a) == list
    assert len(a) == 2
    assert type(a[0]) == np.ndarray

def test_word_segments():
    ap = AudioPerm("tests/i_love_cats.m4a")
    out = ap.word_segments()

    assert len(out) == 3 # 3 words
    assert(type(out[0])) == np.ndarray
    
    ap = AudioPerm("tests/test.wav")
    out = ap.word_segments()

    assert len(out) == 1 # 1 word
    assert(type(out[0])) == np.ndarray

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    out = ap.word_segments()

    assert len(out) == 2 # 2 audio files
    assert len(out[0]) == 1 # 1 word for first audio
    assert len(out[1]) == 3 # 1 word for first audio

    assert type(out[0][0]) == np.ndarray
    assert type(out[1][2]) == np.ndarray 

def test_word_segments_f():
    from audioperm import read_audio, word_segments
    ap = read_audio("tests/i_love_cats.m4a")
    print(ap)
    print(type(ap))
    out = word_segments(ap)

    assert len(out) == 3 # 3 words
    
    ap = read_audio("tests/test.wav")
    out = word_segments(ap)

    assert len(out) == 1 # 1 word

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    out = ap.word_segments()

    assert len(out) == 2 # 2 audio files
    assert len(out[0]) == 1 # 1 word for first audio
    assert len(out[1]) == 3 # 1 word for first audio

def test_permute():
    ap = AudioPerm("tests/i_love_cats.m4a")
    ap.word_segments(return_words = False)
    perms = ap.permutations(4)
    print(len(perms))
    print(perms[0].shape)
    print(perms[1].shape)

    assert len(perms) == 4 # 4 permutations
    assert type(perms[0]) == np.ndarray
    
    ap = AudioPerm("tests/test.wav")
    ap.word_segments(return_words = False)
    perms = ap.permutations(4)

    assert len(perms) == 1 # 1 permutation, even though I requested 4

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    ap.word_segments(return_words = False)
    perms = ap.permutations(5)

    assert len(perms) == 2 # 2 audio files
    assert len(perms[0]) == 1 # 1 max perm possible
    assert len(perms[1]) == 5 # 5 perms

def test_type_chain():
    from audioperm.utils import type_chain 

    assert type_chain([], [list]) == True
    assert type_chain([1,2,3], [list]) == True
    assert type_chain([1,2,3], [list, int]) == True
    assert type_chain([[1,2],[3]], [list, list]) == True
    assert type_chain([[1,2],[3]], [list, list, list]) == False
    assert type_chain([[np.array([1,2,3])], [np.array([1,2,3])], [np.array([1,2,3])]], [list, list, np.ndarray])
    assert type_chain([np.array([1,2,3]), np.array([1,2,3])], [list, np.ndarray, np.int64])

def test_permute_f():
    from audioperm import read_audio, word_segments, permutations
    ap = read_audio("tests/i_love_cats.m4a")
    out = word_segments(ap)
    perms = permutations(out, n_permutations = 4)
    
    assert len(perms) == 4 # 4 permutations
    assert type(perms[0]) == np.ndarray

    ap = read_audio("tests/test.wav")
    out = word_segments(ap)
    perms = permutations(out, n_permutations = 4)
    assert len(perms) == 1 # 1 permutation, even though I requested 4

    ap = AudioPerm(["tests/test.sph", "tests/i_love_cats.m4a"])
    out = ap.word_segments()
    perms = permutations(out, n_permutations = 5)

    assert len(perms) == 2 # 2 audio files
    assert len(perms[0]) == 1 # 1 max perm possible
    assert len(perms[1]) == 5 # 5 perms

def test_fixed_len_segments():
    from audioperm import fixed_len_segments
    out = fixed_len_segments("tests/bangla_demo.wav", return_segments = False, save_path = "tests/fls_out", save = True, segment_size = 0.5)
    assert os.path.isfile("tests/fls_out/bangla_demo_0.wav") == True
    assert out == None
    
    out = fixed_len_segments("tests/bangla_demo.wav", return_segments = True, save = False, segment_size = 0.5)
    assert len(out) == 10
    assert(type(out[0])) == np.ndarray