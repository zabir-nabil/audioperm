<p align="center">
  <a href="#"><img src="docs/images/logo.png" alt="audioperm"></a>
</p>
<p align="center">
    <em>Audioperm, a python library for generating different permutations of audible segments from audio files.</em>
</p>
<p align="center">
<a href="https://pypi.org/project/audioperm/" target="_blank">
    <img src="https://img.shields.io/pypi/v/audioperm?color=%2334D058&label=pypi%20package" alt="Package version">
</a>
<a href="https://colab.research.google.com/github/zabir-nabil/audioperm/blob/main/notebooks/audioperm_demo.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
</p>

---
### Audioperm
A python library for generating different permutations of audible segments from audio files. 

#### Use:

* Silence Removal from Audio
* Audio / Speech augmentation
* Word segmentation
* Word level permutation generation
* Add new synthetic data for deep learning
* Speaker recognition, Speaker verification, Audio classification, Audio fingerprinting


**Documentation**: <a href="https://github.com/zabir-nabil/audioperm/docs" target="_blank">https://github.com/zabir-nabil/audioperm/docs</a>

**Source Code**: <a href="https://github.com/zabir-nabil/audioperm" target="_blank">https://github.com/zabir-nabil/audioperm</a>

---

```pyhon
from audioperm import AudioPerm
from audioperm.utils import save_audio

ap = AudioPerm("i_love_cats.m4a")
label = "i love cats"

words = ap.word_segments()
label_words = label.split()

for i, w in enumerate(words):
  save_audio(w, label_words[i] + ".wav")
```

```
cats.wav  i_love_cats.m4a  i_love_cats.m4a.1  i.wav  love.wav
```

<audio src="notebooks/cats.wav"></audio>



---