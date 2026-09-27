# Third-party notices

Voice Clone Studio bundles the following models and libraries. Each remains under its own license;
the full license texts are included with the respective packages inside the `runtime` folder of the
installed app (`Lib\site-packages\<package>-<version>.dist-info`) and on the pages linked below.

## Models

| Model | Purpose | License | Source |
|---|---|---|---|
| Chatterbox Multilingual | Text to speech, voice cloning | MIT | https://huggingface.co/ResembleAI/chatterbox |
| whisper-hinglish-preview (stored in half precision) | Hindi / Hinglish speech recognition | Apache-2.0 | https://huggingface.co/Trelis/whisper-hinglish-preview |
| Whisper large-v3-turbo (CTranslate2 conversion) | Speech recognition, language identification | MIT | https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo |
| spkrec-ecapa-voxceleb | Speaker verification | Apache-2.0 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| DeepFilterNet3 | Noise reduction | MIT / Apache-2.0 | https://github.com/Rikorose/DeepFilterNet |
| AudioSeal | Audio watermark | MIT | https://github.com/facebookresearch/audioseal |
| Perth | Audio watermark | MIT | https://github.com/resemble-ai/Perth |
| Silero VAD | Voice activity detection | MIT | https://github.com/snakers4/silero-vad |
| spacy-pkuseg data | Tokenizer data required by Chatterbox | MIT | https://github.com/explosion/spacy-pkuseg |

whisper-hinglish-preview builds on `ARTPARK-IISc/whisper-large-v3-vaani-hindi` (Apache-2.0) and
`openai/whisper-large-v3` (MIT).

## Main libraries

| Library | License |
|---|---|
| Python (python-build-standalone) | PSF-2.0 |
| PyTorch, torchaudio | BSD-3-Clause |
| NVIDIA CUDA runtime libraries (redistributed with PyTorch) | NVIDIA Software License |
| chatterbox-tts | MIT |
| transformers, peft, huggingface_hub, safetensors | Apache-2.0 |
| faster-whisper, CTranslate2 | MIT |
| SpeechBrain | Apache-2.0 |
| FastAPI, Uvicorn | MIT, BSD-3-Clause |
| NumPy, SciPy, librosa | BSD-3-Clause, ISC |
| soundfile (libsndfile) | BSD-3-Clause (LGPL-2.1) |
| python-soxr (libsoxr) | LGPL-2.1 |
| lameenc (LAME) | LGPL-3.0 |
| pyloudnorm, audiotsm, num2words | MIT, MIT, LGPL-2.1 |
| wordfreq (word frequency data) | Apache-2.0 (data: CC BY-SA 4.0) |
| c2pa-python | MIT / Apache-2.0 |
| cryptography | Apache-2.0 / BSD-3-Clause |
| React, Vite | MIT |

The LGPL libraries are used unmodified as separate, replaceable modules inside the `runtime` folder.
