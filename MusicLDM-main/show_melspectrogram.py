import scipy
import torch
import torchaudio
import numpy as np
from diffusers import MusicLDMPipeline

import librosa
from transformers import SpeechT5FeatureExtractor
import matplotlib.pyplot as plt
from math import dist
from sklearn import metrics


generator = torch.Generator("cuda").manual_seed(111)
# load the pipeline
repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


#wavefile = 'Lyrical ballad sung by saxophone.wav'
#wavefile = "test480.wav"
wavefile = "test730.wav"
#wavefile = "Piano ballad remembering fresh youth, melodious, catchy melody.wav"

data, wf = librosa.load(wavefile, sr=16000)
# data information
print("Sampling rate:", wf)
print("Frame num:", data.shape[0])
print("Sec:", data.shape[0] / wf)
print("Numpy dtype:", data.dtype)
sec = data.shape[0] / wf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

  #1. 音声波形を入力してノイズに変換
resampled_audio = librosa.resample(
            data, orig_sr=wf, target_sr=16000
        )
#resampled_audio = resampled_audio.astype("float16")
print("type(resampled_audio):",type(resampled_audio))
print("resampled_audio.shape:",resampled_audio.shape)
print("resampled_audio.dtype:",resampled_audio.dtype)
sampling_rate = 16000

feature_ext = SpeechT5FeatureExtractor(num_mel_bins=64, hop_length=10, win_length=64, fmin=0, fmax=8000)

inputs = feature_ext._extract_mel_features(resampled_audio)
print("inputs: ", inputs.shape)
print("type(inputs)", type(inputs))

fig, ax = plt.subplots()
spec = librosa.display.specshow(inputs.T, x_axis="time", y_axis="mel", sr=16000, fmax=8000, fmin=0, ax=ax, hop_length=160, win_length=1024, n_fft=1024, htk=True)
print("spec:", spec)
plt.ylabel("Frequency [Hz]",fontsize = 25)
plt.xlabel("Time [s]",fontsize = 25)
plt.rcParams["font.size"] = 25
ax.tick_params(labelsize = 25)
fig.colorbar(spec, ax=ax, format='%+2.0f dB')
plt.show()