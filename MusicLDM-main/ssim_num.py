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
from skimage.metrics import structural_similarity as ssim

generator = torch.Generator("cuda").manual_seed(111)
# load the pipeline
repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")


wavefile_a = 'Lyrical ballad sung by saxophone.wav'
#wavefile_b = "test480.wav"
wavefile_b = "test66.wav"

#1つ目の音声ファイルの処理
data, wf = librosa.load(wavefile_a, sr=16000)
sec = data.shape[0] / wf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

  #1. 音声波形を入力してノイズに変換
resampled_audio = librosa.resample(
            data, orig_sr=wf, target_sr=16000
        )
sampling_rate = 16000
feature_ext = SpeechT5FeatureExtractor(num_mel_bins=64, hop_length=10, win_length=64, fmin=0, fmax=8000)
inputs_a = feature_ext._extract_mel_features(resampled_audio)
print(np.max(inputs_a))
# fla_inputs_a = torch.flatten(inputs)
#print("type(inputs)",type(inputs))
#print("inputs.shape",inputs.shape)

#2つ目の音声ファイルの処理
data, wf = librosa.load(wavefile_b, sr=16000)
sec = data.shape[0] / wf
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

  #1. 音声波形を入力してノイズに変換
resampled_audio = librosa.resample(
            data, orig_sr=wf, target_sr=16000
        )
sampling_rate = 16000
feature_ext = SpeechT5FeatureExtractor(num_mel_bins=64, hop_length=10, win_length=64, fmin=0, fmax=8000)
inputs_b = feature_ext._extract_mel_features(resampled_audio)
print(np.max(inputs_b))
#print("type(inputs)",type(inputs))
#print("inputs.shape",inputs.shape)
#fla_inputs_b = torch.flatten(inputs)
if np.max(inputs_a) >= np.max(inputs_b):
    max_inputs = np.max(inputs_a)
else: max_inputs = inputs_b
#SSIMの算出
ssim_value, ssim_map = ssim(inputs_a, inputs_b, full=True,data_range=max_inputs)

# 結果の表示
print(f"SSIM: {ssim_value:.4f}")