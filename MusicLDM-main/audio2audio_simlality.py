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

wavefiles_a = 'Lyrical ballad sung by saxophone.wav'
wavefiles_b = 'Lyrical ballad sung by saxophone.wav'

prompt = "an epic orchestral music with soaring strings, powerful brass, and dramatic percussion"
negative_prompt = "low quality, average quality"
#評価指標保存用リスト
euclidean_dist = []
cosine_sim = []

data_a, wf = librosa.load(wavefiles_a, sr=16000)
data_b, wf = librosa.load(wavefiles_b, sr=16000)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#プロンプトのエンコード
num_waveforms_per_prompt = 1
do_classifier_free_guidance = True
prompt_embeds = None
negative_prompt_embeds = None
prompt_embeds = pipe._encode_prompt(
        prompt,
        device,
        num_waveforms_per_prompt,
        do_classifier_free_guidance=do_classifier_free_guidance,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        )
# print("prompt",prompt_embeds.shape)

data_a = torch.from_numpy(data_a)
data_b = torch.from_numpy(data_b)
encoded_audio_a = pipe.score_waveforms_(
    text = prompt,
    audio = data_a,
    num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype
)
encoded_audio_b = pipe.score_waveforms_(
    text = prompt,
    audio = data_b,
    num_waveforms_per_prompt=num_waveforms_per_prompt,
                device=device,
                dtype=prompt_embeds.dtype
)
#評価指標の算出
culc_dist = dist(encoded_audio_a.audio_embeds[0],encoded_audio_b.audio_embeds[0])
euclidean_dist.append(culc_dist)
print("Euclidean_dist",culc_dist)

culc_sim = metrics.pairwise.cosine_similarity(
    encoded_audio_a.audio_embeds[0].reshape(1,-1).cpu().detach().numpy().copy(),
    encoded_audio_b.audio_embeds[0].reshape(1,-1).cpu().detach().numpy().copy()
)
cosine_sim.append(culc_sim)
print("cosine_sim", culc_sim)













