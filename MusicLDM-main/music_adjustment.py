# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import scipy
import torch
import numpy as np
from diffusers import MusicLDMPipeline
import librosa
from transformers import SpeechT5FeatureExtractor
from math import dist
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import ClapModel, ClapProcessor
clap  = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device)
proc  = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")


#実行前に指定するパラメータ
"""
gen=-1: プロンプトを基にノイズから音楽を生成 その際はプロンプトを設定してね
gen=0 : プロンプトから生成した音楽を用いた再度調整音楽の生成・評価
gen=1 : プロンプトから生成した音楽を用いた評価
gen=2 : 再度調整後の音源をもちいた評価
number: テスト音声を作る際の添え字 随,mm更新
"""
gen = 0
number = 853

# set the seed
#torch.manual_seed(111)
generator = torch.Generator("cuda").manual_seed(111)


# load the pipeline
repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#define the prompts
wavefiles = ['Lyrical ballad sung by saxophone.wav',
             #'Royal Film Music Ochestra, catchy melody, simple melody, Wolfgang Amadeus Mozart.wav',
             #'techno.wav',
             #'Piano ballad remembering fresh youth, melodious, catchy melody.wav'
             ]
#評価指標保存用リスト
euclidean_dist = []
cosine_sim = []

for i in range(2):
 
   
    prompt = ""
    negative_prompt = ""
    guidance_scale = 4
 
    # wavefile = 'Lyrical ballad sung by saxophone.wav'
    # ad_wavefile = "guitar4.wav"

    wavefile = 'techno.wav'
    ad_wavefile = 'Lyrical ballad sung by saxophone.wav'

    if gen != -1:
        #以下は元音源に対する処理
        #0. 前処理
        data, wf = librosa.load(wavefile, sr=16000)
        # data information
        print("Sampling rate:", wf)
        print("Frame num:", data.shape[0])
        print("Sec:", data.shape[0] / wf)
        print("Numpy dtype:", data.dtype)
        sec = data.shape[0] / wf
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
        inputs = torch.from_numpy(inputs).to(torch.float16)
        inputs = torch.unsqueeze(torch.unsqueeze(inputs,0),0)
        print("type(inputs):",type(inputs))
        print("inputs.shape:", inputs.shape)
        print("inputs.dtype", inputs.dtype)
        encode_latents = pipe.vae.encode(inputs.to("cuda:0"))
        mean = encode_latents['latent_dist'].mean
        sigma_x = encode_latents['latent_dist'].var
        fla_mean = torch.flatten(mean)
        print("mean.shape", fla_mean.shape)
        #上記は元音源に対する処理

        #以下は条件用楽器音に対する処理
        #0. 前処理
        data_b, wf_b = librosa.load(ad_wavefile, sr=16000)
        # data information
        print("Sampling rate:", wf_b)
        print("Frame num:", data_b.shape[0])
        print("Sec:", data_b.shape[0] / wf_b)
        print("Numpy dtype:", data_b.dtype)
        sec = data_b.shape[0] / wf_b
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16
        #1. 音声波形を入力してノイズに変換
        resampled_audio_b = librosa.resample(
                    data_b, orig_sr=wf_b, target_sr=16000
                )
        #resampled_audio = resampled_audio.astype("float16")
        print("type(resampled_audio):",type(resampled_audio))
        print("resampled_audio.shape:",resampled_audio.shape)
        print("resampled_audio.dtype:",resampled_audio.dtype)

        inputs_b = proc(audios=[resampled_audio_b], sampling_rate = 48000, return_tensors="pt").to(device)

        with torch.no_grad():
            audio_emb_b = clap.get_audio_features(**inputs_b).to(torch.float16)
        
        neg_emb = torch.zeros_like(audio_emb_b)
        prompt_embeds_b = audio_emb_b
   
        #上記は条件用楽器音に対する処理




        #プロンプトのエンコード
        num_waveforms_per_prompt = 1
        do_classifier_free_guidance = True
        prompt_embeds = prompt_embeds_b
        negative_prompt_embeds = neg_emb
        
        
        
        prompt_embeds = pipe._encode_prompt(
            prompt,
            device,
            num_waveforms_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            #ここのprompt_embedsにエンコードした楽器音を入れる
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            )
        # print("prompt",prompt_embeds.shape)

        """
        data = torch.from_numpy(data)
        encoded_audio = pipe.score_waveforms_(
            text = prompt,
            audio = data,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
                        device=device,
                        dtype=torch.float16
        )
        """
        """
        #評価指標の算出
        culc_dist = dist(encoded_audio.text_embeds[0],encoded_audio.audio_embeds[0])
        euclidean_dist.append(culc_dist)
        print("Euclidean_dist",culc_dist)

        culc_sim = metrics.pairwise.cosine_similarity(
            encoded_audio.text_embeds[0].reshape(1,-1).cpu().detach().numpy().copy(),
            encoded_audio.audio_embeds[0].reshape(1,-1).cpu().detach().numpy().copy()
        )
        cosine_sim.append(culc_sim)
        print("cosine_sim", culc_sim)

        """
    if gen == 0:
        #print("type(latents)", type(encode_latents))
        # print("latents.shape", latents.shape)
        time = 600
        alpha = pipe.scheduler.alphas_cumprod[time]
        alpha = alpha.to(device)
        print(alpha)
        noised_mean = ((alpha)**0.5) * mean + (1-alpha)**0.5 * torch.randn(mean.shape,device=device,dtype=torch.float16)
        print(noised_mean.dtype)

    if gen == -1 or gen ==0:
        if gen==-1:
            latents = None
            audio_length_in_s = 30
        else:
            latents = noised_mean
            audio_length_in_s = sec
        
        # run the generation
        audio = pipe(
            prompt=None,
            prompt_embeds = prompt_embeds,
            negative_prompt=None,
            num_inference_steps=998,
            audio_length_in_s=audio_length_in_s,
            num_waveforms_per_prompt=1,
            guidance_scale = guidance_scale,
            latents = latents
        ).audios

        # save the best audio sample (index 0) as a .wav file
        # filename = prompt + ".wav"
        name = "test"+str(number+i)+".wav"
        scipy.io.wavfile.write(name, rate=16000, data=audio[0])

"""
#評価指標のまとめ表示
if gen != -1:
    print("\neuclidean_dist:\n", euclidean_dist)
    print("\nave_list:\n", np.mean(euclidean_dist))
    print("\ncosine_sim:\n", cosine_sim)
    print("\nave_sim\n", np.mean(cosine_sim))
"""