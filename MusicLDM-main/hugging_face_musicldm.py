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
#NUMBA_DISABLE_INTEL_SVML=1

#実行前に指定するパラメータ
"""
gen=-1: プロンプトを基にノイズから音楽を生成 その際はプロンプトを設定してね
gen=0 : プロンプトから生成した音楽を用いた再度調整音楽の生成・評価
gen=1 : プロンプトから生成した音楽を用いた評価
gen=2 : 再度調整後の音源をもちいた評価
number: テスト音声を作る際の添え字 随時更新
"""
gen = 0
number = 841

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

test_waves = []
for k in range(len(wavefiles)):
#for k in range(100):
    test_wave = "test"+str(number+k)+".wav"
    print(test_wave)
    test_waves.append(test_wave)

#評価指標保存用リスト
euclidean_dist = []
cosine_sim = []


#for i in range(len(wavefiles)):
for i in range(2):
    """
    if gen == 0 or gen==1:
        wavefile = wavefiles[i]  #調整したい音声ファイルを定義する
    if gen == 2:
       wavefile = test_waves[i]
    """
    #prompt = "an epic orchestral music with soaring strings, powerful brass, and dramatic percussion, building tension and emotion throughout the arrangement"
    prompt = "upbeat, melodious"
    negative_prompt = "low quality, average quality"


 
    #wavefile = 'Royal Film Music Ochestra, catchy melody, simple melody, Wolfgang Amadeus Mozart.wav'
    #wavefile = 'Piano ballad remembering fresh youth, melodious, catchy melody.wav'
    wavefile = 'Lyrical ballad sung by saxophone.wav'
    if gen != -1:
        #0. 前処理
        #data, wf = sf.read(wavefile)
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
        inputs = torch.from_numpy(inputs).to(torch.float16)
        inputs = torch.unsqueeze(torch.unsqueeze(inputs,0),0)

        # eval_inputs = torch.from_numpy(inputs)




        # fig, ax = plt.subplots()
        # spec = librosa.display.specshow(inputs.T, x_axis='time', y_axis='mel', sr=16000, fmax=8000, fmin=0, ax=ax, hop_length=160, win_length=1024, n_fft=1024)
        # fig.colorbar(spec, ax=ax, format='%+2.0f dB')

        # print("feature_extractor.sampling_rate:",pipe.feature_extractor.sampling_rate)
        print("type(inputs):",type(inputs))
        print("inputs.shape:", inputs.shape)
        print("inputs.dtype", inputs.dtype)
        # #print("tensor(input)", torch.from_numpy(inputs).shape)
        # print("type(spec):",type(spec))
        # print("spec.shape:",spec.shape)
        # print("spec.dtype:", spec.dtype)
        #plt.show()


        encode_latents = pipe.vae.encode(inputs.to("cuda:0"))
        mean = encode_latents['latent_dist'].mean
        sigma_x = encode_latents['xlatent_dist'].var
        # co_sigma_x = torch.cov(sigma_x)

        fla_mean = torch.flatten(mean)
        print("mean.shape", fla_mean.shape)
        # print("sigma_x.shape", co_sigma_x.shape)


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

        data = torch.from_numpy(data)
        encoded_audio = pipe.score_waveforms_(
            text = prompt,
            audio = data,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
                        device=device,
                        dtype=prompt_embeds.dtype
        )

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
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=998,
            audio_length_in_s=audio_length_in_s,
            num_waveforms_per_prompt=1,
            guidance_scale = 4.0,
            latents = latents
        ).audios

        # save the best audio sample (index 0) as a .wav file
        # filename = prompt + ".wav"
        name = "test"+str(number+i)+".wav"
        scipy.io.wavfile.write(name, rate=16000, data=audio[0])

#評価指標のまとめ表示
if gen != -1:
    print("\neuclidean_dist:\n", euclidean_dist)
    print("\nave_list:\n", np.mean(euclidean_dist))
    print("\ncosine_sim:\n", cosine_sim)
    print("\nave_sim\n", np.mean(cosine_sim))
