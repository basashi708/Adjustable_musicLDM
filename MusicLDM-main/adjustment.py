import torch 
import scipy
import soundfile as sf
import librosa
import numpy as np
from matplotlib import pyplot as plt
from diffusers import MusicLDMPipeline
from diffusers import DDIMScheduler
from transformers import SpeechT5FeatureExtractor

# load the pipeline
repo_id = "ucsd-reach/musicldm"
pipe = MusicLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")



#define the prompts
wavefile = "Lyrical ballad sung by saxophone.wav"  #調整したい音声ファイルを定義する
prompt = "Lyrical ballad sung by saxophone, melodious"  #調整したいプロンプトを定義する
negative_prompt = "low quality, average quality"

# wavefile = "blue spring.mp3"  #調整したい音声ファイルを定義する
# prompt = "uptempo rock, melodious"  #調整したいプロンプトを定義する
# negative_prompt = "low quality, average quality"


# set the seed
generator = torch.Generator("cuda").manual_seed(111)

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
# sampling_rate = pipe.feature_extractor.sampling_rate
sampling_rate = 16000

feature_ext = SpeechT5FeatureExtractor(num_mel_bins=64, hop_length=10, win_length=64, fmin=0, fmax=8000)
# inputs = pipe.feature_extractor(
#             list(resampled_audio), return_tensors="pt", sampling_rate=sampling_rate
#         ).input_features.type(dtype)
#         #inputs = inputs.to(device)
#inputs = feature_ext(resampled_audio)
inputs = feature_ext._extract_mel_features(resampled_audio)
inputs = torch.from_numpy(inputs).to(torch.float16)
# inputs = torch.from_numpy(inputs)
#inputs = inputs.transpose(1,0)
inputs = torch.unsqueeze(torch.unsqueeze(inputs,0),0)
#inputs = inputs.repeat(5,1,1,1)


spec = feature_ext.power_to_db(inputs)
spec = librosa.power_to_db(inputs, ref=np.max)

spec = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=wf)

# print("feature_extractor.sampling_rate:",pipe.feature_extractor.sampling_rate)
print("type(inputs):",type(inputs))
print("inputs.shape:", inputs.shape)
print("inputs.dtype", inputs.dtype)
#print("tensor(input)", torch.from_numpy(inputs).shape)
print("type(spec):",type(spec))
print("spec.shape:",spec.shape)
print("spec.dtype:", spec.dtype)
plt.title("mel-spectrogram")
spec = spec.transpose((1,2,0))
spec = spec.squeeze(2)
print("spec.shape:",spec.shape)
#spec = librosa.display.specshow(spec, x_axis='time', y_axis='mel', sr=wf)
plt.imshow(inputs)
plt.show()
"""
encode_latents = pipe.vae.encode(inputs.to("cuda:0"))
mean = encode_latents['latent_dist'].mean
# encoder = AutoencoderKL()
# latent = encoder.encode(inputs)
print("type(latents)", type(encode_latents))
# print("latents.shape", latents.shape)

decoded_mel = pipe.vae.decode(mean.to("cuda:0")).sample
print("type(decoded_mel)", type(decoded_mel))
print("decoded_mel.shape", decoded_mel.shape)

"""
#以下今週からの進捗//////////////////////////////////////////////////////////////
"""
#タイムステップの用意
timestep =  pipe.scheduler.set_timesteps(num_inference_steps=200,device=device)
timesteps = pipe.scheduler.timesteps

batch_size = 1
num_waveforms_per_prompt = 1
num_channels_latents = pipe.unet.config.in_channels
height = sec
generator = None
prompt_embeds = None
negative_prompt_embeds = None
do_classifier_free_guidance = True

#プロンプトのエンコード
prompt_embeds = pipe._encode_prompt(
    prompt,
    device,
    num_waveforms_per_prompt,
    do_classifier_free_guidance=do_classifier_free_guidance,
    negative_prompt=negative_prompt,
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
    )

#潜在変数の用意
latents = pipe.prepare_latents(
        batch_size * num_waveforms_per_prompt,
        num_channels_latents=num_channels_latents,
        height=height,
        device=device,
        dtype = prompt_embeds.dtype,
        generator=generator,
        latents=mean,
    )

#追加の文字列用意
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0)

"""
#以下8/20週の進捗///////////////////////////////////////////////////////////////
"""

#デノイジングループ
num_inference_steps = 500
guidance_scale = 2.0
cross_attention_kwargs = None
callback = None
callback_steps = 1
num_warmup_steps = len(timesteps) - num_inference_steps * pipe.scheduler.order
with pipe.progress_bar(total=num_inference_steps) as progress_bar:
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # predict the noise residual
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            class_labels=prompt_embeds,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

        # call the callback, if provided
        if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
            progress_bar.update()
            if callback is not None and i % callback_steps == 0:
                step_idx = i // getattr(pipe.scheduler, "order", 1)
                callback(step_idx, t, latents)

pipe.maybe_free_model_hooks()

# waveform = pipe.mel_spectrogram_to_waveform(decoded_mel.to("cuda")).detach().to("cpu").numpy()
# #waveform = pipe.mel_spectrogram_to_waveform(inputs.to(device))
# #waveform_np = waveform[0].detach().cpu().numpy()
# #waveform_np = np.expand_dims(waveform_np, axis=0)
# scipy.io.wavfile.write("test7.wav", rate=16000, data=waveform[0])
# print("type(waveform)", type(waveform))
# print("waveform.shape", waveform.shape)
# print("waveform.dtype", waveform.dtype)

"""

