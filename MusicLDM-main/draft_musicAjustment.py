# musicldm_custom_condition.py (variables-in-code version)
"""
Condition MusicLDM with an instrument‑audio CLAP embedding **without** any command‑line
arguments. 変更したい値はファイル冒頭の定数を書き換えるだけで OK です。

* `SRC_FILE`    … 元音源 (16 kHz)
* `COND_FILE`   … 条件に使う楽器音 (48 kHz)
* `GUIDANCE`    … classifier‑free guidance 強度 (>1 で有効)
* `NOISE_T`     … latent に掛けるノイズ時刻 (浅いほど原曲の構造を残す)
* `NUM_STEPS`   … 拡散ステップ数
* `OUTPUT_FILE` … 出力 WAV
"""

import pathlib
import librosa
import scipy.io.wavfile as wavfile
import torch
from diffusers import MusicLDMPipeline
from transformers import ClapModel, ClapProcessor

# ──────────────────────────────────────────────
# 変数はここだけ書き換えれば動作をカスタムできる
# ──────────────────────────────────────────────
SRC_FILE     = "Lyrical ballad sung by saxophone.wav"  # 16 kHz
COND_FILE    = "piano2.wav"                            # 48 kHz
GUIDANCE     = 2.0
NOISE_T      = 400
NUM_STEPS    = 400
OUTPUT_FILE  = "output.wav"
SEED         = 111
# ──────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float16 if DEVICE.type == "cuda" else torch.float32

torch.manual_seed(SEED)

print("Loading models …")
pipe = MusicLDMPipeline.from_pretrained("ucsd-reach/musicldm", torch_dtype=DTYPE).to(DEVICE)
clap = ClapModel.from_pretrained("laion/clap-htsat-unfused", torch_dtype=DTYPE).to(DEVICE)
proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
pipe.enable_model_cpu_offload()

# 1. source → latent
src_wave, _ = librosa.load(SRC_FILE, sr=16_000)
mel = pipe.feature_extractor([src_wave], sr=16_000, return_tensors="pt").input_features.to(DEVICE, DTYPE)
latent = pipe.vae.encode(mel).latent_dist.mean
alpha  = pipe.scheduler.alphas_cumprod[NOISE_T].to(DEVICE, DTYPE)
latent_noised = (alpha**0.5)*latent + (1-alpha)**0.5*torch.randn_like(latent)

# 2. condition → CLAP embedding
cond_wave, _ = librosa.load(COND_FILE, sr=48_000)
inputs = proc(audios=[cond_wave], sampling_rate=48_000, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    pos_emb = clap.get_audio_features(**inputs).to(DTYPE)   # 1×512
neg_emb = torch.zeros_like(pos_emb)                         # 1×512

prompt_embeds = pipe._encode_prompt(
    prompt=None,
    device=DEVICE,
    num_waveforms_per_prompt=1,
    do_classifier_free_guidance=(GUIDANCE > 1.0),
    negative_prompt=None,
    prompt_embeds=pos_emb,
    negative_prompt_embeds=neg_emb,
)

# 3. generate
print("Running diffusion …")
audio = pipe(
    prompt=None,
    prompt_embeds=prompt_embeds,
    guidance_scale=GUIDANCE,
    num_inference_steps=NUM_STEPS,
    latents=latent_noised,
    audio_length_in_s=len(src_wave)/16_000,
).audios[0]

# 4. save
pathlib.Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
wavfile.write(OUTPUT_FILE, 16_000, audio)
print("Saved", OUTPUT_FILE)
