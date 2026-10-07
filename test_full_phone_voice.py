from pocket_tts import TTSModel
import wave
import numpy as np


# Load Pocket TTS
print("Loading model...")
model = TTSModel.load_model()

# Create voice profile from the full phone recording
print("Creating voice profile from full phone recording...")
voice_state = model.get_state_for_audio_prompt(
    "my_voice_phone_long.wav"
)

print("Voice profile created.")

# Text to synthesize
text = """
Hello. This is a test of my voice generated locally using Pocket TTS.

Technology does not have to be complicated to be useful.

Good software should save people time, reduce repetitive work,
and make everyday tasks easier.
"""

# Generate speech
print("Generating speech...")

audio = model.generate_audio(
    voice_state,
    text,
    frames_after_eos=2,
    copy_state=True,
)

# Convert PyTorch tensor to NumPy
audio_np = audio.detach().cpu().numpy()

# Convert float audio [-1, 1] to 16-bit PCM
audio_int16 = np.clip(audio_np, -1, 1)
audio_int16 = (audio_int16 * 32767).astype(np.int16)

# Save WAV file
output_file = "my_voice_phone_full_test.wav"

with wave.open(output_file, "wb") as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(model.sample_rate)
    f.writeframes(audio_int16.tobytes())

print()
print("Generated:", output_file)
print(
    "Duration:",
    round(len(audio_np) / model.sample_rate, 1),
    "seconds"
)
