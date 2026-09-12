import warnings
import time
import queue
import threading
import numpy as np
import soundcard as sc
from faster_whisper import WhisperModel

# Desativa avisos visuais do soundcard no terminal
warnings.filterwarnings("ignore", category=sc.SoundcardRuntimeWarning)

print("Carregando modelo Whisper...")
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
print("Modelo pronto!")

# Dispositivo de áudio (Loopback)
default_speaker = sc.default_speaker()
loopback_mic = sc.get_microphone(id=str(default_speaker.name), include_loopback=True)

SAMPLE_RATE = 16000
BLOCK_DURATION = 2  # Blocos curtos de 2s para menor atraso

audio_queue = queue.Queue()

def audio_recorder_thread():
    """Captura o áudio continuamente em segundo plano"""
    with loopback_mic.recorder(samplerate=SAMPLE_RATE) as recorder:
        while True:
            data = recorder.record(numframes=SAMPLE_RATE * BLOCK_DURATION)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            audio_queue.put(data)

# Inicia a gravação paralela
threading.Thread(target=audio_recorder_thread, daemon=True).start()

print("\n--- TRANSCRIÇÃO EM TEMPO REAL INICIADA (Pressione Ctrl+C para encerrar) ---\n")

try:
    while True:
        if not audio_queue.empty():
            data = audio_queue.get()
            
            # Se for silêncio, descarta
            if np.max(np.abs(data)) < 0.01:
                continue

            segments, _ = model.transcribe(
                data, 
                task="translate", 
                language="en", 
                beam_size=1, 
                best_of=1
            )

            for segment in segments:
                if segment.text.strip():
                    print(f"Legenda: {segment.text}")
        else:
            time.sleep(0.05)

except KeyboardInterrupt:
    print("\nPrograma encerrado.")