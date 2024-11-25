import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fftpack import fft
import tkinter as tk
from matplotlib.animation import FuncAnimation
import time

# Parameters
fs = 16000  # Lagere sample rate om overflow te voorkomen
duration = 0.2  # Buffertijd in seconden (grotere buffer)
buffer_size = int(fs * duration)  # Buffergrootte
device_id = 2  # Specifiek apparaat-ID (USB MICROPHONE)
audio_buffer = np.zeros(buffer_size)  # Initieer lege buffer

# Callback voor live audio
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Status: {status}")  # Druk foutmeldingen af
    if len(indata[:, 0]) == buffer_size:
        audio_buffer = indata[:, 0]  # Alleen het eerste kanaal

# Audio stream
try:
    stream = sd.InputStream(
        samplerate=fs, channels=1, device=device_id,
        callback=audio_callback, blocksize=buffer_size
    )
    stream.start()
except Exception as e:
    print(f"Kan audio stream niet starten: {e}")
    exit()

# Tkinter GUI
root = tk.Tk()
root.title("Live Audio Visualizer")
root.state('zoomed')  # Vergroot het venster tot het volledige scherm


# Sluitprogramma met Escape
def close_fullscreen(event):
    root.attributes('-fullscreen', False)  # Exit fullscreen
    root.destroy()

root.bind("<Escape>", close_fullscreen)

# Matplotlib Figure
fig, (ax_waveform, ax_spectrum) = plt.subplots(2, 1, figsize=(8, 6))
time_array = np.linspace(0, duration, buffer_size)
freq = np.fft.fftfreq(buffer_size, d=1/fs)[:buffer_size // 2]

# Voorbereiden van de grafieken
line_waveform, = ax_waveform.plot(time_array, np.zeros(buffer_size), lw=2)
line_spectrum, = ax_spectrum.plot(freq, np.zeros(buffer_size // 2), lw=2)

# Instellingen voor de grafieken
ax_waveform.set_title("Live Golfvorm")
ax_waveform.set_xlabel("Tijd (s)")
ax_waveform.set_ylabel("Amplitude")
ax_waveform.set_ylim(-1, 1)

ax_spectrum.set_title("Live Frequentiespectrum")
ax_spectrum.set_xlabel("Frequentie (Hz)")
ax_spectrum.set_ylabel("Amplitude")
ax_spectrum.set_xlim(0, fs // 2)
ax_spectrum.set_ylim(0, 100)

# Plaatsen van de grafiek in tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=1)

# Update functie voor animatie
def update(frame):
    global audio_buffer
    time.sleep(0.01)  # Voorkom te snelle updates

    if len(audio_buffer) != buffer_size:
        return line_waveform, line_spectrum

    # Normaliseer het microfoonsignaal
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer /= max_val

    # Update golfvorm
    line_waveform.set_ydata(audio_buffer)

    # Frequentiespectrum berekenen
    fft_data = fft(audio_buffer)
    magnitude = np.abs(fft_data[:buffer_size // 2])
    line_spectrum.set_ydata(magnitude)

    canvas.draw()
    return line_waveform, line_spectrum

# Animatie starten
ani = FuncAnimation(fig, update, interval=int(duration * 1000), blit=True, cache_frame_data=False)

# Tkinter event loop
root.mainloop()

# Stream stoppen bij afsluiten
stream.stop()
stream.close()
