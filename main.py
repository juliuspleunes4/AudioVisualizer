import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.fftpack import fft
import tkinter as tk
from matplotlib.animation import FuncAnimation
import time

# Parameters
fs = 16000  # Lagere sample rate
duration = 0.2  # Buffertijd in seconden
buffer_size = int(fs * duration)  # Buffergrootte
device_id = 2  # Specifiek apparaat-ID (USB MICROPHONE)
audio_buffer = np.zeros(buffer_size)  # Initieer lege buffer

# Callback voor live audio
def audio_callback(indata, frames, time, status):
    global audio_buffer
    if status:
        print(f"Status: {status}")
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
root.state('zoomed')

# Sluitprogramma met Escape
def close_fullscreen(event):
    root.destroy()

root.bind("<Escape>", close_fullscreen)

# Matplotlib Figure
fig, (ax_waveform, ax_spectrum) = plt.subplots(2, 1, figsize=(8, 6))
time_array = np.linspace(0, duration, buffer_size)
freq = np.fft.fftfreq(buffer_size, d=1/fs)[:buffer_size // 2]

# Voorbereiden van de grafieken
line_waveform, = ax_waveform.plot(time_array, np.zeros(buffer_size), lw=2, color='blue', label="Golfvorm")
line_spectrum, = ax_spectrum.plot(freq, np.zeros(buffer_size // 2), lw=2, color='green', label="Spectrum")

# Grid toevoegen voor betere leesbaarheid
ax_waveform.grid(True, linestyle='--', alpha=0.6)
ax_spectrum.grid(True, linestyle='--', alpha=0.6)

# Instellingen voor de golfvorm
ax_waveform.set_title("Live Golfvorm")
ax_waveform.set_xlabel("Tijd (s)")
ax_waveform.set_ylabel("Amplitude")
ax_waveform.set_xlim(0, duration)
ax_waveform.set_ylim(-1, 1)
ax_waveform.legend()

# Instellingen voor het frequentiespectrum
ax_spectrum.set_title("Live Frequentiespectrum")
ax_spectrum.set_xlabel("Frequentie (Hz)")
ax_spectrum.set_ylabel("Amplitude (dB)")
ax_spectrum.set_xlim(0, fs // 4)
ax_spectrum.set_ylim(0, 50)
ax_spectrum.legend()

# Plaatsen van de grafiek in tkinter
canvas = FigureCanvasTkAgg(fig, master=root)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(fill=tk.BOTH, expand=1)

# Annotatie voor piekfrequentie
peak_freq_annotation = ax_spectrum.annotate(
    "", xy=(0, 0), xytext=(50, 30),
    textcoords="offset points", arrowprops=dict(arrowstyle="->"),
    fontsize=10, color="red"
)

# Frequentiespectrum over meerdere frames middelen
spectrum_average = np.zeros(buffer_size // 2)
alpha = 0.1  # Gewicht voor smoothness

# Live-informatie
volume_label = tk.Label(root, text="Volume: 0.0", font=("Arial", 12))
volume_label.pack(anchor='w')
frequency_label = tk.Label(root, text="Luidste Frequentie: 0 Hz", font=("Arial", 12))
frequency_label.pack(anchor='w')

# Update functie voor animatie
def update(frame):
    global audio_buffer, spectrum_average

    time.sleep(0.05)  # Verlaag update snelheid

    if len(audio_buffer) != buffer_size:
        return line_waveform, line_spectrum

    # Normaliseer het microfoonsignaal
    max_val = np.max(np.abs(audio_buffer))
    if max_val > 0:
        audio_buffer /= max_val

    # Update golfvorm
    line_waveform.set_ydata(audio_buffer)

    # Frequentiespectrum berekenen en middelen
    fft_data = fft(audio_buffer)
    magnitude = np.abs(fft_data[:buffer_size // 2])
    spectrum_average = alpha * magnitude + (1 - alpha) * spectrum_average

    # Convert to decibel scale for readability
    spectrum_dB = 20 * np.log10(spectrum_average + 1e-10)
    line_spectrum.set_ydata(spectrum_dB)

    # Vind de sterkste frequentie
    peak_idx = np.argmax(spectrum_average)
    peak_freq = freq[peak_idx]
    peak_amplitude = spectrum_dB[peak_idx]

    # Update annotatie
    peak_freq_annotation.set_position((peak_freq, peak_amplitude))
    peak_freq_annotation.set_text(f"{int(peak_freq)} Hz")

    # Update labels
    volume_label.config(text=f"Volume: {max_val:.2f}")
    frequency_label.config(text=f"Luidste Frequentie: {int(peak_freq)} Hz")

    canvas.draw()
    return line_waveform, line_spectrum

# Animatie starten
ani = FuncAnimation(fig, update, interval=200, blit=True, cache_frame_data=False)

# Tkinter event loop
root.mainloop()

# Stream stoppen bij afsluiten
stream.stop()
stream.close()
