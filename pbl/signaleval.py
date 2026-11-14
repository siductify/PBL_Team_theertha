
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from pydub import AudioSegment

def analyze_audio_systems(input_mp3, output_mp3):
    # Load audio files
    a_in = AudioSegment.from_file(input_mp3).set_channels(1)
    a_out = AudioSegment.from_file(output_mp3).set_channels(1)
    
    sr = a_in.frame_rate
    s_in = np.array(a_in.get_array_of_samples(), dtype=np.float32)
    s_out = np.array(a_out.get_array_of_samples(), dtype=np.float32)
    
    # Match lengths for fair comparison
    min_len = min(len(s_in), len(s_out))
    s_in, s_out = s_in[:min_len], s_out[:min_len]
    
    # 1. Compute Fourier Transforms (Signals & Systems: X(f) and Y(f))
    freqs = rfftfreq(min_len, 1.0 / sr)
    X_f = np.abs(rfft(s_in))
    Y_f = np.abs(rfft(s_out))
    
    # Normalize magnitudes to dB
    X_dB = 20 * np.log10(X_f / np.max(X_f) + 1e-6)
    Y_dB = 20 * np.log10(Y_f / np.max(Y_f) + 1e-6)
    
    # 2. Estimate System Transfer Function: |H(f)| = |Y(f)| / |X(f)|
    H_est = Y_dB - X_dB
    
    # 3. Plotting Verification
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 1, 1)
    t = np.linspace(0, min_len / sr, min_len)
    plt.plot(t[:1000], s_in[:1000] / np.max(np.abs(s_in)), label="x(t) Input Audio", alpha=0.7)
    plt.plot(t[:1000], s_out[:1000] / np.max(np.abs(s_out)), label="y(t) Circuit Output (LTI)", alpha=0.7)
    plt.title("Signals & Systems Time-Domain Response: x(t) vs y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized Amplitude")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.semilogx(freqs, X_dB, label="|X(f)| Input Spectrum", alpha=0.6)
    plt.semilogx(freqs, Y_dB, label="|Y(f)| Output Spectrum", alpha=0.8)
    plt.title("Frequency-Domain Fourier Analysis: |X(f)| vs |Y(f)|")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude [dB]")
    plt.xlim(20, 20000)
    plt.ylim(-60, 5)
    plt.grid(True, which="both")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("signals_system_validation.png", dpi=300)
    print("[+] Plot saved: signals_system_validation.png")
    plt.show()

if __name__ == "__main__":
    analyze_audio_systems("input_sample.mp3", "processed_circuit_output.mp3")