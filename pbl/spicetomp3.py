
import os
import argparse
import numpy as np
from scipy.interpolate import interp1d
import soundfile as sf
from pydub import AudioSegment

def spice_raw_to_mp3(spice_output_txt, mp3_output_path, sample_rate=44100):
    """Reads ngspice transient data, interpolates timesteps, removes DC offset, and exports MP3."""
    if not os.path.exists(spice_output_txt):
        raise FileNotFoundError(f"SPICE export file not found: {spice_output_txt}")

    print(f"[+] Parsing SPICE simulation output: {spice_output_txt}")
    data = []
    with open(spice_output_txt, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    t_val = float(parts[0])
                    v_val = float(parts[1])
                    data.append((t_val, v_val))
                except ValueError:
                    continue  # Skip headers or text banners

    if not data:
        raise ValueError("No valid numeric time/voltage data found in SPICE text file.")

    data = np.array(data)
    raw_time = data[:, 0]
    raw_volts = data[:, 1]

    # 1. Deduplicate time points if SPICE logged identical timesteps
    raw_time, unique_indices = np.unique(raw_time, return_index=True)
    raw_volts = raw_volts[unique_indices]

    # 2. Interpolate variable SPICE steps onto a uniform audio sampling grid
    t_start = raw_time[0]
    t_end = raw_time[-1]
    total_samples = int((t_end - t_start) * sample_rate)
    uniform_time = np.linspace(t_start, t_end, num=total_samples, endpoint=False)
    interp_func = interp1d(raw_time, raw_volts, kind='linear', fill_value="extrapolate")
    uniform_volts = interp_func(uniform_time)

    # 3. Remove DC bias (AC Coupling emulation)
    uniform_volts = uniform_volts - np.mean(uniform_volts)

    # 4. Peak normalize to prevent digital clipping (-1.0 to +1.0 full scale)
    peak = np.max(np.abs(uniform_volts))
    if peak > 0:
        uniform_volts = uniform_volts / peak

    # Temporary WAV buffer for lossless conversion
    temp_wav = mp3_output_path + ".temp.wav"
    sf.write(temp_wav, uniform_volts, sample_rate)

    # Convert WAV to final MP3 output via pydub
    audio_segment = AudioSegment.from_wav(temp_wav)
    audio_segment.export(mp3_output_path, format="mp3", bitrate="192k")

    if os.path.exists(temp_wav):
        os.remove(temp_wav)

    print(f"[+] Audio successfully reconstructed and saved to: {mp3_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct MP3 audio from SPICE output text file.")
    parser.add_argument("-i", "--input", required=True, help="Path to ngspice exported text file")
    parser.add_argument("-o", "--output", default="processed_output.mp3", help="Path for output MP3 file")
    parser.add_argument("-r", "--rate", type=int, default=44100, help="Output sample rate in Hz (default: 44100)")
    args = parser.parse_args()

    spice_raw_to_mp3(args.input, args.output, args.rate)