
import os
import argparse
import numpy as np
from pydub import AudioSegment


def write_pwl(time_array, samples, output_path):
    """Write time-voltage pairs in SPICE PWL text format."""
    print(f"[+] Writing {output_path} ({len(samples)} points)")

    with open(output_path, "w", encoding="utf-8") as f:
        for t, v in zip(time_array, samples):
            f.write(f"{t:.8e} {v:.8e}\n")


def mp3_to_pwl(
    mp3_path,
    pwl_output_path,
    duration_sec=1.0,
    target_sr=44100,
    peak_voltage=1.0,
    noise_rms=0.10,
    seed=42,
    clean_output="audio_clean.pwl",
):
    """Convert audio into clean and Gaussian-noise SPICE PWL files."""

    if not os.path.exists(mp3_path):
        raise FileNotFoundError(f"Input file not found: {mp3_path}")

    if duration_sec <= 0 or target_sr <= 0 or peak_voltage <= 0:
        raise ValueError("Duration, sample rate and peak voltage must be positive.")

    if noise_rms < 0:
        raise ValueError("Noise RMS fraction cannot be negative.")

    print(f"[+] Loading audio: {mp3_path}")
    audio = AudioSegment.from_file(mp3_path).set_channels(1)

    # Limit duration and resample
    audio = audio[:int(duration_sec * 1000)]
    audio = audio.set_frame_rate(target_sr)

    if len(audio) == 0:
        raise ValueError("The selected audio segment is empty.")

    # Convert integer audio samples to floating-point values
    samples = np.asarray(
        audio.get_array_of_samples(), dtype=np.float64
    )
    max_int = float(1 << (8 * audio.sample_width - 1))
    samples /= max_int

    # Normalize clean audio to the requested peak voltage
    current_peak = np.max(np.abs(samples))
    if current_peak > 0:
        clean = (samples / current_peak) * peak_voltage
    else:
        clean = samples.copy()

    # Generate Gaussian white noise
    rng = np.random.default_rng(seed)
    noise_std_volts = noise_rms * peak_voltage
    noise = rng.normal(
        loc=0.0,
        scale=noise_std_volts,
        size=len(clean),
    )

    # Add noise without clipping the Gaussian waveform
    noisy = clean + noise

    # Time points: sample spacing is 1 / target_sr
    time_array = np.arange(len(clean), dtype=np.float64) / target_sr

    # Write clean reference and noisy circuit input
    write_pwl(time_array, clean, clean_output)
    write_pwl(time_array, noisy, pwl_output_path)

    print("\n[+] Conversion complete.")
    print(f"    Sample rate: {target_sr} Hz")
    print(f"    Duration: {len(clean) / target_sr:.4f} s")
    print(f"    Clean audio peak: {np.max(np.abs(clean)):.4f} V")
    print(f"    Noise standard deviation: {noise_std_volts:.4f} V")
    print(f"    Noisy input peak: {np.max(np.abs(noisy)):.4f} V")
    print(f"    Clean reference: {clean_output}")
    print(f"    Noisy input: {pwl_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert MP3 audio to clean and noisy SPICE PWL files."
    )

    parser.add_argument("-i", "--input", required=True,
                        help="Input MP3 file")
    parser.add_argument("-o", "--output", default="audio_noisy.pwl",
                        help="Noisy PWL output")
    parser.add_argument("-d", "--duration", type=float, default=1.0,
                        help="Duration in seconds")
    parser.add_argument("-r", "--rate", type=int, default=44100,
                        help="Sample rate in Hz")
    parser.add_argument("-v", "--voltage", type=float, default=1.0,
                        help="Clean audio peak voltage in volts")
    parser.add_argument("--noise-rms", type=float, default=0.10,
                        help="Noise standard deviation as a fraction of peak voltage")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible noise")
    parser.add_argument("--clean-output", default="audio_clean.pwl",
                        help="Clean reference PWL output")

    args = parser.parse_args()

    mp3_to_pwl(
        args.input,
        args.output,
        args.duration,
        args.rate,
        args.voltage,
        args.noise_rms,
        args.seed,
        args.clean_output,
    )
