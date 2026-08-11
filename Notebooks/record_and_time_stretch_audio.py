"""Record from the microphone and change speed without the chipmunk effect."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_conda_dll_path() -> None:
    """On Windows, conda DLLs (PortAudio, MKL, ...) live in Library/bin."""
    libbin = Path(sys.prefix) / "Library" / "bin"
    if not libbin.is_dir():
        return
    os.environ["PATH"] = str(libbin) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(libbin))


def record_and_time_stretch_audio(
    duration: float = 3.0,
    fs: int = 22050,
    rate: float = 1.5,
    play: bool = True,
    show_players: bool = True,
):
    """Record from the default mic, time-stretch (pitch preserved), and play back.

    Parameters
    ----------
    duration :
        Seconds to record.
    fs :
        Sample rate in Hz.
    rate :
        Speed factor: >1 faster, <1 slower. Pitch is preserved (no chipmunk).
    play :
        Play original and stretched audio via the speakers.
    show_players :
        Embed ``IPython.display.Audio`` widgets when running in a notebook.

    Returns
    -------
    y, y_stretched :
        Original and time-stretched float32 mono arrays.
    """
    _ensure_conda_dll_path()

    import numpy as np
    import sounddevice as sd
    import librosa

    print(f"Recording {duration} s from the default microphone ...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    y = np.asarray(audio[:, 0], dtype=np.float32)
    print(f"Done. Original length: {len(y) / fs:.2f} s")

    # Phase-vocoder time stretch: changes tempo, keeps pitch
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    print(f"Stretched length at rate={rate}: {len(y_stretched) / fs:.2f} s")

    if play:
        print("Playing original ...")
        sd.play(y, fs)
        sd.wait()
        print("Playing time-stretched (same pitch) ...")
        sd.play(y_stretched, fs)
        sd.wait()

    if show_players:
        try:
            from IPython.display import Audio, display

            print("Original:")
            display(Audio(y, rate=fs))
            print(f"Time-stretched (rate={rate}, pitch preserved):")
            display(Audio(y_stretched, rate=fs))
        except Exception:
            pass

    return y, y_stretched


if __name__ == "__main__":
    record_and_time_stretch_audio()
