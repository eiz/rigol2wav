#!/usr/bin/env python3
"""
Convert Rigol .bin oscilloscope files to per-channel .wav (float32 or int16) and a JSON metadata file.

Dependencies: numpy, scipy
Recommended install/run: `uv run --with numpy --with scipy rigol_bin_to_wav.py <file.bin>`

Format notes:
- Mirrors a DHO800-series .bin layout matching the MATLAB snippet you shared.
- Offsets match the MATLAB fseek pattern (y_units at 68 + stride*i, name at 128 + stride*i, data at 172 + stride*i).
- stride = 156 + buffer_size
- Data points are float32 little-endian.

WAV:
- dtype 'float32' writes IEEE float WAV (format code 3) via scipy.io.wavfile.write.
- dtype 'int16' writes PCM16 via the same function.
- `--normalize peak` scales per-channel peak to `--target-peak` (float WAV uses [-1,1] scale; int16 uses full-scale 32767).
- `--normalize none` preserves original volt units (float WAV can exceed 1.0 if your volts are large; most DAWs handle it).

Usage:
  uv run --with numpy --with scipy rigol_bin_to_wav.py input.bin --wav-dtype float32
"""

from __future__ import annotations
import argparse
import json
import math
import struct
from pathlib import Path
from typing import BinaryIO, Dict, List, Tuple

import numpy as np
from scipy.io import wavfile as sciowav


UNIT_TYPES = [
    "Unknown", "Volts (V)", "Seconds (s)", "Constant",
    "Amps (A)", "Decibel (dB)", "Hertz (Hz)"
]


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if len(b) != n:
        raise EOFError(f"Unexpected EOF: wanted {n} bytes, got {len(b)}")
    return b


def _read_u8chars(f: BinaryIO, n: int) -> str:
    raw = _read_exact(f, n)
    return raw.decode("ascii", errors="ignore").rstrip("\x00 ").strip()


def _read_struct(f: BinaryIO, fmt: str):
    fmt_le = "<" + fmt
    size = struct.calcsize(fmt_le)
    return struct.unpack(fmt_le, _read_exact(f, size))


def read_rigol_bin(path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Read a Rigol .bin file and return (Y, nfo)
    - Y: ndarray shape (n_channels, n_pts), dtype=float32
    - nfo: dict with header & derived info
    """
    with path.open("rb") as f:
        cookie = _read_exact(f, 2).decode("ascii", errors="ignore")
        if cookie != "RG":
            raise ValueError("Not a Rigol waveform file: cookie != 'RG'")

        version = _read_exact(f, 2).decode("ascii", errors="ignore")

        (file_size,)   = _read_struct(f, "Q")
        (n_waveforms,) = _read_struct(f, "I")

        (header_size,)      = _read_struct(f, "I")
        (waveform_type,)    = _read_struct(f, "I")
        (n_buffers,)        = _read_struct(f, "I")
        (n_pts,)            = _read_struct(f, "I")
        (count,)            = _read_struct(f, "I")
        (x_range,)          = _read_struct(f, "f")
        (x_disp_origin,)    = _read_struct(f, "d")
        (x_increment,)      = _read_struct(f, "d")
        (x_origin,)         = _read_struct(f, "d")
        (x_units_code,)     = _read_struct(f, "I")
        (y_units_global,)   = _read_struct(f, "I")
        f_date = _read_u8chars(f, 16)
        f_time = _read_u8chars(f, 16)
        model  = _read_u8chars(f, 24)

        # Secondary header at absolute 156
        f.seek(156, 0)
        (wfm_header_size,) = _read_struct(f, "I")
        (buffer_type,)     = _read_struct(f, "H")
        (bytes_per_point,) = _read_struct(f, "H")
        (buffer_size,)     = _read_struct(f, "Q")

        ch_names: List[str] = []
        ch_units: List[str] = []
        Y = np.empty((n_waveforms, n_pts), dtype=np.float32)

        stride = 156 + buffer_size  # bytes between per-channel blocks
        for i in range(n_waveforms):
            # Per-channel y_units
            f.seek(68 + stride * i, 0)
            (yu_code,) = _read_struct(f, "I")
            unit = UNIT_TYPES[yu_code] if 0 <= yu_code < len(UNIT_TYPES) - 1 else "Unknown"
            ch_units.append(unit)

            # Per-channel name
            f.seek(128 + stride * i, 0)
            ch_name = _read_u8chars(f, 16) or f"CH{i+1}"
            ch_names.append(ch_name)

            # Data block (float32 LE)
            f.seek(172 + stride * i, 0)
            raw = _read_exact(f, n_pts * 4)
            data = np.frombuffer(raw, dtype="<f4")
            if data.size != n_pts:
                raise EOFError(f"Incomplete data block for channel {i+1}")
            # Clean NaNs/Infs
            data = np.nan_to_num(data, copy=True)
            Y[i, :] = data

        nfo: Dict = {
            "cookie": cookie,
            "version": version,
            "file_size": int(file_size),
            "n_waveforms": int(n_waveforms),
            "n_buffers": int(n_buffers),
            "n_pts": int(n_pts),
            "count": int(count),
            "model": model,
            "f_date": f_date,
            "f_time": f_time,
            "x_range": float(x_range),
            "x_display_origin": float(x_disp_origin),
            "x_increment": float(x_increment),
            "x_origin": float(x_origin),
            "x_units": UNIT_TYPES[x_units_code] if 0 <= x_units_code < len(UNIT_TYPES) else "Unknown",
            "y_units_global": UNIT_TYPES[y_units_global] if 0 <= y_units_global < len(UNIT_TYPES) else "Unknown",
            "wfm_header_size": int(wfm_header_size),
            "buffer_type": int(buffer_type),
            "bytes_per_point": int(bytes_per_point),
            "buffer_size": int(buffer_size),
            "channel_names": ch_names,
            "y_units": ch_units,
            "x_start": float(-x_origin),   # matches MATLAB's nfo.x_start for 'screen' save
            "dx": float(x_increment),
            "sample_rate_nominal_hz": float(1.0 / x_increment) if x_increment != 0 else None,
        }
        return Y, nfo


def write_wavs_and_metadata(
    Y: np.ndarray,
    nfo: Dict,
    base_out: Path,
    wav_dtype: str = "float32",
    normalize: str = "peak",
    target_peak: float = 0.98,
) -> Dict:
    """
    Write one WAV per channel and a JSON metadata file.
    wav_dtype: 'float32' (IEEE float) or 'int16'
    normalize: 'peak' or 'none'
    """
    base_out.parent.mkdir(parents=True, exist_ok=True)
    sr = nfo.get("sample_rate_nominal_hz")
    if sr is None or not math.isfinite(sr) or sr <= 0:
        raise ValueError("Invalid sample rate derived from x_increment")
    sample_rate = int(round(sr))

    ch_stats: List[Dict] = []
    for i, ch_name in enumerate(nfo["channel_names"]):
        y = Y[i, :].astype(np.float64, copy=False)
        absmax = float(np.max(np.abs(y))) if y.size else 0.0

        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in ch_name) or f"CH{i+1}"
        wav_path = base_out.with_name(f"{base_out.name}_ch{i+1}_{safe_name}.wav")

        if wav_dtype == "float32":
            # In float WAV, typical full-scale is +/-1.0
            if normalize == "peak":
                scale = (target_peak / absmax) if absmax > 0 else 1.0
            elif normalize == "none":
                scale = 1.0
            else:
                raise ValueError("normalize must be 'peak' or 'none'")
            y_out = (y * scale).astype(np.float32, copy=False)
            sciowav.write(str(wav_path), sample_rate, y_out)
        elif wav_dtype == "int16":
            if normalize == "peak":
                scale = (target_peak * 32767.0 / absmax) if absmax > 0 else 1.0
            elif normalize == "none":
                scale = 1.0
            else:
                raise ValueError("normalize must be 'peak' or 'none'")
            y_i16 = np.clip(np.rint(y * scale), -32768, 32767).astype(np.int16, copy=False)
            sciowav.write(str(wav_path), sample_rate, y_i16)
        else:
            raise ValueError("wav_dtype must be 'float32' or 'int16'")

        ch_stats.append({
            "channel_index": i + 1,
            "channel_name": ch_name,
            "y_units": nfo["y_units"][i] if i < len(nfo["y_units"]) else "Unknown",
            "wav_file": wav_path.name,
            "wav_dtype": wav_dtype,
            "normalize": normalize,
            "target_peak": target_peak if normalize == "peak" else None,
            "scale_applied": None if absmax == 0 else (
                target_peak / absmax if wav_dtype == "float32" and normalize == "peak" else
                (target_peak * 32767.0 / absmax) if wav_dtype == "int16" and normalize == "peak" else
                1.0
            ),
            "absmax_input_volts": absmax,
            "mean_volts": float(np.mean(y)) if y.size else 0.0,
            "min_volts": float(np.min(y)) if y.size else 0.0,
            "max_volts": float(np.max(y)) if y.size else 0.0,
        })

    meta = dict(nfo)
    meta.update({
        "output_base": base_out.name,
        "sample_rate": sample_rate,
        "channels": ch_stats,
    })

    json_path = base_out.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2)

    return meta


def main():
    ap = argparse.ArgumentParser(description="Convert Rigol .bin to per-channel .wav and JSON metadata.")
    ap.add_argument("input", type=Path, help="Path to Rigol .bin file")
    ap.add_argument("--outdir", type=Path, default=None, help="Output directory (default: alongside input)")
    ap.add_argument("--wav-dtype", choices=["float32", "int16"], default="float32",
                    help="WAV sample type (IEEE float32 or PCM16). Default: float32.")
    ap.add_argument("--normalize", choices=["peak", "none"], default="peak",
                    help="Scaling: 'peak' maps channel peak to target; 'none' preserves raw volts.")
    ap.add_argument("--target-peak", type=float, default=0.98,
                    help="When normalize=peak, map channel peak to this fraction of FS.")
    args = ap.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise SystemExit(f"Input not found: {in_path}")

    Y, nfo = read_rigol_bin(in_path)
    outdir = args.outdir if args.outdir is not None else in_path.parent
    base = outdir / in_path.with_suffix("").name

    meta = write_wavs_and_metadata(
        Y, nfo,
        base_out=base,
        wav_dtype=args.wav_dtype,
        normalize=args.normalize,
        target_peak=args.target_peak,
    )

    print(f"Wrote metadata: {base.with_suffix('.json')}")
    for ch in meta["channels"]:
        print(f"Wrote: {ch['wav_file']}  ({ch['channel_name']})")


if __name__ == "__main__":
    main()
