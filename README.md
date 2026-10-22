# QianLi IR thermal camera viewer for Linux

Native Linux viewer for the QianLi IR (PCB/microscope) thermal camera, a rebranded
InfiRay Tiny1-C module (USB `0bda:5830`, 256×192, firmware `TC1_V1.02`). It replaces
the Windows-only "QianLi IR" app (`PCBDiagnose.exe`), which talks to the camera through
libusb/WinUSB and therefore does not work under Wine.

## How it works

The camera enumerates as a standard UVC webcam but sends black frames until it receives
the vendor command `y16_preview_start(path=0, mode=0)`. After that every frame carries a
16-bit temperature per pixel (`value / 64 = Kelvin`). The mode is reset whenever the
stream closes, so the viewer sends it on every start. The first ~10 s after start are
noise until the camera's shutter calibrates.

The vendor commands were taken from the Windows app's `libircmd.dll` and the
[P2Pro-Viewer](https://github.com/LeoDJ/P2Pro-Viewer) project:

| Command | Bytes / code | Purpose |
|---|---|---|
| `y16_preview_start` | `0a 01 00 00 00 <mode> 00 00` | start temperature stream (mode 0) |
| `ooc_b_update` | `0d c1 01 00 00 00 00 00` | shutter + flat-field recalibration |
| `set/get_prop_tpd_params` | `0xc514` / `0x8514` | emissivity, distance, reflected/ambient temperature, transmittance |

## Setup

```bash
sudo apt install python3-opencv python3-usb python3-numpy ffmpeg
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="5830", MODE="0660", GROUP="plugdev", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/99-qianli-ir.rules
sudo udevadm control --reload && sudo udevadm trigger
```

The udev rule allows sending the vendor commands without root. Replug the camera afterwards.

## Usage

```bash
python3 thermal_viewer.py                        # live view
python3 thermal_viewer.py --emissivity 0.95      # with temperature correction
python3 thermal_viewer.py --snapshot out.png     # headless: save one calibrated frame (+ out.npy)
python3 thermal_viewer.py --record 60            # headless: record 60 s of video
python3 thermal_viewer.py --help
```

Keys: `q` quit · `s` snapshot (PNG + `.npy` in °C) · `v` start/stop recording · `c` colormap · `l`/`r` lock/reset
colour range · `f` shutter/recalibrate · `+`/`-` emissivity ±0.05. Hover the mouse to read
the temperature under the cursor.

Correction options (`--emissivity`, `--distance` in m, `--reflected` and `--ambient` in °C,
`--transmittance`) are applied by the camera and stay set until it is unplugged; omitted
options keep the camera's current value. The distance unit (1/128 m) is inferred, not
documented.

## Video recording

Each recording (`v` key or `--record SECONDS`) writes two files to `--outdir`:

- `thermal_<time>.mp4`: the coloured view as displayed (H.264), for watching and sharing.
- `thermal_<time>_raw.mkv`: lossless 16-bit temperature data (FFV1, `gray16le`, 256×192,
  real frame timestamps), about 1.2 GB per hour. Pixel value / 64 = Kelvin.

Reading the raw recording back:

```python
import subprocess, numpy as np
data = subprocess.run(["ffmpeg", "-v", "error", "-i", "thermal_..._raw.mkv",
                       "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], capture_output=True).stdout
celsius = np.frombuffer(data, "<u2").reshape(-1, 192, 256) / 64 - 273.15   # (frames, rows, cols)
```
