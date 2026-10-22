#!/usr/bin/env python3
"""Native Linux viewer for the QianLi IR camera (InfiRay Tiny1-C module, USB 0bda:5830).

The camera streams black frames until it receives the vendor command
y16_preview_start(path=0, mode=0), which switches the stream to 16-bit
temperature data (value / 64 = Kelvin). The command is lost whenever the
stream is closed, so it is re-sent on every start.

Temperature correction (emissivity, distance, reflected/ambient temperature,
atmospheric transmittance) is applied by the camera itself; set it with the
command-line options or adjust emissivity live.

Video recording writes two files through ffmpeg: <name>.mp4 (the coloured view
as displayed, H.264) and <name>_raw.mkv (lossless 16-bit temperature data, FFV1
gray16le, value / 64 = Kelvin).

Keys: q quit · s save snapshot (PNG + .npy in °C) · v start/stop video recording
      c next colormap · r reset range lock · l lock range
      f shutter / recalibrate now · + / - emissivity ±0.05
Mouse: hover shows the temperature under the cursor; the RECORD/STOP button in the
       toolbar above the image does the same as v.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import usb.core

VID, PID = 0x0BDA, 0x5830
W, H = 256, 192
Y16_PREVIEW_START = 0x010A
Y16_MODE_TEMPERATURE = 0
OOC_B_UPDATE = 0xC10D       # shutter + flat-field recalibration
SET_PROP_TPD_PARAMS = 0xC514
GET_PROP_TPD_PARAMS = 0x8514
BLANK = 0x8000

# name: (camera parameter id, to camera units, from camera units, allowed camera range)
TPD_PARAMS = {
    "distance": (0, lambda m: m * 128, lambda v: v / 128, (0, 25600)),            # metres
    "reflected": (1, lambda c: c + 273.15, lambda v: v - 273.15, (230, 900)),     # °C
    "ambient": (2, lambda c: c + 273.15, lambda v: v - 273.15, (230, 900)),       # °C
    "emissivity": (3, lambda e: e * 128, lambda v: v / 128, (1, 128)),
    "transmittance": (4, lambda t: t * 128, lambda v: v / 128, (1, 128)),
}


class CameraError(Exception):
    pass

COLORMAPS = [("inferno", cv2.COLORMAP_INFERNO), ("jet", cv2.COLORMAP_JET),
             ("turbo", cv2.COLORMAP_TURBO), ("hot", cv2.COLORMAP_HOT),
             ("bone", cv2.COLORMAP_BONE)]


def find_video_device():
    for node in sorted(glob.glob("/sys/class/video4linux/video*")):
        usbdir = os.path.realpath(os.path.join(node, "device", ".."))
        try:
            vid = open(os.path.join(usbdir, "idVendor")).read().strip()
            pid = open(os.path.join(usbdir, "idProduct")).read().strip()
            index = open(os.path.join(node, "index")).read().strip()
        except OSError:
            continue
        if (int(vid, 16), int(pid, 16)) == (VID, PID) and index == "0":
            return "/dev/" + os.path.basename(node)
    return None


class Camera:
    def __init__(self, device):
        self.usb = usb.core.find(idVendor=VID, idProduct=PID)
        if self.usb is None:
            sys.exit("Camera 0bda:5830 not found on USB.")
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            sys.exit(f"Cannot open {device}.")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.read_raw()  # the stream must be running before the command is sent
        try:
            # takes ~5 s to be acknowledged right after the camera powers up
            self._short_cmd(Y16_PREVIEW_START, bytes([0, 0, 0, Y16_MODE_TEMPERATURE]), timeout=15)
        except CameraError as e:
            sys.exit(f"Starting the temperature stream failed: {e}")

    def _write(self, index, data):
        try:
            self.usb.ctrl_transfer(0x41, 0x45, 0x78, index, data)
        except usb.core.USBError as e:
            raise CameraError(f"USB write failed ({e}). Missing udev rule / permissions?") from e

    def _wait(self, timeout=5):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.usb.ctrl_transfer(0xC1, 0x44, 0x78, 0x0200, 1)[0]
            if status & 0xFC:
                raise CameraError(f"camera reported error {status:#x}")
            if not status & 0x03:
                return
            time.sleep(0.001)
        raise CameraError("camera did not acknowledge the command")

    def _short_cmd(self, cmd, params=b"", wait=True, timeout=5):
        self._write(0x1D00, cmd.to_bytes(2, "little") + params.ljust(6, b"\0"))
        if wait:
            self._wait(timeout)

    def _long_cmd(self, cmd, p1, p2=0):
        self._write(0x9D00, cmd.to_bytes(2, "little") + p1.to_bytes(2, "big") + p2.to_bytes(4, "big"))
        self._write(0x1D08, bytes(8))
        self._wait()

    def get_tpd(self, name):
        pid, _, from_cam, _ = TPD_PARAMS[name]
        self._write(0x9D00, GET_PROP_TPD_PARAMS.to_bytes(2, "little") + pid.to_bytes(2, "big") + bytes(4))
        self._write(0x1D08, bytes(4) + (2).to_bytes(4, "big"))
        self._wait()
        return from_cam(int.from_bytes(self.usb.ctrl_transfer(0xC1, 0x44, 0x78, 0x1D10, 2), "big"))

    def set_tpd(self, name, value):
        pid, to_cam, from_cam, (lo, hi) = TPD_PARAMS[name]
        raw = round(to_cam(value))
        if not lo <= raw <= hi:
            raise CameraError(f"{name} must be between {from_cam(lo):.3g} and {from_cam(hi):.3g}")
        self._long_cmd(SET_PROP_TPD_PARAMS, pid, raw)

    def shutter(self):
        # The camera keeps reporting status 0x0e after this command although the
        # recalibration runs, so the status is not checked.
        self._short_cmd(OOC_B_UPDATE, bytes([1]), wait=False)

    def read_raw(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return np.frombuffer(frame.tobytes(), dtype="<u2")[:W * H].reshape(H, W)

    def read_frame(self):
        """Returns (raw, celsius), or None while the camera sends no data yet."""
        raw = self.read_raw()
        if raw is None or (raw == BLANK).all():
            return None
        return raw, raw.astype(np.float32) / 64.0 - 273.15

    def read_celsius(self):
        frame = self.read_frame()
        return None if frame is None else frame[1]

    def close(self):
        self.cap.release()


class Recorder:
    """Records the displayed view and the raw temperature data through ffmpeg."""

    def __init__(self, base, view_size):
        self.base = base
        self.start = time.time()
        self.frames = 0
        self.view = self._ffmpeg("bgr24", view_size, base + ".mp4",
                                 ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"])
        self.raw = self._ffmpeg("gray16le", (W, H), base + "_raw.mkv", ["-c:v", "ffv1", "-level", "3"])

    @staticmethod
    def _ffmpeg(pix_fmt, size, path, codec):
        return subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", pix_fmt, "-s", f"{size[0]}x{size[1]}",
             "-use_wallclock_as_timestamps", "1", "-i", "-",
             *codec, "-fps_mode", "vfr", path],
            stdin=subprocess.PIPE)

    def write(self, img, raw):
        self.view.stdin.write(img.tobytes())
        self.raw.stdin.write(raw.astype("<u2").tobytes())
        self.frames += 1

    def elapsed(self):
        return time.time() - self.start

    def stop(self):
        for proc in (self.view, self.raw):
            proc.stdin.close()
            proc.wait()
        print(f"saved {self.base}.mp4 / {self.base}_raw.mkv ({self.frames} frames, {self.elapsed():.1f} s)")


def start_recording(outdir, view_size):
    base = os.path.join(outdir, time.strftime("thermal_%Y%m%d_%H%M%S"))
    try:
        rec = Recorder(base, view_size)
    except FileNotFoundError:
        print("Recording needs ffmpeg: sudo apt install ffmpeg")
        return None
    print(f"recording to {base}.mp4 / {base}_raw.mkv")
    return rec


def render(temp, cmap, lo, hi, scale, mouse, info=""):
    norm = np.clip((temp - lo) / max(hi - lo, 0.1) * 255, 0, 255).astype(np.uint8)
    img = cv2.applyColorMap(norm, cmap)
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_CUBIC)

    def label(text, org, color=(255, 255, 255)):
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    def marker(y, x, text, color):
        p = (int(x * scale + scale / 2), int(y * scale + scale / 2))
        cv2.drawMarker(img, p, color, cv2.MARKER_CROSS, 14, 2)
        label(text, (p[0] + 8, p[1] - 8), color)

    marker(*np.unravel_index(temp.argmax(), temp.shape), f"{temp.max():.1f}C", (0, 0, 255))
    marker(*np.unravel_index(temp.argmin(), temp.shape), f"{temp.min():.1f}C", (255, 128, 0))
    marker(H // 2, W // 2, f"{temp[H // 2, W // 2]:.1f}C", (255, 255, 255))
    if mouse:
        mx, my = min(mouse[0] // scale, W - 1), min(mouse[1] // scale, H - 1)
        marker(my, mx, f"{temp[my, mx]:.1f}C", (0, 255, 0))
    label(f"min {temp.min():.1f}  max {temp.max():.1f}  range {lo:.1f}..{hi:.1f} C", (8, 20))
    label(info, (8, 40))
    return img


TOOLBAR_H = 36
REC_BUTTON = (8, 5, 130, 31)    # x1, y1, x2, y2 in window coordinates


def in_rect(rect, x, y):
    return rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]


def toolbar(width, rec):
    """The strip above the image; it is shown on screen only, never recorded or saved."""
    bar = np.full((TOOLBAR_H, width, 3), 48, np.uint8)
    x1, y1, x2, y2 = REC_BUTTON
    if rec:
        secs = int(rec.elapsed())
        cv2.rectangle(bar, (x1, y1), (x2, y2), (40, 40, 200), -1)
        cv2.rectangle(bar, (x1 + 10, y1 + 8), (x1 + 20, y1 + 18), (255, 255, 255), -1)
        text = f"STOP {secs // 60:02d}:{secs % 60:02d}"
    else:
        cv2.rectangle(bar, (x1, y1), (x2, y2), (80, 80, 80), -1)
        cv2.circle(bar, (x1 + 15, y1 + 13), 6, (0, 0, 255), -1)
        text = "RECORD"
    cv2.rectangle(bar, (x1, y1), (x2, y2), (160, 160, 160), 1)
    cv2.putText(bar, text, (x1 + 28, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def settings_text(settings):
    return (f"e={settings['emissivity']:.2f}  dist {settings['distance']:.2f}m  "
            f"refl {settings['reflected']:.0f}C  amb {settings['ambient']:.0f}C  "
            f"tau {settings['transmittance']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="V4L2 device (auto-detected)")
    ap.add_argument("--scale", type=int, default=3, help="display scale factor")
    ap.add_argument("--outdir", default=".", help="where snapshots and recordings are saved")
    ap.add_argument("--snapshot", metavar="FILE.png",
                    help="headless: wait for a calibrated frame, save it and exit")
    ap.add_argument("--record", type=float, metavar="SECONDS",
                    help="headless: wait for calibration, record SECONDS of video and exit")
    corr = ap.add_argument_group("temperature correction (applied by the camera; omitted = keep current)")
    corr.add_argument("--emissivity", type=float, help="object emissivity 0.01-1.0 (camera default 1.0)")
    corr.add_argument("--distance", type=float, help="object distance in metres (camera default 0.25)")
    corr.add_argument("--reflected", type=float, help="reflected apparent temperature in °C")
    corr.add_argument("--ambient", type=float, help="atmospheric temperature in °C")
    corr.add_argument("--transmittance", type=float, help="atmospheric transmittance 0.01-1.0")
    args = ap.parse_args()

    device = args.device or find_video_device()
    if not device:
        sys.exit("No V4L2 device for camera 0bda:5830 found.")
    cam = Camera(device)
    try:
        for name in TPD_PARAMS:
            if getattr(args, name) is not None:
                cam.set_tpd(name, getattr(args, name))
        settings = {name: cam.get_tpd(name) for name in TPD_PARAMS}
    except CameraError as e:
        cam.close()
        sys.exit(f"Setting temperature correction failed: {e}")
    print("Camera correction:", settings_text(settings))

    if args.snapshot:
        # The first ~10 s after start are uncalibrated noise until the shutter runs.
        deadline, temp = time.time() + 20, None
        while time.time() < deadline:
            frame = cam.read_celsius()
            if frame is not None:
                temp = frame
        cam.close()
        if temp is None:
            sys.exit("No frames received.")
        cv2.imwrite(args.snapshot, render(temp, COLORMAPS[0][1], temp.min(), temp.max(), args.scale, None,
                                          settings_text(settings)))
        np.save(os.path.splitext(args.snapshot)[0] + ".npy", temp)
        print(f"saved {args.snapshot}: min {temp.min():.1f} max {temp.max():.1f} "
              f"center {temp[H // 2, W // 2]:.1f} °C")
        return

    if args.record:
        deadline = time.time() + 12   # skip the uncalibrated start-up frames
        while time.time() < deadline:
            cam.read_frame()
        rec = start_recording(args.outdir, (W * args.scale, H * args.scale))
        if rec is None:
            cam.close()
            sys.exit(1)
        try:
            while rec.elapsed() < args.record:
                frame = cam.read_frame()
                if frame is not None:
                    raw, temp = frame
                    rec.write(render(temp, COLORMAPS[0][1], temp.min(), temp.max(), args.scale, None,
                                     settings_text(settings)), raw)
        finally:
            rec.stop()
            cam.close()
        return

    win = "Thermal camera"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mouse, clicks = [], []

    def on_mouse(event, x, y, *_):
        # mouse holds image coordinates; empty while the pointer is over the toolbar
        mouse[:] = [x, y - TOOLBAR_H] if y >= TOOLBAR_H else []
        if event == cv2.EVENT_LBUTTONDOWN and in_rect(REC_BUTTON, x, y):
            clicks.append(REC_BUTTON)

    cv2.setMouseCallback(win, on_mouse)
    cmap_idx, locked, rec = 0, None, None
    try:
        while True:
            frame = cam.read_frame()
            temp = None if frame is None else frame[1]
            if temp is None:
                img = np.zeros((H * args.scale, W * args.scale, 3), np.uint8)
                cv2.putText(img, "waiting for camera...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                lo, hi = locked or (float(temp.min()), float(temp.max()))
                img = render(temp, COLORMAPS[cmap_idx][1], lo, hi, args.scale, mouse, settings_text(settings))
                if rec:
                    rec.write(img, frame[0])
            cv2.imshow(win, np.vstack([toolbar(img.shape[1], rec), img]))
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if temp is None:
                clicks.clear()
                continue
            if clicks:
                clicks.clear()
                key = ord("v")
            if key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
            elif key == ord("l"):
                locked = (float(temp.min()), float(temp.max()))
            elif key == ord("r"):
                locked = None
            elif key == ord("v"):
                if rec:
                    rec.stop()
                    rec = None
                else:
                    rec = start_recording(args.outdir, (img.shape[1], img.shape[0]))
            elif key == ord("f"):
                cam.shutter()
            elif key in (ord("+"), ord("="), ord("-")):
                step = -0.05 if key == ord("-") else 0.05
                value = min(1.0, max(0.05, round(settings["emissivity"] + step, 2)))
                try:
                    cam.set_tpd("emissivity", value)
                    settings["emissivity"] = cam.get_tpd("emissivity")
                except CameraError as e:
                    print(f"Setting emissivity failed: {e}")
            elif key == ord("s"):
                base = os.path.join(args.outdir, time.strftime("thermal_%Y%m%d_%H%M%S"))
                cv2.imwrite(base + ".png", img)
                np.save(base + ".npy", temp)
                print(f"saved {base}.png / .npy")
    finally:
        if rec:
            rec.stop()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
