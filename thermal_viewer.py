#!/usr/bin/env python3
"""Native Linux viewer for the QianLi IR camera (InfiRay Tiny1-C module, USB 0bda:5830).

The camera streams black frames until it receives the vendor command
y16_preview_start(path=0, mode=0), which switches the stream to 16-bit
temperature data (value / 64 = Kelvin). The command is lost whenever the
stream is closed, so it is re-sent on every start.

Temperature correction (emissivity, distance, reflected/ambient temperature,
atmospheric transmittance) is applied by the camera itself; set it with the
command-line options or adjust emissivity live.

Keys: q quit · s save snapshot (PNG + .npy in °C) · c next colormap · r reset range lock · l lock range
      f shutter / recalibrate now · + / - emissivity ±0.05
Mouse: hover shows the temperature under the cursor.
"""
import argparse
import glob
import os
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

    def read_celsius(self):
        raw = self.read_raw()
        if raw is None or (raw == BLANK).all():
            return None
        return raw.astype(np.float32) / 64.0 - 273.15

    def close(self):
        self.cap.release()


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


def settings_text(settings):
    return (f"e={settings['emissivity']:.2f}  dist {settings['distance']:.2f}m  "
            f"refl {settings['reflected']:.0f}C  amb {settings['ambient']:.0f}C  "
            f"tau {settings['transmittance']:.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", help="V4L2 device (auto-detected)")
    ap.add_argument("--scale", type=int, default=3, help="display scale factor")
    ap.add_argument("--outdir", default=".", help="where snapshots are saved")
    ap.add_argument("--snapshot", metavar="FILE.png",
                    help="headless: wait for a calibrated frame, save it and exit")
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

    win = "Thermal camera"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    mouse = []

    def on_mouse(event, x, y, *_):
        mouse[:] = [x, y]

    cv2.setMouseCallback(win, on_mouse)
    cmap_idx, locked = 0, None
    try:
        while True:
            temp = cam.read_celsius()
            if temp is None:
                img = np.zeros((H * args.scale, W * args.scale, 3), np.uint8)
                cv2.putText(img, "waiting for camera...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            else:
                lo, hi = locked or (float(temp.min()), float(temp.max()))
                img = render(temp, COLORMAPS[cmap_idx][1], lo, hi, args.scale, mouse, settings_text(settings))
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if temp is None:
                continue
            if key == ord("c"):
                cmap_idx = (cmap_idx + 1) % len(COLORMAPS)
            elif key == ord("l"):
                locked = (float(temp.min()), float(temp.max()))
            elif key == ord("r"):
                locked = None
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
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
