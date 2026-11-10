# Nova5 support

Installable `handeye` calibration/robot modules and `pgi_hand.protocol` / `motion`
helpers. Calibration files and device settings are supplied per installation.
During deployment, `nova5_arm` owns the arm and gripper connections. Stop that
deployment before using the calibration CLI to access those devices directly.

## Install

Use Linux and Python 3.11 or 3.12. Camera capture requires a graphical desktop,
OpenGL/GLib/libusb libraries and USB permissions. Building `evdev` from source
requires GCC or Clang, Linux input headers and headers matching the selected
Python. If that Python requests an unavailable Clang, use `CC=gcc CXX=g++` for
the install command when GCC is available. Bundled Python headers must come
from that interpreter, not a different system Python version.
From this repository's root:

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[camera,robot,test]'
```

`camera` installs `pyorbbecsdk2==2.1.3` (import `pyorbbecsdk`), used with Python 3.12.
Follow the [Orbbec USB permission instructions](https://github.com/orbbec/pyorbbecsdk/tree/v2-main).
`robot` installs [Dobot V4 SDK](https://github.com/Dobot-Arm/TCP-IP-Python-V4/tree/55ec1ec82201aaf2e6d54aa47b6272bbaa14c815)
commit `55ec1ec82201aaf2e6d54aa47b6272bbaa14c815`; verify controller/firmware
compatibility before motion. Providers can install `.[camera]` or `.[robot]`
separately. SDK source is not bundled; upstream licenses apply (Dobot: MIT).
This package uses MulanPSL-2.0.

## Calibrate

Copy `config.example.json` to `config.json`. Set the robot IP, camera serial/profile
and measured TCP offset. Keep `robot.sdk_path` null to use the installed SDK.
The 174 mm TCP template includes the measured 165 mm gripper assembly (with its black
top plate) and the 9 mm white adapter plate. Verify the total from the robot flange. All data paths
resolve relative to the configuration file; the CLI default is `./config.json`.

The example workspace `[220, -90, -90]` to `[630, 0, 620]` mm covers recorded endpoints of the demonstrated pick-and-return task with a provisional margin. Confirm it for the installation; it is not a collision-free-space guarantee. Motion remains disabled in the template.

```bash
cp config.example.json config.json
nova5-checkerboard --output boards/checkerboard.pdf
nova5-calibrate --config config.json doctor
```

Print at actual size, measure the squares, and set `square_size_mm` and
`print_size_verified`. The template uses 9 x 6 inner corners and 25 mm squares.
Fix the robot base and camera mount before collecting data.

1. For intrinsics, vary board angle, distance and position across the image.
   The board may move. Press S to save and Q to quit.

   ```bash
   nova5-calibrate --config config.json capture-intrinsics
   nova5-calibrate --config config.json calibrate-intrinsics
   ```

2. Fix the board on the work surface. Keep the board, base and camera mount fixed;
   change arm pose with rotations about different axes and the full board visible.
   Exit drag/jog mode and remain stationary for at least one second before S.
   Capture checks stationary image/feedback pairs using host receipt timestamps.

   ```bash
   nova5-calibrate --config config.json capture-handeye
   nova5-calibrate --config config.json calibrate-handeye
   ```

3. Keep the board on the work plane and the arm stationary. Set a nonzero offset
   if the board face is above the target surface along the board normal.

   ```bash
   nova5-calibrate --config config.json capture-plane --surface-offset-mm 0
   ```

Outputs are `calibrations/intrinsics.json`, `handeye.json` and `plane.json`. Keep
quality limits unchanged; resolve rejected data before proceeding. Recalibrate
when the camera/profile, wrist mount or work plane changes. Coordinates use mm,
degrees, User0/Tool0 and software TCP compensation.

`project` and `tcp-target` calculate targets offline; `robot-status` reads feedback.
`click` previews targets; execution needs `--execute`, M confirmation and all
configuration/runtime checks. `record-validation` records operator measurements
already performed; it does not measure accuracy. See `nova5-calibrate --help`.

## Verify without hardware

CI installs `.[test]` and runs `python -m pytest -q` on Linux with Python 3.12.

```bash
python -m pytest -q
python -m pip wheel --no-deps . --wheel-dir dist
nova5-calibrate tcp-target --tcp-pose 100 200 300 180 0 0 --offset 0 0 174 0 0 0
```

Tests use fake devices. They cover geometry, calibration, provenance, stationary
capture, camera failures, robot state checks and bounded Modbus exchanges. Install
the wheel into a separate environment and run the CLI outside this checkout to
check packaging without connecting hardware.
