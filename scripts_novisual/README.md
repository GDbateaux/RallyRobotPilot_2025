# Scripts No Visual - Pure Physics Demos

This folder contains demonstration scripts for the **headless physics-only** simulation system.

## Purpose

These scripts demonstrate the `rallyrobopilot_novisual` package, which provides:
- Pure Python physics calculations (NO Ursina, NO rendering)
- Headless simulation (NO window, NO GPU)
- Same physics as the visual game (0.1s timestep)
- Infinite speed (no FPS cap)

## Scripts

### `main_novisual.py`

Demonstrates basic physics simulation with car state output.

**Usage:**
```bash
python scripts_novisual/main_novisual.py
```

**Output:**
- Track information (name, start position, orientation)
- Car state every 10 frames (position, speed, angle, rotation speed)
- Multiple test scenarios:
  - Driving forward
  - Turning right
  - Coasting
  - Braking

**Example Output:**
```
Frame    0 | Pos: (  -0.25,    1.00,   -0.00) | Speed:   2.50 | Angle:  270.0° | RotSpeed:  0.00
Frame   10 | Pos: ( -16.50,    1.00,   -0.00) | Speed:  27.50 | Angle:  270.0° | RotSpeed:  0.00
Frame   20 | Pos: ( -57.50,    1.00,   -0.00) | Speed:  50.00 | Angle:  270.0° | RotSpeed:  0.00
```

## Use Cases

- **Testing physics calculations** - Verify car behavior without graphics
- **Debugging GA training** - Inspect individual frames during evolution
- **Performance benchmarking** - Measure pure physics speed (no rendering overhead)
- **CI/CD testing** - Run physics tests in headless environments

## Related

- `rallyrobopilot_novisual/` - Physics-only package
- `scripts/train_ga_full_lap.py --headless` - Fast GA training using this system
- `scripts/main.py` - Visual game (for manual play and replay)