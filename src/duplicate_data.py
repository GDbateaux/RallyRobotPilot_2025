import pickle
import lzma
from pathlib import Path

if __name__ == "__main__":
    TRACK_NAME = "not_so_simple_track_2"
    dir_path = Path(Path(__file__).parent.parent / "data" / TRACK_NAME)

    for filepath in sorted(dir_path.glob("*.npz")):
        if filepath.stem.endswith("_flipped") or filepath.stem + "_flipped" in [f.stem for f in dir_path.glob("*.npz")]:
            continue

        with lzma.open(filepath, "rb") as file:
            data = pickle.load(file)

        augmented_data = []

        for e in data:
            e.image = e.image[:, ::-1]

            ctrl = list(e.current_controls)
            ctrl[2], ctrl[3] = ctrl[3], ctrl[2]
            e.current_controls = tuple(ctrl)

            augmented_data.append(e)

        out_name = filepath.stem + "_flipped.npz"
        out_path = dir_path / out_name

        with lzma.open(out_path, "wb") as f:
            pickle.dump(augmented_data, f)

        print(f"Saved: {out_path}")
