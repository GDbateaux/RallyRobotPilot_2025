import pickle
import lzma
import copy

from pathlib import Path


if __name__ == "__main__":
    TRACK_NAME = "simple_track"
    dir_path = Path(Path(__file__).parent.parent / "data" / TRACK_NAME)

    for filepath in sorted(dir_path.glob("*.npz")):
        if filepath.stem.endswith("_flipped"):
            continue
        
        with lzma.open(filepath, "rb") as file:
            data = pickle.load(file)

        augmented_data = []

        for e in data:
            new_e = copy.deepcopy(e)

            new_e.image = new_e.image[:, ::-1]

            ctrl = list(new_e.current_controls)
            ctrl[2], ctrl[3] = ctrl[3], ctrl[2]

            new_e.current_controls = tuple(ctrl)

            augmented_data.append(new_e)
        
        out_name = filepath.stem + "_flipped.npz"
        out_path = dir_path / out_name

        with lzma.open(out_path, "wb") as f:
            pickle.dump(augmented_data, f)

        print(f"Saved: {out_path}")
