if __name__ == "__main__":
    from pathlib import Path
    import pickle
    import lzma
    import csv

    root_dir = Path(__file__).resolve().parent.parent
    records_dir = root_dir / "records"
    all_data = []

    for filepath in records_dir.glob("*.npz"):
        with lzma.open(filepath, "rb") as file:
            data = pickle.load(file)
            print("Reading", filepath.name, "length:", len(data))
        all_data.extend(data)

    csv_path = root_dir / "records_merged.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        first = all_data[0]
        n_rays = len(getattr(first, "raycast_distances", []))
        n_ctrls = len(getattr(first, "current_controls", []))
        header = (
            [f"pos_{i}" for i in range(3)] +
            ["speed", "angle"] +
            [f"ray_{i}" for i in range(n_rays)] +
            ["Forward", "Backward", "Left", "Right"]
        )
        writer.writerow(header)

        total_written = 0
        for s in all_data:
            pos = list(getattr(s, "car_position", (0, 0, 0)))
            speed = getattr(s, "car_speed", 0)
            angle = getattr(s, "car_angle", 0)
            rays = list(getattr(s, "raycast_distances", []))
            ctrls = list(getattr(s, "current_controls", []))
            writer.writerow(pos + [speed, angle] + rays + ctrls)
            total_written += 1

            mirrored_pos = [pos[0], pos[1], -pos[2]]
            mirrored_angle = -angle
            mirrored_rays = list(reversed(rays))
            mirrored_ctrls = [
                ctrls[0],
                ctrls[1],
                ctrls[3],
                ctrls[2],
            ]

            writer.writerow(mirrored_pos + [speed, mirrored_angle] + mirrored_rays + mirrored_ctrls)
            total_written += 1

    print(f"\n💾 Data saved in : {csv_path}  ({total_written} lines)")
