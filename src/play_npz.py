import lzma
import pickle
import cv2
from pathlib import Path


def play_recording(path, delay_ms=30):
    path = Path(path)

    with lzma.open(path, "rb") as f:
        data = pickle.load(f)

    print(f"{len(data)} frames dans {path}")

    for i, e in enumerate(data):
        frame = e.image

        cv2.imshow("Replay", frame)
        key = cv2.waitKey(delay_ms) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    play_recording("data/simple_track_2/record_0_flipped.npz")
