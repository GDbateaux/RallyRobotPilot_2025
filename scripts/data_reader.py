if __name__ == "__main__":
    import numpy
    import pickle
    import lzma


    with lzma.open("simple_track/record_0.npz", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        print([e.current_controls for e in data])
