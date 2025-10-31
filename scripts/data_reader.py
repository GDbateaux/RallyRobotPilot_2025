

if __name__ == "__main__":
    import numpy
    import pickle
    import lzma


    with lzma.open("old_data.bak", "rb") as file:
        data = pickle.load(file)

        print("Read", len(data), "snapshotwas")
        print(data[0].image)
        print([e.current_controls for e in data])
