import torch
from collections import deque

from PyQt6 import QtWidgets

from scripts.data_collector import DataCollectionUI
from src.model import DrivingCNN
from pathlib import Path
from src.utils import preprocess_image
from src.config import N_FRAMES, IMAGE_RESIZED_DIMENSIONS

"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class CNNMsgProcessor:
    def __init__(self, device="cpu", n_frames=N_FRAMES):
        self.device = torch.device(device)
        self.n_frames = n_frames

        self.frame_buffer = deque(maxlen=n_frames)
        
        input_shape = (3 * n_frames, IMAGE_RESIZED_DIMENSIONS[1], IMAGE_RESIZED_DIMENSIONS[0])
        self.model = DrivingCNN(input_shape).to(self.device)
    
        model_path = Path(__file__).parent.parent / "data/models/driving_cnn.pt"
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def nn_infer(self, message):
        frame = message.image

        img = preprocess_image(frame)
        while len(self.frame_buffer) < self.n_frames:
            self.frame_buffer.append(img)
        self.frame_buffer.append(img)

        frames = list(self.frame_buffer)
        stacked = torch.stack(frames, dim=0)
        stacked = stacked.view(-1, *stacked.shape[2:])
        inp = stacked.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(inp)
            pred = pred.squeeze(0).cpu()
            pred = torch.sigmoid(pred)

        actions = ["forward", "back", "left", "right"]
        threshold = 0.5
        print(pred)

        commands = [p > threshold for p in pred]

        if commands[0] and commands[1]:
            commands[0] = pred[0] > pred[1]
            commands[1] = pred[1] > pred[0]
        if commands[2] and commands[3]:
            commands[2] = pred[2] > pred[3]
            commands[3] = pred[3] > pred[2]

        commands = [(actions[i], d) for i, d in enumerate(commands)]
        print(commands)
        return commands

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)

        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys.__excepthook__(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = CNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()