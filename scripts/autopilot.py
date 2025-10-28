import torch
import cv2
import numpy as np

from PyQt6 import QtWidgets

from scripts.data_collector import DataCollectionUI
from src.model import DrivingCNN
from pathlib import Path

"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class ExampleNNMsgProcessor:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        
        input_shape = (3, 150, 200)
        self.model = DrivingCNN(input_shape).to(self.device)
    
        model_path = Path(__file__).parent.parent / "data/models/driving_cnn.pt"
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def preprocess_image(self, img_bgr: np.ndarray) -> torch.Tensor:
        img_resized = cv2.resize(img_bgr, (200, 150)).astype(np.float32)
        img_resized = img_resized / 255.0
        img_chw = np.transpose(img_resized, (2, 0, 1))
        tensor = torch.from_numpy(img_chw).unsqueeze(0)
        tensor = tensor.to(self.device)
        return tensor

    def nn_infer(self, message):
        frame = message.image

        inp = self.preprocess_image(frame)

        with torch.no_grad():
            pred = self.model(inp)
            pred = pred.squeeze(0).cpu()

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

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()