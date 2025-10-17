import torch
import numpy as np

from pathlib import Path
from model import NeuralNetwork

from PyQt6 import QtWidgets

from data_collector import DataCollectionUI
from rallyrobopilot.sensing_message import SensingSnapshot
"""
This file is provided as an example of what a simplistic controller could be done.
It simply uses the DataCollectionUI interface zo receive sensing_messages and send controls.

/!\ Be warned that if the processing time of NNMsgProcessor.process_message is superior to the message reception period, a lag between the images processed and commands sent.
One might want to only process the last sensing_message received, etc. 
Be warned that this could also cause crash on the client side if socket sending buffer overflows

/!\ Do not work directly in this file (make a copy and rename it) to prevent future pull from erasing what you write here.
"""


class ExampleNNMsgProcessor:
    def __init__(self):
        self.model = NeuralNetwork()
        model_path = Path(__file__).resolve().parent.parent / "rally_model.pth"
        checkpoint = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.scaler = checkpoint.get("scaler", None)

        th = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        self.thresholds = torch.tensor(th, dtype=torch.float32)

        self.actions = ["forward", "back", "left", "right"]
        self.state = {name: False for name in self.actions}

    def nn_infer(self, message):
        rays = torch.tensor(message.raycast_distances, dtype=torch.float32)
        speed = torch.tensor([message.car_speed], dtype=torch.float32)
        x = torch.cat([speed, rays])
        x_np = x.unsqueeze(0).numpy()
        x_np = self.scaler.transform(x_np)
        x = torch.tensor(x_np, dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(x)
        probs = torch.sigmoid(logits).squeeze(0)

        actions = ["forward", "back", "left", "right"]
        threshold = 0.5

        decisions = [p > threshold for p in probs]

        if decisions[0] and decisions[1]:
            decisions[0] = probs[0] > probs[1]
            decisions[1] = probs[1] > probs[0]
        if decisions[2] and decisions[3]:
            decisions[2] = probs[2] > probs[3]
            decisions[3] = probs[3] > probs[2]
        
        commands = [(actions[i], d) for i, d in enumerate(decisions)]
        return commands

    def process_message(self, message, data_collector):
        commands = self.nn_infer(message)
        for command, start in commands:
            data_collector.onCarControlled(command, start)

if  __name__ == "__main__":
    import sys
    def except_hook(cls, exception, traceback):
        sys._excepthook_(cls, exception, traceback)
    sys.excepthook = except_hook

    app = QtWidgets.QApplication(sys.argv)

    nn_brain = ExampleNNMsgProcessor()
    data_window = DataCollectionUI(nn_brain.process_message)
    data_window.show()

    app.exec()
