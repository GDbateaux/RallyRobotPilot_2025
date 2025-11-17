from rallyrobopilot import prepare_game_app, RemoteController
from flask import Flask, request, jsonify
from threading import Thread
from panda3d.core import ClockObject

from genetics_algo.config import (
    TRACK_NAME
)


# Setup Flask

flask_app = Flask(__name__)
flask_thread = Thread(target=flask_app.run, kwargs={'host': "0.0.0.0", 'port': 5000})
print("Flask server running on port 5000")
flask_thread.start()

# app, car = prepare_game_app("VisualTrack/track_circuit2_metadata.json")
app, car = prepare_game_app(f"{TRACK_NAME}/track_metadata.json")


clock = ClockObject.getGlobalClock()
clock.setMode(ClockObject.MLimited)
clock.setFrameRate(10)


remote_controller = RemoteController(car = car, connection_port=7654, flask_app=flask_app)
app.run()