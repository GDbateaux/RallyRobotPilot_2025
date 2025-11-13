"""
Physics simulation loop - NO Ursina task manager.
Pure Python loop for running physics updates.
"""


class SimulationLoop:
    """
    Replacement for Ursina's task manager.

    Runs physics updates in a simple Python loop - no rendering, no window.
    """

    def __init__(self, controller):
        """
        Args:
            controller (DirectController): Controller managing the car
        """
        self.controller = controller
        self.running = False

    def step(self):
        """
        Execute one physics timestep.

        This replaces app.taskMgr.step() from Ursina.
        """
        # Update car physics with current controls
        self.controller.update_car()

    def run_steps(self, num_steps):
        """
        Run multiple physics steps.

        Args:
            num_steps (int): Number of steps to execute
        """
        for _ in range(num_steps):
            self.step()