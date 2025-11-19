import timeit

import setuptools
from ursina import *
from ursina import curve
from .particles import Particles, TrailRenderer
from math import pow, atan2
import json

sign = lambda x: -1 if x < 0 else (1 if x > 0 else 0)
Text.default_resolution = 1080 * Text.size
DT = 0.04

class Car(Entity):
    def __init__(self, position = (0, 0, 4), rotation = (0, 0, 0), topspeed = 30, acceleration = 0.35, braking_strength = 30, friction = 1.5, camera_speed = 8):
        super().__init__(
            model = "assets/cars/sports-car.obj",
            texture = "assets/cars/garage/sports-car/sports-red.png",
            collider = "sphere",
            position = position,
            rotation = rotation,
        )

        # Controls
        self.controls = "wasd"

        # Car's values
        self.speed = 0
        self.velocity_y = 0
        self.rotation_speed = 0
        self.max_rotation_speed = 1.6
        self.steering_amount = 8
        self.topspeed = topspeed
        self.braking_strenth = braking_strength
        self.camera_speed = camera_speed
        self.acceleration = acceleration
        self.friction = friction
        self.turning_speed = 5
        self.pivot_rotation_distance = 1

        self.reset_position = (0, 0, 0)
        self.reset_rotation = (0, 0, 0)

        # Camera Follow
        self.camera_angle = "top"
        self.camera_offset = (0, 30, -35) # <-- Stuff to change to change the camera distance
        self.camera_rotation = 40
        self.camera_follow = False
        self.change_camera = False
        self.c_pivot = Entity()
        self.camera_pivot = Entity(parent = self.c_pivot, position = self.camera_offset)

        # Pivot for drifting
        self.pivot = Entity()
        self.pivot.position = self.position
        self.pivot.rotation = self.rotation
        self.drifting = False

        # Car Type
        self.car_type = "sports"

        # Particles
        self.particle_time = 0
        self.particle_amount = 0.07 # The lower, the more
        self.particle_pivot = Entity(parent = self)
        self.particle_pivot.position = (0, -1, -2)

        # TrailRenderer
        self.trail_pivot = Entity(parent = self, position = (0, -1, 2))

        self.trail_renderer1 = TrailRenderer(parent = self.particle_pivot, position = (0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer2 = TrailRenderer(parent = self.particle_pivot, position = (-0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer3 = TrailRenderer(parent = self.trail_pivot, position = (0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        self.trail_renderer4 = TrailRenderer(parent = self.trail_pivot, position = (-0.8, -0.2, 0), color = color.black, alpha = 0, thickness = 7, length = 200)
        
        self.trails = [self.trail_renderer1, self.trail_renderer2, self.trail_renderer3, self.trail_renderer4]
        self.start_trail = True

        # Collision
        self.copy_normals = False
        self.hitting_wall = False


        self.track = None

        # Graphics
        self.graphics = "fancy"

        # Stopwatch/Timer
        self.timer_running = False
        self.count = 0.0

        self.last_count = self.count
        self.reset_count = 0.0
        self.timer = Text(text = "", origin = (0, 0), size = 0.05, scale = (1, 1), position = (-0.7, 0.43))

        self.laps_text = Text(text = "", origin = (0, 0), size = 0.05, scale = (1.1, 1.1), position = (0, 0.43))
        self.reset_count_timer = Text(text = str(round(self.reset_count, 1)), origin = (0, 0), size = 0.05, scale = (1, 1), position = (-0.7, 0.43))
        
        self.timer.disable()

        self.laps_text.disable()
        self.reset_count_timer.disable()

        self.gamemode = "race"
        self.start_time = False
        self.laps = 0
        self.laps_hs = 0
        self.anti_cheat = 1

        # Bools
        self.driving = False
        self.braking = False

        # Multiplayer
        self.multiplayer = False
        self.multiplayer_update = False
        self.server_running = False

        # Shows whether you are connected to a server or not
        self.connected_text = True
        self.disconnected_text = True

        # Camera shake
        self.shake_amount = 0.1
        self.can_shake = False
        self.camera_shake_option = True

        self.username_text = "Username"

        self.model_path = str(self.model).replace("render/scene/car/", "")

        invoke(self.update_model_path, delay = 1)

        self.multiray_sensor = None

    def set_track(self, track):
        self.track = track
        self.reset_position = track.car_default_reset_position
        self.reset_orientation = track.car_default_reset_orientation
        self.position = self.reset_position
        self.rotation_y = self.reset_orientation[1]

    def sports_car(self):
        self.car_type = "sports"
        self.model = "assets/cars/sports-car.obj"
        self.texture = "assets/cars/garage/sports-car/sports-red.png"
        self.topspeed = 50
        self.minspeed = -15
        self.acceleration = 25
        self.braking_strenth = 50
        self.turning_speed = 6
        self.max_rotation_speed = 1.6
        self.steering_amount = 9
        self.particle_pivot.position = (0, -1, -1.5)
        self.trail_pivot.position = (0, -1, 1.5)

    def update_camera(self):
        if self.camera_follow:
            if self.change_camera:
                camera.rotation_x = 35
                self.camera_rotation = 40
            self.camera_offset = (0, 60, -70)
            self.camera_speed = 4
            self.change_camera = False
            #camera.rotation_x = self.camera_rotation
            camera.world_position = self.camera_pivot.world_position
            camera.world_rotation_y = self.world_rotation_y

    def check_respawn(self):
        # Respawn
        if held_keys["g"]:
            self.reset_car()

        if held_keys["v"]:
            self.multiray_sensor.set_enabled_rays(not self.multiray_sensor.enabled)

        # Reset the car's position if y value is less than -100
        if self.y <= -100:
            self.reset_car()

        # Reset the car's position if y value is greater than 300
        if self.y >= 300:
            self.reset_car()

    def display_particles(self):
        # Particles
        self.particle_time += DT
        if self.particle_time >= self.particle_amount:
            self.particle_time = 0
            self.particles = Particles(self, self.particle_pivot.world_position - (0, 1, 0))
            self.particles.destroy(1)

    def hand_brake(self):
        # Hand Braking
        if held_keys["space"]:
            if self.rotation_speed < 0:
                self.rotation_speed -= 3 * DT
            elif self.rotation_speed > 0:
                self.rotation_speed += 3 * DT
            self.speed -= 20 * DT

    def compute_steering(self):
        # Steering
        self.rotation_y += self.rotation_speed * 50 * DT

        # The car's linear momentum decreases the rotation.
        if self.rotation_speed > 0:
            self.rotation_speed -= self.speed / 6 * DT
        elif self.rotation_speed < 0:
            self.rotation_speed += self.speed / 6 * DT

        # Can only turn if |speed| > 0.5
        if self.speed > 0.5 or self.speed < -0.5:
            if held_keys[self.controls[1]] or held_keys["left arrow"]:
                self.rotation_speed -= self.steering_amount * DT

                # Turning decreases our speed.
                if self.speed > 1:
                    self.speed -= self.turning_speed * DT
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * DT

            elif held_keys[self.controls[3]] or held_keys["right arrow"]:
                self.rotation_speed += self.steering_amount * DT
                if self.speed > 1:
                    self.speed -= self.turning_speed * DT
                elif self.speed < 0:
                    self.speed += self.turning_speed / 5 * DT
            # If no keys pressed, the rotation speed goes down.
            else:
                if self.rotation_speed > 0:
                    self.rotation_speed -= 5 * DT
                elif self.rotation_speed < 0:
                    self.rotation_speed += 5 * DT
        else:
            self.rotation_speed = 0

    def cap_kinetic_parameters(self):
        # Cap the speed
        if self.speed >= self.topspeed:
            self.speed = self.topspeed
        if self.speed <= -15:
            self.speed = -15
        if self.speed <= 0:
            self.pivot.rotation_y = self.rotation_y

        # Cap the steering
        if self.rotation_speed >= self.max_rotation_speed:
            self.rotation_speed = self.max_rotation_speed
        if self.rotation_speed <= -self.max_rotation_speed:
            self.rotation_speed = -self.max_rotation_speed
            
        # Cap the camera rotation
        if self.camera_rotation >= 40:
            self.camera_rotation = 40
        elif self.camera_rotation <= 30:
            self.camera_rotation = 30


    def update_vertical_position(self, y_ray, movementY):
        # Check if car is hitting the ground
        if self.visible:
            if y_ray.distance <= self.scale_y * 1.7 + abs(movementY):
                self.velocity_y = 0
                # Check if hitting a wall or steep slope
                if y_ray.world_normal.y > 0.7 and y_ray.world_point.y - self.world_y < 0.5:
                    # Set the y value to the ground's y value
                    self.y = y_ray.world_point.y + 1.4
                    self.hitting_wall = False
                else:
                    # Car is hitting a wall
                    self.hitting_wall = True

                if self.copy_normals:
                    self.ground_normal = self.position + y_ray.world_normal
                else:
                    self.ground_normal = self.position + (0, 180, 0)
            else:
                self.y += movementY * 50 * DT
                self.velocity_y -= 50 * DT


    def update(self):
        # Exit if esc pressed.
        if held_keys["escape"]:
            quit()

        self.check_respawn()

        #   Process inputs & update speed
        if held_keys[self.controls[0]] or held_keys["up arrow"]:
            self.speed += self.acceleration * DT
            self.driving = True

            self.display_particles()
        else:
            self.driving = False
            if self.speed > 1:
                self.speed -= self.friction * 5 * DT
            elif self.speed < -1:
                self.speed += self.friction * 5 * DT

        # Braking
        if held_keys[self.controls[2] or held_keys["down arrow"]]:
            if self.speed > 0:
                self.speed -= self.braking_strenth * DT
            else:
                self.speed -= self.acceleration * DT
            self.braking = True
        else:
            self.braking = False

        #   Check physical constrains
        if self.speed > self.topspeed:
            self.speed = self.topspeed
        elif self.speed < self.minspeed:
            self.speed = self.minspeed

        if held_keys[self.controls[1]] or held_keys["left arrow"] or held_keys[self.controls[3]] or held_keys["right arrow"]:
            turn_right = held_keys[self.controls[3]] or held_keys["right arrow"]
            rotation_sign = (1 if turn_right else -1)

            #   Max angular speed
            normalized_speed = abs(self.speed / self.topspeed)
            #   function to map unit speed (between 0 and max speed) to a rotation coefficient space.
            #   Rotation radius is function of speed
            def rotation_radius(normalized_speed):
                smallest_radius = 1.5
                biggest_radius = 25
                return pow(normalized_speed, 1.5) * (biggest_radius-smallest_radius) + smallest_radius

            #   Get rotation radius
            radius = rotation_radius(normalized_speed)

            #   Get travelled distance
            travelled_dist = abs(self.speed * DT)
            #   Project on circle radius & compute angle variation seen from the center of the circle
            travelled_circle_center_angle = travelled_dist / radius
            #   Compute variation in Y & X
            dx = 1 - cos(travelled_circle_center_angle)
            dy = sin(travelled_circle_center_angle)

            da = atan2(dx, dy) / 3.14159 * 180

            self.rotation_y += da * rotation_sign

        #   Integrate speed into movement
        total_dist_to_move = self.speed * DT

        #   Check collision via recast


        #   Return residual distance to travel and residual speed.
        def move_car(distance_to_travel, direction):
            front_collision = boxcast(origin = self.world_position, direction = self.forward * direction, thickness = (0.1, 0.1), distance = self.scale_x + distance_to_travel, ignore = [self, ])

            #   Detect collision
            if front_collision.distance < self.scale_x + distance_to_travel:
                free_dist = front_collision.distance - self.scale_x + distance_to_travel

                #   cancel speed going directly into the obstacle
                next_forward = self.forward - (self.forward.dot(front_collision.world_normal)) * front_collision.world_normal
                self.speed = self.speed * (0.5 + 0.5 * (self.forward.dot(front_collision.world_normal))) # Loose half speed on collision and some depending on the angle

                self.rotation_y = atan2(next_forward[0], next_forward[2]) / 3.14159 * 180
                dist_left_to_travel = distance_to_travel - free_dist

                #   Move car away from obstacle to prevent overlap due to *¦@+!? physics system
                OBSTACLE_DISPLACEMENT_MARGIN = 1
                self.x += (front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN).x
                self.z += (front_collision.world_normal * OBSTACLE_DISPLACEMENT_MARGIN).z

                return 0

            else:
                self.x += self.forward[0] * distance_to_travel
                self.z += self.forward[2] * distance_to_travel

                return 0

        for i in range(2):
            total_dist_to_move = move_car(total_dist_to_move, 1 if self.speed > 0 else -1)

            if total_dist_to_move <= 0:
                break

        self.c_pivot.position = self.position
        self.c_pivot.rotation_y = self.rotation_y
        self.update_camera()

        self.pivot.position = self.position

    def reset_car(self):
        """
        Resets the car
        """
        #   Project car directly on ground when resetting
        self.position = self.reset_position
        #y_ray = raycast(origin = self.reset_position, direction = (0,-1,0), ignore = [self,])
        #self.y = y_ray.world_point.y + 1.4
        print(self.reset_orientation)
        self.rotation_y = self.reset_orientation[1]

        print("reseting at", str(self.position), " --> ", self.rotation_y)

        camera.world_rotation_y = self.rotation_y
        self.speed = 0
        self.velocity_y = 0
        self.timer_running = False
        for trail in self.trails:
            if trail.trailing:
                trail.end_trail()
        self.start_trail = True

    def simple_intersects(self, entity):
        """
        A faster AABB intersects for detecting collision with
        simple objects, doesn't take rotation into account
        """
        minXA = self.x - self.scale_x
        maxXA = self.x + self.scale_x
        minYA = self.y - self.scale_y + (self.scale_y / 2)
        maxYA = self.y + self.scale_y - (self.scale_y / 2)
        minZA = self.z - self.scale_z
        maxZA = self.z + self.scale_z

        minXB = entity.x - entity.scale_x + (entity.scale_x / 2)
        maxXB = entity.x + entity.scale_x - (entity.scale_x / 2)
        minYB = entity.y - entity.scale_y + (entity.scale_y / 2)
        maxYB = entity.y + entity.scale_y - (entity.scale_y / 2)
        minZB = entity.z - entity.scale_z + (entity.scale_z / 2)
        maxZB = entity.z + entity.scale_z - (entity.scale_z / 2)
        
        return (
            (minXA <= maxXB and maxXA >= minXB) and
            (minYA <= maxYB and maxYA >= minYB) and
            (minZA <= maxZB and maxZA >= minZB)
        )

    def reset_timer(self):
        """
        Resets the timer
        """
        self.count = self.reset_count
        self.timer.enable()
        self.reset_count_timer.disable()

    def animate_text(self, text, top = 1.2, bottom = 0.6):
        """
        Animates the scale of text
        """
        if self.gamemode != "drift":
            if self.last_count > 1:
                text.animate_scale((top, top, top), curve = curve.out_expo)
                invoke(text.animate_scale, (bottom, bottom, bottom), delay = 0.2)
        else:
            text.animate_scale((top, top, top), curve = curve.out_expo)
            invoke(text.animate_scale, (bottom, bottom, bottom), delay = 0.2)

    def update_model_path(self):
        """
        Updates the model's file path for multiplayer
        """
        self.model_path = str(self.model).replace("render/scene/car/", "")
        invoke(self.update_model_path, delay = 3)

# Class for copying the car's position, rotation for multiplayer
class CarRepresentation(Entity):
    def __init__(self, car, position = (0, 0, 0), rotation = (0, 65, 0)):
        super().__init__(
            parent = scene,
            model = "assets/cars/sports-car.obj",
            texture = "assets/cars/garage/sports-car/sports-red.png",
            position = position,
            rotation = rotation,
            scale = (1, 1, 1)
        )

        self.model_path = str(self.model).replace("render/scene/car_representation/", "")

        self.text_object = None


# Username shown above the car
class CarUsername(Text):
    def __init__(self, car):
        super().__init__(
            parent = car,
            text = "Guest",
            y = 3,
            scale = 30,
            color = color.white,
            billboard = True
        )
    
        self.username_text = "Guest"

    def update(self):
        self.text = self.username_text
