"""
Pure physics car simulation - EXACT copy of car.py physics.
No visual components, but includes collision detection.
"""
import math
import sys
from pathlib import Path
from panda3d.core import Vec3

# Import collision parameters from config
sys.path.insert(0, str(Path(__file__).parent.parent))
from genetics_algo.config import (
    COLLISION_SAFETY_BUFFER, PHYSICS_SUBSTEPS, CAR_WIDTH,
    MIN_SAFE_DISTANCE, NUM_PROXIMITY_RAYS, PROXIMITY_HALF_ANGLE
)


class CarPhysics:
    """
    Car physics copied EXACTLY from car.py update() method.
    Includes collision detection via Panda3D.
    """

    def __init__(self, position=(0, 1, 0), rotation_y=-90.0, collision_system=None):
        # Position in 3D space (x, y, z)
        self.position = list(position)
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]

        # Rotation (we only care about Y-axis rotation for 2D movement)
        self.rotation_y = rotation_y  # Degrees

        # Physics properties
        self.speed = 0.0
        self.velocity_y = 0
        self.rotation_speed = 0.0

        # Car parameters - EXACT values from sports_car() in car.py:144-156
        self.topspeed = 50.0
        self.minspeed = -15.0
        self.acceleration = 25.0
        self.braking_strenth = 50.0  # Note: keeping typo from original
        self.friction = 1.5
        self.turning_speed = 6.0
        self.max_rotation_speed = 1.6
        self.steering_amount = 9.0

        # Entity scale (default 1.0)
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale_z = 1.0

        # Sphere collider radius (from sports-car.obj bounding sphere)
        self.collision_radius = 2.66

        # Fixed physics timestep (from car.py:286)
        self.dt = 0.1

        # Collision system
        self.collision_system = collision_system

        # Collision tracking (for GA fitness penalty)
        self.collision_occurred = False
        self.collision_count = 0
        self.collision_point = None  # (x, y, z) where collision occurred
        self.collision_normal = None  # Surface normal at collision

    @property
    def world_position(self):
        """Return position as tuple for compatibility."""
        return (self.x, self.y, self.z)

    @property
    def forward(self):
        """
        Calculate forward direction vector based on rotation_y.
        Matches Ursina's forward vector.
        """
        angle_rad = math.radians(self.rotation_y)
        forward_x = math.sin(angle_rad)
        forward_z = math.cos(angle_rad)
        return Vec3(forward_x, 0, forward_z)

    def reset(self, position, angle, speed):
        """Reset car to specific state."""
        self.x = position[0]
        self.y = position[1]
        self.z = position[2]
        self.position = [self.x, self.y, self.z]
        self.rotation_y = angle
        self.speed = speed
        # Reset collision tracking
        self.collision_occurred = False
        self.collision_count = 0

    def update(self, forward=False, backward=False, left=False, right=False, debug_frame=None):
        """
        Update car physics for one timestep with sub-stepping for collision detection.
        Based on car.py:284-413 physics logic, but split into substeps.

        Args:
            forward (bool): Accelerate forward
            backward (bool): Brake/reverse
            left (bool): Steer left
            right (bool): Steer right
            debug_frame (int): If provided, print debug info for this frame
        """
        # Physics sub-stepping: run multiple smaller updates per frame
        # This prevents high-speed tunneling through thin walls
        substeps = PHYSICS_SUBSTEPS
        substep_dt = self.dt / substeps  # Divide timestep by number of substeps

        for substep in range(substeps):
            # Use substep_dt instead of dt for all calculations
            dt = substep_dt

            # === Speed Update (car.py:310-320) ===
            if forward:
                self.speed += self.acceleration * dt
            else:
                # Natural deceleration
                if self.speed > 1:
                    self.speed -= self.friction * 5 * dt
                elif self.speed < -1:
                    self.speed += self.friction * 5 * dt

            # === Braking (car.py:322-330) ===
            if backward:
                if self.speed > 0:
                    self.speed -= self.braking_strenth * dt
                else:
                    self.speed -= self.acceleration * dt

            # === Speed Clamping (car.py:332-336) ===
            if self.speed > self.topspeed:
                self.speed = self.topspeed
            elif self.speed < self.minspeed:
                self.speed = self.minspeed

            # === Rotation Update - Circle-Based Turning (car.py:338-364) ===
            if left or right:
                turn_right = right
                rotation_sign = (1 if turn_right else -1)

                # Max angular speed
                normalized_speed = abs(self.speed / self.topspeed)

                # Rotation radius function (car.py:346-349)
                smallest_radius = 1.5
                biggest_radius = 25
                radius = pow(normalized_speed, 1.5) * (biggest_radius - smallest_radius) + smallest_radius

                # Get travelled distance (car.py:355)
                travelled_dist = abs(self.speed * dt)

                # Project on circle radius & compute angle variation (car.py:357)
                travelled_circle_center_angle = travelled_dist / radius

                # Compute variation in Y & X (car.py:359-360)
                dx = 1 - math.cos(travelled_circle_center_angle)
                dy = math.sin(travelled_circle_center_angle)

                # Calculate angle delta (car.py:362)
                da = math.atan2(dx, dy) / 3.14159 * 180

                # Apply rotation (car.py:364)
                self.rotation_y += da * rotation_sign

            # === Movement with Collision (car.py:366-404) ===
            total_dist_to_move = self.speed * dt

            # Move car with collision detection (car.py:373-398)
            for i in range(2):
                total_dist_to_move = self._move_car(total_dist_to_move, 1 if self.speed > 0 else -1, debug_frame=debug_frame)
                if total_dist_to_move <= 0:
                    break

            # If collision occurred in this substep, stop processing further substeps
            if self.collision_occurred:
                break

        # Check proximity collision AFTER all substeps complete (only if no direct collision yet)
        if not self.collision_occurred and self.collision_system is not None:
            if self._check_proximity_collision():
                self.collision_occurred = True
                self.collision_count += 1

        # Check for backwards movement (going in reverse is not allowed)
        if not self.collision_occurred and self.speed < 0:
            self.collision_occurred = True
            self.collision_count += 1

    def _move_car(self, distance_to_travel, direction, debug_frame=None):
        """
        Move car with collision detection.
        EXACT COPY of move_car() from car.py:373-398.

        Args:
            distance_to_travel (float): Distance to move
            direction (int): 1 for forward, -1 for backward
            debug_frame (int): If provided, print debug info for this frame

        Returns:
            float: Remaining distance to travel (0 if collision or moved successfully)
        """
        if self.collision_system is None:
            # No collision system - just move freely
            forward_vec = self.forward
            self.x += forward_vec.x * distance_to_travel
            self.z += forward_vec.z * distance_to_travel
            self.position = [self.x, self.y, self.z]
            return 0

        # Cast a box to check for collisions (car.py:374)
        forward_vec = self.forward * direction

        # CAR HITBOX INFO
        hitbox_info = {
            'position': (self.x, self.y, self.z),
            'scale_x': self.scale_x,
            'scale_y': self.scale_y,
            'scale_z': self.scale_z,
            'rotation_y': self.rotation_y,
            'speed': self.speed,
            'direction': direction,
            'distance_to_travel': distance_to_travel
        }

        # MULTI-RAY COLLISION DETECTION
        # Cast 3 rays (left, center, right) to detect side collisions during turns
        # This prevents cutting corners too tight and hitting walls in the real game

        boxcast_direction = (forward_vec.x, forward_vec.y, forward_vec.z)  # 3D direction
        boxcast_distance = self.collision_radius + distance_to_travel + COLLISION_SAFETY_BUFFER

        # Calculate perpendicular (right) vector for offset positions
        angle_rad = math.radians(self.rotation_y)
        right_x = math.cos(angle_rad)  # Perpendicular to forward
        right_z = -math.sin(angle_rad)

        # Define 3 ray origins: left, center, right
        half_width = CAR_WIDTH / 2.0
        ray_offsets = [
            (-half_width * right_x, 0, -half_width * right_z),  # Left side
            (0, 0, 0),                                           # Center
            (+half_width * right_x, 0, +half_width * right_z)   # Right side
        ]

        # Cast all 3 rays and find the closest collision
        closest_collision = None
        min_distance = float('inf')

        for offset in ray_offsets:
            ray_origin = (
                self.x + offset[0],
                self.y + offset[1],
                self.z + offset[2]
            )

            collision = self.collision_system.boxcast(
                origin=ray_origin,
                direction=boxcast_direction,
                thickness=(0.1, 0.1),
                distance=boxcast_distance
            )

            # Track the closest collision
            if collision.hit and collision.distance < min_distance:
                min_distance = collision.distance
                closest_collision = collision

        # Use the closest collision (or no collision if none hit)
        if closest_collision is None:
            # No collisions detected - create a "no hit" result
            from rallyrobopilot_novisual.collision import CollisionResult
            front_collision = CollisionResult(
                hit=False,
                distance=boxcast_distance + 999,
                point=None,
                normal=None
            )
        else:
            front_collision = closest_collision

        # COLLISION THRESHOLD
        # Use collision_radius to match boxcast distance
        collision_threshold = self.collision_radius + distance_to_travel + COLLISION_SAFETY_BUFFER

        # DEBUG OUTPUT
        if debug_frame is not None:
            print(f"\n{'='*80}")
            print(f"FRAME {debug_frame} COLLISION DEBUG (MULTI-RAY)")
            print(f"{'='*80}")
            print(f"CAR HITBOX:")
            print(f"  Position: ({self.x:.2f}, {self.y:.2f}, {self.z:.2f})")
            print(f"  Car Width: {CAR_WIDTH:.2f}")
            print(f"  Collision Radius: {self.collision_radius}")
            print(f"  Rotation Y: {self.rotation_y:.2f}°")
            print(f"  Speed: {self.speed:.2f} units/s")
            print(f"  Forward vector: ({forward_vec.x:.3f}, {forward_vec.y:.3f}, {forward_vec.z:.3f})")
            print(f"\nMULTI-RAY BOXCAST:")
            print(f"  3 rays cast: LEFT, CENTER, RIGHT")
            print(f"  Direction: ({boxcast_direction[0]:.3f}, {boxcast_direction[1]:.3f}, {boxcast_direction[2]:.3f})")
            print(f"  Cast Distance: {boxcast_distance:.2f}")
            print(f"\nCOLLISION RESULT:")
            print(f"  Closest obstacle: {front_collision.distance:.2f}")
            print(f"  Collision threshold: {collision_threshold:.2f}")
            print(f"  Will collide: {front_collision.distance < collision_threshold}")
            if front_collision.hit:
                print(f"  Hit object: {front_collision.object_name}")
                print(f"  Hit point: ({front_collision.world_point.x:.2f}, {front_collision.world_point.y:.2f}, {front_collision.world_point.z:.2f})")
                print(f"  Hit normal: ({front_collision.world_normal.x:.3f}, {front_collision.world_normal.y:.3f}, {front_collision.world_normal.z:.3f})")
            print(f"{'='*80}\n")

        # Detect collision (car.py:377)
        if front_collision.distance < collision_threshold:
            # Collision detected! Track it for GA fitness penalty
            self.collision_occurred = True
            self.collision_count += 1

            # Store collision point for visualization
            if hasattr(front_collision, 'world_point') and front_collision.world_point:
                if hasattr(front_collision.world_point, 'x'):
                    self.collision_point = (front_collision.world_point.x,
                                           front_collision.world_point.y,
                                           front_collision.world_point.z)
                else:
                    # Ursina Vec3 format
                    self.collision_point = (front_collision.world_point[0],
                                           front_collision.world_point[1],
                                           front_collision.world_point[2])

            free_dist = front_collision.distance - self.scale_x + distance_to_travel

            # Cancel speed going directly into the obstacle (car.py:381-382)
            forward_vec_normalized = self.forward * direction

            # Handle different collision result types - EXTRACT FULL 3D NORMAL
            if hasattr(front_collision.world_normal, 'x'):
                normal_x = front_collision.world_normal.x
                normal_y = front_collision.world_normal.y
                normal_z = front_collision.world_normal.z
            else:
                # Ursina Vec3
                normal_x = front_collision.world_normal[0]
                normal_y = front_collision.world_normal[1]
                normal_z = front_collision.world_normal[2]

            # Store collision normal for visualization
            self.collision_normal = (normal_x, normal_y, normal_z)

            # CRITICAL FIX: Use FULL 3D dot product (not 2D!)
            # This matches the original game's 3D collision physics
            dot_product = (forward_vec_normalized.x * normal_x +
                          forward_vec_normalized.y * normal_y +
                          forward_vec_normalized.z * normal_z)

            # CRITICAL FIX: Calculate FULL 3D reflection vector (not 2D!)
            # Original: next_forward = self.forward - (self.forward.dot(normal)) * normal
            next_forward = Vec3(
                forward_vec_normalized.x - dot_product * normal_x,
                forward_vec_normalized.y - dot_product * normal_y,
                forward_vec_normalized.z - dot_product * normal_z
            )

            # Lose half speed on collision (car.py:382)
            self.speed = self.speed * (0.5 + 0.5 * dot_product)

            # Update rotation based on collision normal (car.py:384)
            self.rotation_y = math.atan2(next_forward.x, next_forward.z) / 3.14159 * 180

            # Move car away from obstacle (car.py:388-390)
            OBSTACLE_DISPLACEMENT_MARGIN = 1
            self.x += normal_x * OBSTACLE_DISPLACEMENT_MARGIN
            self.z += normal_z * OBSTACLE_DISPLACEMENT_MARGIN

            return 0

        else:
            # No collision - move freely (car.py:395-396)
            forward_vec = self.forward
            self.x += forward_vec.x * distance_to_travel
            self.z += forward_vec.z * distance_to_travel
            self.position = [self.x, self.y, self.z]
            return 0

    def _check_proximity_collision(self):
        """
        Check if car is too close to any obstacles using proximity sensors.
        Casts multiple rays in an arc around the car (like the original game's MultiRaySensor).

        Returns:
            bool: True if any sensor detects obstacle closer than MIN_SAFE_DISTANCE
        """
        # Cast rays in an arc (from -PROXIMITY_HALF_ANGLE to +PROXIMITY_HALF_ANGLE)
        for i in range(NUM_PROXIMITY_RAYS):
            # Calculate angle for this ray relative to car's forward direction
            if NUM_PROXIMITY_RAYS == 1:
                angle_offset = 0  # Single ray points forward
            else:
                # Spread rays evenly across the arc
                angle_offset = -PROXIMITY_HALF_ANGLE + (2 * PROXIMITY_HALF_ANGLE / (NUM_PROXIMITY_RAYS - 1)) * i

            # Calculate ray direction in world space
            ray_angle = self.rotation_y + angle_offset
            ray_angle_rad = math.radians(ray_angle)
            ray_dir_x = math.sin(ray_angle_rad)
            ray_dir_z = math.cos(ray_angle_rad)

            # Cast ray from car position
            ray_origin = (self.x, self.y, self.z)
            ray_direction = (ray_dir_x, 0, ray_dir_z)

            # Cast with long distance to detect obstacles ahead
            collision = self.collision_system.boxcast(
                origin=ray_origin,
                direction=ray_direction,
                thickness=(0.1, 0.1),
                distance=MIN_SAFE_DISTANCE * 2  # Cast twice the safe distance to be sure
            )

            # Check if obstacle is too close
            if collision.hit and collision.distance < MIN_SAFE_DISTANCE:
                # Too close to obstacle! Treat as proximity collision
                # Store collision point for visualization
                if hasattr(collision, 'world_point') and collision.world_point:
                    if hasattr(collision.world_point, 'x'):
                        self.collision_point = (collision.world_point.x,
                                               collision.world_point.y,
                                               collision.world_point.z)
                    else:
                        self.collision_point = (collision.world_point[0],
                                               collision.world_point[1],
                                               collision.world_point[2])
                return True

        # All sensors clear - safe distance maintained
        return False

    def get_state(self):
        """Get current state as dict."""
        return {
            'position': (self.x, self.y, self.z),
            'angle': self.rotation_y,
            'speed': self.speed
        }

    def __repr__(self):
        return f"CarPhysics(pos=({self.x:.1f}, {self.y:.1f}, {self.z:.1f}), angle={self.rotation_y:.1f}, speed={self.speed:.1f})"