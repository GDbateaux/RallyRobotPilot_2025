"""
Collision detection using Panda3D headless with proper mesh collision setup.
"""

from panda3d.core import (
    CollisionTraverser, CollisionNode, CollisionHandlerQueue,
    CollisionRay, loadPrcFileData, BitMask32, GeomNode
)
from direct.showbase.ShowBase import ShowBase
import json
from pathlib import Path


class CollisionSystem:
    """
    Headless collision detection system using Panda3D.
    Properly sets up mesh collision geometry.
    """

    def __init__(self, track_path="SimpleTrack/track_metadata.json"):
        """
        Initialize collision system with track geometry.

        Args:
            track_path (str): Path to track metadata JSON
        """
        # Configure Panda3D for headless operation
        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "audio-library-name null")

        # Initialize minimal Panda3D
        self.base = ShowBase(windowType='none')
        self.loader = self.base.loader

        # Collision traverser and handler
        self.traverser = CollisionTraverser()
        self.handler = CollisionHandlerQueue()

        # Load track collision geometry
        self._load_track(track_path)

        print("✓ Collision system initialized (headless)")

    def _load_track(self, track_path):
        """Load track mesh and enable collision on it."""
        # Load track metadata
        root_dir = Path(__file__).resolve().parent.parent
        if ".json" not in track_path:
            metadata_path = root_dir / f"assets/{track_path}/track_metadata.json"
        else:
            metadata_path = root_dir / f"assets/{track_path}"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Get track folder
        track_folder = metadata_path.parent

        # Load track model
        track_model = metadata["track_model"]
        track_model_path = track_folder / track_model

        print(f"Loading collision mesh from: {track_model_path}")

        # Load the track mesh
        self.track_node = self.loader.loadModel(str(track_model_path))

        if self.track_node.isEmpty():
            raise RuntimeError(f"Failed to load track model: {track_model_path}")

        # Apply track transform
        origin_position = metadata["origin_position"]
        origin_rotation = metadata["origin_rotation"]
        origin_scale = metadata["origin_scale"]

        self.track_node.setPos(origin_position[0], origin_position[1], origin_position[2])
        self.track_node.setHpr(origin_rotation[0], origin_rotation[1], origin_rotation[2])
        # FIX: Flip X-axis to match Ursina coordinate system (-X scale)
        self.track_node.setScale(-origin_scale[0], origin_scale[1], origin_scale[2])

        # CRITICAL: Enable collision on the visible geometry
        # This tells Panda3D to treat the mesh as collidable
        self.track_node.setCollideMask(BitMask32.allOn())

        # Tag with object type for identification
        self.track_node.setTag("object_type", "Track.obj")

        # Flatten for better collision performance
        self.track_node.flattenStrong()

        # Reparent to render so collision system can find it
        self.track_node.reparentTo(self.base.render)

        print(f"  ✓ Track collision enabled")

        # Load obstacles
        self.obstacle_nodes = []
        for obstacle in metadata.get("obstacles", []):
            obstacle_model = obstacle["model"]
            obstacle_path = track_folder / obstacle_model

            print(f"Loading obstacle collision mesh: {obstacle_path}")
            obstacle_node = self.loader.loadModel(str(obstacle_path))

            if not obstacle_node.isEmpty():
                obstacle_node.setPos(origin_position[0], origin_position[1], origin_position[2])
                obstacle_node.setHpr(origin_rotation[0], origin_rotation[1], origin_rotation[2])
                # FIX: Flip X-axis to match Ursina coordinate system (-X scale)
                obstacle_node.setScale(-origin_scale[0], origin_scale[1], origin_scale[2])

                # Enable collision on obstacles
                obstacle_node.setCollideMask(BitMask32.allOn())

                # Tag with object type for identification
                obstacle_node.setTag("object_type", obstacle_model)

                obstacle_node.flattenStrong()
                obstacle_node.reparentTo(self.base.render)

                self.obstacle_nodes.append(obstacle_node)
                print(f"  ✓ Obstacle collision enabled: {obstacle_model}")

        print(f"✓ Loaded collision geometry: track + {len(self.obstacle_nodes)} obstacles")

    def boxcast(self, origin, direction, thickness=(0.1, 0.1), distance=1.0, ignore=None):
        """
        Cast a ray to check for collisions with track geometry.

        Args:
            origin: (x, y, z) tuple - starting position
            direction: (x, y, z) tuple - direction vector (will be normalized)
            thickness: (width, height) tuple - not used, for compatibility
            distance: float - how far to cast

        Returns:
            CollisionResult object
        """
        from panda3d.core import Point3, Vec3
        import math

        # Create collision ray
        ray_origin = Point3(origin[0], origin[1], origin[2])

        # Normalize direction
        dir_vec = Vec3(direction[0], direction[1], direction[2])
        dir_length = math.sqrt(dir_vec.x**2 + dir_vec.y**2 + dir_vec.z**2)
        if dir_length > 0:
            dir_vec = dir_vec / dir_length

        ray = CollisionRay()
        ray.setOrigin(ray_origin)
        ray.setDirection(dir_vec)

        # Create collision node for the ray
        ray_node = CollisionNode('boxcast_ray')
        ray_node.addSolid(ray)
        ray_node.setFromCollideMask(BitMask32.allOn())
        ray_node.setIntoCollideMask(BitMask32.allOff())  # Ray doesn't collide with other rays

        ray_np = self.base.render.attachNewNode(ray_node)

        # Perform collision check
        self.handler.clearEntries()
        self.traverser.addCollider(ray_np, self.handler)
        self.traverser.traverse(self.base.render)
        self.traverser.removeCollider(ray_np)

        # Clean up
        ray_np.removeNode()

        # Process results
        if self.handler.getNumEntries() > 0:
            # Sort by distance
            self.handler.sortEntries()
            entry = self.handler.getEntry(0)

            # Get collision info
            hit_point = entry.getSurfacePoint(self.base.render)
            hit_normal = entry.getSurfaceNormal(self.base.render)
            hit_distance = (ray_origin - hit_point).length()

            # CRITICAL FIX: Only return hits within the requested distance
            # This matches Ursina's boxcast behavior - only detect obstacles
            # within the specified distance (predictive collision detection)
            if hit_distance > distance:
                # Hit is beyond requested distance - treat as no collision
                return CollisionResult(
                    hit=False,
                    distance=distance + 999,
                    point=None,
                    normal=None
                )

            # Identify which object was hit
            hit_node = entry.getIntoNodePath()
            object_name = "Unknown"
            if hit_node.hasTag("object_type"):
                object_name = hit_node.getTag("object_type")

            return CollisionResult(
                hit=True,
                distance=hit_distance,
                point=(hit_point.x, hit_point.y, hit_point.z),
                normal=(hit_normal.x, hit_normal.y, hit_normal.z),
                object_name=object_name
            )
        else:
            # No collision
            return CollisionResult(
                hit=False,
                distance=distance + 999,
                point=None,
                normal=None
            )

    def boxcast_all(self, origin, direction, thickness=(0.1, 0.1), distance=1.0):
        """
        Cast a ray and return ALL collisions along the path (not just first).

        Returns:
            list: List of CollisionResult objects, sorted by distance
        """
        from panda3d.core import Point3, Vec3
        import math

        # Create collision ray
        ray_origin = Point3(origin[0], origin[1], origin[2])

        # Normalize direction
        dir_vec = Vec3(direction[0], direction[1], direction[2])
        dir_length = math.sqrt(dir_vec.x**2 + dir_vec.y**2 + dir_vec.z**2)
        if dir_length > 0:
            dir_vec = dir_vec / dir_length

        ray = CollisionRay()
        ray.setOrigin(ray_origin)
        ray.setDirection(dir_vec)

        # Create collision node for the ray
        ray_node = CollisionNode('boxcast_all_ray')
        ray_node.addSolid(ray)
        ray_node.setFromCollideMask(BitMask32.allOn())
        ray_node.setIntoCollideMask(BitMask32.allOff())

        ray_np = self.base.render.attachNewNode(ray_node)

        # Perform collision check
        self.handler.clearEntries()
        self.traverser.addCollider(ray_np, self.handler)
        self.traverser.traverse(self.base.render)
        self.traverser.removeCollider(ray_np)

        # Clean up
        ray_np.removeNode()

        # Process ALL results (not just first)
        results = []
        if self.handler.getNumEntries() > 0:
            self.handler.sortEntries()

            for i in range(self.handler.getNumEntries()):
                entry = self.handler.getEntry(i)

                # Get collision info
                hit_point = entry.getSurfacePoint(self.base.render)
                hit_normal = entry.getSurfaceNormal(self.base.render)
                hit_distance = (ray_origin - hit_point).length()

                # Only include hits within distance
                if hit_distance <= distance:
                    # Identify which object was hit
                    hit_node = entry.getIntoNodePath()
                    object_name = "Unknown"
                    if hit_node.hasTag("object_type"):
                        object_name = hit_node.getTag("object_type")

                    results.append(CollisionResult(
                        hit=True,
                        distance=hit_distance,
                        point=(hit_point.x, hit_point.y, hit_point.z),
                        normal=(hit_normal.x, hit_normal.y, hit_normal.z),
                        object_name=object_name
                    ))

        return results


class CollisionResult:
    """Result of a collision query."""

    def __init__(self, hit, distance, point, normal, object_name="Unknown"):
        from panda3d.core import Vec3

        self.hit = hit
        self.distance = distance
        self.object_name = object_name  # Which object was hit

        # Make these accessible as properties like Ursina
        if point:
            self.world_point = type('Point', (), {'x': point[0], 'y': point[1], 'z': point[2]})()
        else:
            self.world_point = None

        if normal:
            self.world_normal = Vec3(normal[0], normal[1], normal[2])
        else:
            self.world_normal = Vec3(0, 1, 0)

    def __bool__(self):
        return self.hit

    def __repr__(self):
        if self.hit:
            return f"CollisionResult(hit=True, distance={self.distance:.2f}, object={self.object_name})"
        else:
            return "CollisionResult(hit=False)"