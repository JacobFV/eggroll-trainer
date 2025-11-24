"""World building utilities for procedural generation."""

import numpy as np
from typing import List, Tuple, Optional, Dict
import xml.etree.ElementTree as ET

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


class ObstacleCourseBuilder:
    """Builder for procedurally generated obstacle courses."""
    
    def __init__(self, width: float = 20.0, height: float = 20.0):
        """
        Initialize obstacle course builder.
        
        Args:
            width: Course width
            height: Course height
        """
        self.width = width
        self.height = height
        self.obstacles = []
        self.moving_platforms = []
        self.pendulums = []
    
    def add_box_obstacle(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        name: Optional[str] = None,
    ):
        """Add a box obstacle."""
        if name is None:
            name = f"obstacle_{len(self.obstacles)}"
        
        self.obstacles.append({
            'type': 'box',
            'name': name,
            'pos': position,
            'size': size,
        })
    
    def add_moving_platform(
        self,
        start_pos: Tuple[float, float, float],
        end_pos: Tuple[float, float, float],
        size: Tuple[float, float, float],
        speed: float = 1.0,
        name: Optional[str] = None,
    ):
        """Add a moving platform."""
        if name is None:
            name = f"platform_{len(self.moving_platforms)}"
        
        self.moving_platforms.append({
            'name': name,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'size': size,
            'speed': speed,
        })
    
    def add_pendulum(
        self,
        anchor_pos: Tuple[float, float, float],
        length: float,
        mass: float = 1.0,
        name: Optional[str] = None,
    ):
        """Add a swinging pendulum obstacle."""
        if name is None:
            name = f"pendulum_{len(self.pendulums)}"
        
        self.pendulums.append({
            'name': name,
            'anchor_pos': anchor_pos,
            'length': length,
            'mass': mass,
        })
    
    def generate_xml(self) -> str:
        """Generate MJCF XML for the obstacle course."""
        mujoco = ET.Element("mujoco")
        mujoco.set("model", "obstacle_course")
        
        # Options
        option = ET.SubElement(mujoco, "option")
        option.set("timestep", "0.002")
        option.set("gravity", "0 0 -9.81")
        
        # Assets
        asset = ET.SubElement(mujoco, "asset")
        
        # Materials
        obstacle_mat = ET.SubElement(asset, "material")
        obstacle_mat.set("name", "obstacle")
        obstacle_mat.set("rgba", "0.7 0.3 0.3 1")
        
        platform_mat = ET.SubElement(asset, "material")
        platform_mat.set("name", "platform")
        platform_mat.set("rgba", "0.3 0.7 0.3 1")
        
        # World body
        worldbody = ET.SubElement(mujoco, "worldbody")
        
        # Ground
        ground = ET.SubElement(worldbody, "geom")
        ground.set("name", "ground")
        ground.set("type", "plane")
        ground.set("size", f"{self.width} {self.height} 0.1")
        ground.set("rgba", "0.5 0.5 0.5 1")
        
        # Add obstacles
        for obs in self.obstacles:
            body = ET.SubElement(worldbody, "body")
            body.set("name", obs['name'])
            body.set("pos", f"{obs['pos'][0]} {obs['pos'][1]} {obs['pos'][2]}")
            
            geom = ET.SubElement(body, "geom")
            geom.set("type", obs['type'])
            geom.set("size", f"{obs['size'][0]} {obs['size'][1]} {obs['size'][2]}")
            geom.set("material", "obstacle")
        
        # Add moving platforms
        for platform in self.moving_platforms:
            body = ET.SubElement(worldbody, "body")
            body.set("name", platform['name'])
            body.set("pos", f"{platform['start_pos'][0]} {platform['start_pos'][1]} {platform['start_pos'][2]}")
            
            # Slider joint for movement
            joint = ET.SubElement(body, "joint")
            joint.set("name", f"{platform['name']}_joint")
            joint.set("type", "slide")
            joint.set("axis", "1 0 0")  # Move along x-axis
            
            geom = ET.SubElement(body, "geom")
            geom.set("type", "box")
            geom.set("size", f"{platform['size'][0]} {platform['size'][1]} {platform['size'][2]}")
            geom.set("material", "platform")
            
            # Actuator for movement
            actuator = ET.SubElement(mujoco, "actuator")
            motor = ET.SubElement(actuator, "motor")
            motor.set("name", f"{platform['name']}_motor")
            motor.set("joint", f"{platform['name']}_joint")
            motor.set("gear", str(platform['speed']))
        
        # Add pendulums
        for pend in self.pendulums:
            # Anchor
            anchor = ET.SubElement(worldbody, "body")
            anchor.set("name", f"{pend['name']}_anchor")
            anchor.set("pos", f"{pend['anchor_pos'][0]} {pend['anchor_pos'][1]} {pend['anchor_pos'][2]}")
            
            anchor_geom = ET.SubElement(anchor, "geom")
            anchor_geom.set("type", "sphere")
            anchor_geom.set("size", "0.05")
            anchor_geom.set("rgba", "0.5 0.5 0.5 1")
            
            # Pendulum bob
            bob = ET.SubElement(anchor, "body")
            bob.set("name", f"{pend['name']}_bob")
            bob.set("pos", f"0 0 -{pend['length']}")
            
            joint = ET.SubElement(bob, "joint")
            joint.set("name", f"{pend['name']}_joint")
            joint.set("type", "hinge")
            joint.set("axis", "0 1 0")
            
            geom = ET.SubElement(bob, "geom")
            geom.set("type", "sphere")
            geom.set("size", "0.2")
            geom.set("mass", str(pend['mass']))
            geom.set("rgba", "0.8 0.2 0.2 1")
        
        return ET.tostring(mujoco, encoding='unicode', method='xml')


class TerrainGenerator:
    """Generator for procedural terrain."""
    
    def __init__(self, width: int = 100, height: int = 100):
        """
        Initialize terrain generator.
        
        Args:
            width: Terrain width in samples
            height: Terrain height in samples
        """
        self.width = width
        self.height = height
        self.heightmap = None
    
    def generate_hills(
        self,
        num_hills: int = 5,
        hill_height: float = 2.0,
        hill_width: float = 5.0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate hilly terrain.
        
        Args:
            num_hills: Number of hills
            hill_height: Maximum hill height
            hill_width: Hill width
            seed: Random seed
            
        Returns:
            Heightmap array
        """
        if seed is not None:
            np.random.seed(seed)
        
        heightmap = np.zeros((self.height, self.width))
        
        for _ in range(num_hills):
            center_x = np.random.uniform(0, self.width)
            center_y = np.random.uniform(0, self.height)
            
            x = np.arange(self.width)
            y = np.arange(self.height)
            X, Y = np.meshgrid(x, y)
            
            dist = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
            hill = hill_height * np.exp(-(dist**2) / (2 * hill_width**2))
            heightmap += hill
        
        # Normalize
        heightmap = heightmap / heightmap.max() if heightmap.max() > 0 else heightmap
        
        self.heightmap = heightmap
        return heightmap
    
    def generate_noise_terrain(
        self,
        scale: float = 10.0,
        roughness: float = 0.5,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate noise-based terrain.
        
        Args:
            scale: Noise scale
            roughness: Roughness factor
            seed: Random seed
            
        Returns:
            Heightmap array
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Simple noise generation
        heightmap = np.random.randn(self.height, self.width) * roughness
        
        # Smooth
        try:
            from scipy import ndimage
            heightmap = ndimage.gaussian_filter(heightmap, sigma=scale)
        except ImportError:
            # Simple blur fallback
            kernel = np.ones((5, 5)) / 25
            heightmap = np.convolve(
                heightmap.flatten(),
                kernel.flatten(),
                mode='same'
            ).reshape(heightmap.shape)
        
        # Normalize
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-8)
        
        self.heightmap = heightmap
        return heightmap
    
    def generate_xml(self, scale: float = 1.0) -> str:
        """
        Generate MJCF XML for terrain.
        
        Args:
            scale: Scale factor for terrain size
            
        Returns:
            MJCF XML string
        """
        if self.heightmap is None:
            self.generate_noise_terrain()
        
        mujoco = ET.Element("mujoco")
        mujoco.set("model", "terrain")
        
        # Options
        option = ET.SubElement(mujoco, "option")
        option.set("timestep", "0.002")
        
        # Assets
        asset = ET.SubElement(mujoco, "asset")
        
        # Heightfield
        hfield = ET.SubElement(asset, "hfield")
        hfield.set("name", "terrain_hfield")
        hfield.set("size", f"{self.width*scale} {self.height*scale} 2.0 0.1")
        hfield.set("nrow", str(self.height))
        hfield.set("ncol", str(self.width))
        
        # Height data (flattened)
        heights_flat = self.heightmap.flatten()
        hfield.set("height", " ".join(f"{h:.4f}" for h in heights_flat))
        
        # World body
        worldbody = ET.SubElement(mujoco, "worldbody")
        
        # Terrain geom
        terrain = ET.SubElement(worldbody, "geom")
        terrain.set("name", "terrain")
        terrain.set("type", "hfield")
        terrain.set("hfield", "terrain_hfield")
        terrain.set("rgba", "0.6 0.6 0.4 1")
        terrain.set("friction", "1.0 0.5 0.0001")
        
        return ET.tostring(mujoco, encoding='unicode', method='xml')


class WorldRandomizer:
    """Randomizer for world parameters."""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize world randomizer.
        
        Args:
            seed: Random seed
        """
        self.rng = np.random.RandomState(seed)
    
    def randomize_obstacle_positions(
        self,
        obstacles: List[Dict],
        bounds: Tuple[float, float, float, float],
    ) -> List[Dict]:
        """
        Randomize obstacle positions within bounds.
        
        Args:
            obstacles: List of obstacle dictionaries
            bounds: (min_x, max_x, min_y, max_y)
            
        Returns:
            List of obstacles with randomized positions
        """
        min_x, max_x, min_y, max_y = bounds
        
        randomized = []
        for obs in obstacles:
            new_obs = obs.copy()
            new_obs['pos'] = (
                self.rng.uniform(min_x, max_x),
                self.rng.uniform(min_y, max_y),
                obs['pos'][2],  # Keep z the same
            )
            randomized.append(new_obs)
        
        return randomized
    
    def randomize_material_properties(
        self,
        friction_range: Tuple[float, float] = (0.5, 1.5),
        restitution_range: Tuple[float, float] = (0.0, 0.5),
    ) -> Dict[str, float]:
        """
        Generate random material properties.
        
        Args:
            friction_range: (min, max) friction
            restitution_range: (min, max) restitution
            
        Returns:
            Dictionary of material properties
        """
        return {
            'friction': self.rng.uniform(*friction_range),
            'restitution': self.rng.uniform(*restitution_range),
        }


class ObjectSpawner:
    """Spawner for dynamic objects."""
    
    def __init__(self):
        """Initialize object spawner."""
        self.spawned_objects = []
    
    def spawn_box(
        self,
        position: Tuple[float, float, float],
        size: Tuple[float, float, float],
        mass: float = 1.0,
        name: Optional[str] = None,
    ) -> Dict:
        """
        Spawn a box object.
        
        Args:
            position: Initial position
            size: Box size
            mass: Box mass
            name: Object name
            
        Returns:
            Object dictionary
        """
        if name is None:
            name = f"box_{len(self.spawned_objects)}"
        
        obj = {
            'type': 'box',
            'name': name,
            'pos': position,
            'size': size,
            'mass': mass,
        }
        
        self.spawned_objects.append(obj)
        return obj
    
    def spawn_sphere(
        self,
        position: Tuple[float, float, float],
        radius: float,
        mass: float = 1.0,
        name: Optional[str] = None,
    ) -> Dict:
        """
        Spawn a sphere object.
        
        Args:
            position: Initial position
            radius: Sphere radius
            mass: Sphere mass
            name: Object name
            
        Returns:
            Object dictionary
        """
        if name is None:
            name = f"sphere_{len(self.spawned_objects)}"
        
        obj = {
            'type': 'sphere',
            'name': name,
            'pos': position,
            'radius': radius,
            'mass': mass,
        }
        
        self.spawned_objects.append(obj)
        return obj
    
    def generate_xml(self) -> str:
        """Generate MJCF XML for spawned objects."""
        mujoco = ET.Element("mujoco")
        mujoco.set("model", "spawned_objects")
        
        worldbody = ET.SubElement(mujoco, "worldbody")
        
        for obj in self.spawned_objects:
            body = ET.SubElement(worldbody, "body")
            body.set("name", obj['name'])
            body.set("pos", f"{obj['pos'][0]} {obj['pos'][1]} {obj['pos'][2]}")
            
            geom = ET.SubElement(body, "geom")
            geom.set("type", obj['type'])
            
            if obj['type'] == 'box':
                geom.set("size", f"{obj['size'][0]} {obj['size'][1]} {obj['size'][2]}")
            elif obj['type'] == 'sphere':
                geom.set("size", str(obj['radius']))
            
            geom.set("mass", str(obj['mass']))
            geom.set("rgba", "0.8 0.2 0.2 1")
        
        return ET.tostring(mujoco, encoding='unicode', method='xml')

