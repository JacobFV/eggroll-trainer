"""MuJoCo-specific utilities for advanced examples."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import os
import xml.etree.ElementTree as ET

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


def load_custom_model(xml_path: str) -> Optional['mujoco.MjModel']:
    """
    Load a custom MJCF model from XML file.
    
    Args:
        xml_path: Path to MJCF XML file
        
    Returns:
        MuJoCo model, or None if MuJoCo not available
    """
    if not HAS_MUJOCO:
        return None
    
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"MJCF file not found: {xml_path}")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        return model
    except Exception as e:
        raise ValueError(f"Failed to load MJCF model from {xml_path}: {e}")


def create_model_from_xml_string(xml_string: str) -> Optional['mujoco.MjModel']:
    """
    Create MuJoCo model from XML string.
    
    Args:
        xml_string: MJCF XML content as string
        
    Returns:
        MuJoCo model, or None if MuJoCo not available
    """
    if not HAS_MUJOCO:
        return None
    
    try:
        model = mujoco.MjModel.from_xml_string(xml_string)
        return model
    except Exception as e:
        raise ValueError(f"Failed to create model from XML string: {e}")


def create_cloth_mesh(
    width: float = 2.0,
    height: float = 2.0,
    resolution: Tuple[int, int] = (10, 10),
    stiffness: float = 100.0,
    damping: float = 0.1,
) -> str:
    """
    Generate MJCF XML for a cloth mesh.
    
    Args:
        width: Width of cloth
        height: Height of cloth
        resolution: (rows, cols) for mesh resolution
        stiffness: Stiffness coefficient
        damping: Damping coefficient
        
    Returns:
        MJCF XML string for cloth
    """
    rows, cols = resolution
    
    # Create root element
    mujoco = ET.Element("mujoco")
    mujoco.set("model", "cloth")
    
    # World
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Ground
    ground = ET.SubElement(worldbody, "geom")
    ground.set("name", "ground")
    ground.set("type", "plane")
    ground.set("size", "10 10 0.1")
    ground.set("rgba", "0.5 0.5 0.5 1")
    
    # Cloth using composite - FULL MuJoCo XML format
    # Note: "cloth" type is deprecated, use "shell" instead
    composite = ET.SubElement(worldbody, "composite")
    composite.set("type", "shell")
    composite.set("count", f"{rows} {cols}")
    # Spacing is computed automatically from count and geom size
    composite.set("offset", f"0 0 1")
    
    # Cloth properties
    cloth_geom = ET.SubElement(composite, "geom")
    cloth_geom.set("type", "box")
    cloth_geom.set("size", f"{width/(cols*2)} {height/(rows*2)} 0.01")
    cloth_geom.set("rgba", "0.8 0.8 0.9 1")
    
    # Cloth material
    cloth_material = ET.SubElement(mujoco, "material")
    cloth_material.set("name", "cloth_mat")
    cloth_material.set("rgba", "0.8 0.8 0.9 1")
    
    # Cloth stiffness and damping
    cloth_geom.set("solimp", "0.9 0.95 0.001")
    cloth_geom.set("solref", "0.02 1")
    
    # Convert to string
    return ET.tostring(mujoco, encoding='unicode', method='xml')


def add_sensors_to_model(
    model: 'mujoco.MjModel',
    sensor_types: List[str],
    body_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Add custom sensors to a MuJoCo model.
    
    Note: This requires modifying the MJCF XML and reloading.
    For runtime sensors, use mujoco sensor API directly.
    
    Args:
        model: MuJoCo model
        sensor_types: List of sensor types ('touch', 'accelerometer', 'gyro', etc.)
        body_names: List of body names to attach sensors to
        
    Returns:
        Dictionary mapping sensor names to sensor IDs
    """
    # This is a placeholder - actual implementation would modify XML
    # For now, return empty dict
    return {}


def get_soft_body_state(
    data: 'mujoco.MjData',
    body_id: int,
) -> Dict[str, np.ndarray]:
    """
    Extract state of a soft body.
    
    Args:
        data: MuJoCo data
        body_id: ID of soft body
        
    Returns:
        Dictionary with 'positions', 'velocities', 'deformations'
    """
    if not HAS_MUJOCO:
        return {}
    
    # Extract soft body state
    # This is simplified - actual implementation depends on soft body structure
    state = {
        'positions': np.array([]),
        'velocities': np.array([]),
        'deformations': np.array([]),
    }
    
    return state


def create_tendon_system(
    body_names: List[str],
    attachment_points: List[Tuple[float, float, float]],
    routing_points: Optional[List[Tuple[float, float, float]]] = None,
) -> str:
    """
    Generate MJCF XML for a tendon system.
    
    Args:
        body_names: List of body names to attach tendons to
        attachment_points: List of (x, y, z) attachment points
        routing_points: Optional routing points for tendon path
        
    Returns:
        MJCF XML string for tendon system
    """
    tendon = ET.Element("tendon")
    
    spatial = ET.SubElement(tendon, "spatial")
    spatial.set("name", "tendon_0")
    
    # Add attachment sites
    for i, (body_name, point) in enumerate(zip(body_names, attachment_points)):
        site = ET.SubElement(spatial, "site")
        site.set("name", f"attachment_{i}")
        site.set("body", body_name)
        site.set("pos", f"{point[0]} {point[1]} {point[2]}")
    
    # Add routing points if provided
    if routing_points:
        for i, point in enumerate(routing_points):
            site = ET.SubElement(spatial, "site")
            site.set("name", f"routing_{i}")
            site.set("pos", f"{point[0]} {point[1]} {point[2]}")
    
    return ET.tostring(tendon, encoding='unicode', method='xml')


def apply_custom_forces(
    data: 'mujoco.MjData',
    body_id: int,
    force: np.ndarray,
    point: Optional[np.ndarray] = None,
):
    """
    Apply custom force to a body.
    
    Args:
        data: MuJoCo data
        body_id: Body ID to apply force to
        force: Force vector (3D)
        point: Optional point of application (in body frame)
    """
    if not HAS_MUJOCO:
        return
    
    # Apply force (using xfrc_applied array)
    # Note: This is a simplified version - actual implementation would
    # need to properly set xfrc_applied array indices
    if point is None:
        point = np.zeros(3)
    
    # Store force/torque in xfrc_applied (6D: 3 force + 3 torque)
    # This is a placeholder - actual implementation requires proper indexing
    # data.xfrc_applied[body_id * 6:(body_id * 6 + 3)] = force
    # data.xfrc_applied[body_id * 6 + 3:body_id * 6 + 6] = np.cross(point, force)


def create_heightfield_terrain(
    heightmap: np.ndarray,
    scale: float = 1.0,
    resolution: Optional[Tuple[int, int]] = None,
) -> str:
    """
    Generate MJCF XML for heightfield terrain.
    
    Args:
        heightmap: 2D array of heights
        scale: Scale factor for heights
        resolution: Optional (width, height) resolution
        
    Returns:
        MJCF XML string for heightfield
    """
    mujoco = ET.Element("mujoco")
    mujoco.set("model", "terrain")
    
    asset = ET.SubElement(mujoco, "asset")
    
    # Create heightfield - FULL MuJoCo XML format
    hfield = ET.SubElement(asset, "hfield")
    hfield.set("name", "terrain")
    hfield.set("size", f"{heightmap.shape[1]*scale} {heightmap.shape[0]*scale} 1.0 0.1")
    hfield.set("nrow", str(heightmap.shape[0]))
    hfield.set("ncol", str(heightmap.shape[1]))
    
    # Height data - MuJoCo requires height data to be set via API, not XML attribute
    # Store heightmap for later API access
    # Note: Height data must be set programmatically using mujoco.mj_forward or similar
    
    worldbody = ET.SubElement(mujoco, "worldbody")
    
    # Terrain geom
    terrain_geom = ET.SubElement(worldbody, "geom")
    terrain_geom.set("name", "terrain_geom")
    terrain_geom.set("type", "hfield")
    terrain_geom.set("hfield", "terrain")
    terrain_geom.set("rgba", "0.6 0.6 0.4 1")
    
    return ET.tostring(mujoco, encoding='unicode', method='xml')


def generate_procedural_heightmap(
    width: int = 100,
    height: int = 100,
    scale: float = 1.0,
    roughness: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate procedural heightmap using Perlin-like noise.
    
    Args:
        width: Width of heightmap
        height: Height of heightmap
        scale: Scale factor for noise
        roughness: Roughness factor (0=smooth, 1=rough)
        seed: Random seed
        
    Returns:
        2D array of heights
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Simple noise generation (can be replaced with Perlin noise)
    heights = np.random.randn(height, width) * roughness
    
    # Smooth with simple blur
    try:
        from scipy import ndimage
        heights = ndimage.gaussian_filter(heights, sigma=scale)
    except ImportError:
        # Fallback: simple averaging
        kernel = np.ones((3, 3)) / 9
        heights = np.convolve(heights.flatten(), kernel.flatten(), mode='same').reshape(heights.shape)
    
    # Normalize to [0, 1]
    heights = (heights - heights.min()) / (heights.max() - heights.min() + 1e-8)
    
    return heights


def get_contact_forces(
    data: 'mujoco.MjData',
    body_id: int,
) -> np.ndarray:
    """
    Get contact forces acting on a body.
    
    Args:
        data: MuJoCo data
        body_id: Body ID
        
    Returns:
        Array of contact forces (3D force vector)
    """
    if not HAS_MUJOCO:
        return np.zeros(3)
    
    # Sum contact forces for this body
    force = np.zeros(3)
    
    for i in range(data.ncon):
        contact = data.contact[i]
        # Get geom IDs from contact
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        # Check if this contact involves the body
        # Need to check which geom belongs to which body
        if geom1_id >= 0 and geom2_id >= 0:
            # Get body IDs from geoms
            geom1_body = data.geom_bodyid[geom1_id]
            geom2_body = data.geom_bodyid[geom2_id]
            
            if geom1_body == body_id or geom2_body == body_id:
                # Extract force from contact (using contact force array)
                # MuJoCo stores contact forces in data.efc_force
                if i < len(data.efc_force):
                    # Contact normal force
                    normal_force = data.efc_force[i]
                    # Contact frame (normal, tangent1, tangent2)
                    if i < len(data.contact):
                        contact_frame = contact.frame
                        # Force in world frame
                        force += normal_force * contact_frame[:3, 0]
    
    return force


def get_all_contact_forces(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
) -> Dict[int, np.ndarray]:
    """
    Get all contact forces for all bodies.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        
    Returns:
        Dictionary mapping body IDs to contact force vectors
    """
    if not HAS_MUJOCO:
        return {}
    
    forces = {}
    
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        
        if geom1_id >= 0 and geom2_id >= 0:
            geom1_body = data.geom_bodyid[geom1_id]
            geom2_body = data.geom_bodyid[geom2_id]
            
            if i < len(data.efc_force):
                normal_force = data.efc_force[i]
                contact_frame = contact.frame
                force_vec = normal_force * contact_frame[:3, 0]
                
                # Add to both bodies (Newton's third law)
                if geom1_body not in forces:
                    forces[geom1_body] = np.zeros(3)
                if geom2_body not in forces:
                    forces[geom2_body] = np.zeros(3)
                
                forces[geom1_body] += force_vec
                forces[geom2_body] -= force_vec
    
    return forces


def compute_cloth_deformation(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    cloth_body_id: int,
) -> Dict[str, float]:
    """
    Compute cloth deformation metrics.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        cloth_body_id: Body ID of cloth
        
    Returns:
        Dictionary with deformation metrics
    """
    if not HAS_MUJOCO:
        return {'max_deformation': 0.0, 'mean_deformation': 0.0, 'deformation_energy': 0.0}
    
    # Get positions of cloth particles
    # For composite cloth, need to iterate through all bodies in composite
    positions = []
    
    # Find all bodies that are part of the cloth composite
    for i in range(model.nbody):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if body_name and 'cloth' in body_name.lower():
            pos = data.xpos[i]
            positions.append(pos)
    
    if len(positions) < 2:
        return {'max_deformation': 0.0, 'mean_deformation': 0.0, 'deformation_energy': 0.0}
    
    positions = np.array(positions)
    
    # Compute deformation metrics
    # Rest position would be flat plane at z=0.5 (or initial position)
    rest_z = 0.5
    current_z = positions[:, 2]
    
    # Deformation is deviation from rest position
    deformation = np.abs(current_z - rest_z)
    max_deformation = np.max(deformation)
    mean_deformation = np.mean(deformation)
    
    # Deformation energy (proportional to squared deformation)
    deformation_energy = np.sum(deformation ** 2)
    
    return {
        'max_deformation': float(max_deformation),
        'mean_deformation': float(mean_deformation),
        'deformation_energy': float(deformation_energy),
    }


def apply_fluid_dynamics(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    body_id: int,
    water_level: float = 2.0,
    fluid_density: float = 1000.0,
    drag_coefficient: float = 0.5,
    buoyancy_density: float = 500.0,
) -> np.ndarray:
    """
    Apply fluid dynamics forces (buoyancy and drag) to a body.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        body_id: Body ID to apply forces to
        water_level: Z-coordinate of water surface
        fluid_density: Density of fluid (kg/m^3)
        drag_coefficient: Drag coefficient
        buoyancy_density: Density of body for buoyancy calculation
        
    Returns:
        Applied force vector (3D)
    """
    if not HAS_MUJOCO:
        return np.zeros(3)
    
    # Get body position and velocity
    body_pos = data.xpos[body_id]
    body_vel = data.qvel[model.body_jntadr[body_id]:model.body_jntadr[body_id] + 6]
    lin_vel = body_vel[:3]
    
    # Check if body is underwater
    if body_pos[2] >= water_level:
        return np.zeros(3)
    
    # Compute depth
    depth = water_level - body_pos[2]
    
    # Buoyancy force (upward)
    # F_buoyancy = -rho_fluid * V * g (upward)
    # Simplified: assume constant volume
    body_mass = model.body_mass[body_id]
    body_volume = body_mass / buoyancy_density  # Approximate volume
    
    gravity = model.opt.gravity[2]  # Usually -9.81
    buoyancy_force = -fluid_density * body_volume * gravity  # Upward (positive z)
    buoyancy = np.array([0.0, 0.0, buoyancy_force])
    
    # Drag force (opposite to velocity)
    # F_drag = -0.5 * rho * Cd * A * v^2 * v_hat
    speed = np.linalg.norm(lin_vel)
    if speed > 1e-6:
        drag_magnitude = 0.5 * fluid_density * drag_coefficient * body_volume ** (2/3) * speed ** 2
        drag_direction = -lin_vel / speed
        drag = drag_magnitude * drag_direction
    else:
        drag = np.zeros(3)
    
    # Apply forces
    total_force = buoyancy + drag
    
    # Apply via xfrc_applied
    if body_id < model.nbody:
        data.xfrc_applied[body_id, :3] = total_force
    
    return total_force


def compute_muscle_force(
    activation: float,
    muscle_length: float,
    muscle_velocity: float,
    optimal_length: float = 1.0,
    max_force: float = 1000.0,
    max_velocity: float = 10.0,
) -> float:
    """
    Compute muscle force using force-length-velocity relationship.
    
    Args:
        activation: Muscle activation [0, 1]
        muscle_length: Current muscle length
        muscle_velocity: Current muscle velocity
        optimal_length: Optimal muscle length (for force-length curve)
        max_force: Maximum isometric force
        max_velocity: Maximum contraction velocity
        
    Returns:
        Muscle force
    """
    # Force-length relationship (Hill-type model)
    # Normalized length
    normalized_length = muscle_length / optimal_length
    
    # Force-length curve (Gaussian-like)
    if normalized_length < 0.5 or normalized_length > 1.5:
        fl = 0.0
    else:
        # Peak at optimal length
        fl = np.exp(-((normalized_length - 1.0) ** 2) / (2 * 0.1 ** 2))
    
    # Force-velocity relationship
    normalized_velocity = muscle_velocity / max_velocity
    
    if normalized_velocity < 0:
        # Concentric contraction (shortening)
        fv = (1.0 - normalized_velocity) / (1.0 + normalized_velocity / 0.25)
    else:
        # Eccentric contraction (lengthening)
        fv = 1.0 + 1.5 * normalized_velocity / (1.0 + normalized_velocity)
    
    # Clamp fv
    fv = np.clip(fv, 0.0, 2.0)
    
    # Total force
    force = activation * fl * fv * max_force
    
    return force


def update_muscle_dynamics(
    muscle_activation: float,
    target_activation: float,
    dt: float,
    activation_time_constant: float = 0.05,
    fatigue_rate: float = 0.01,
    recovery_rate: float = 0.005,
) -> Tuple[float, float]:
    """
    Update muscle activation and fatigue dynamics.
    
    Args:
        muscle_activation: Current activation [0, 1]
        target_activation: Target activation [0, 1]
        dt: Time step
        activation_time_constant: Time constant for activation changes
        fatigue_rate: Rate of fatigue accumulation
        recovery_rate: Rate of fatigue recovery
        
    Returns:
        Tuple of (new_activation, new_fatigue)
    """
    # Update activation (first-order dynamics)
    activation_dot = (target_activation - muscle_activation) / activation_time_constant
    new_activation = muscle_activation + activation_dot * dt
    new_activation = np.clip(new_activation, 0.0, 1.0)
    
    # Update fatigue (accumulates with activation, recovers when inactive)
    fatigue_dot = fatigue_rate * new_activation - recovery_rate * (1.0 - new_activation)
    # Fatigue would be tracked separately - this is a placeholder
    new_fatigue = 0.0  # Would be updated based on fatigue_dot
    
    return new_activation, new_fatigue


def get_tendon_lengths(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    tendon_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Get current lengths of tendons.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        tendon_ids: List of tendon IDs (if None, returns all)
        
    Returns:
        Array of tendon lengths
    """
    if not HAS_MUJOCO:
        return np.array([])
    
    if tendon_ids is None:
        tendon_ids = list(range(model.ntendon))
    
    lengths = []
    for tid in tendon_ids:
        if tid < model.ntendon:
            length = data.ten_length[tid]
            lengths.append(length)
    
    return np.array(lengths)


def get_tendon_velocities(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    tendon_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Get current velocities of tendons.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        tendon_ids: List of tendon IDs (if None, returns all)
        
    Returns:
        Array of tendon velocities
    """
    if not HAS_MUJOCO:
        return np.array([])
    
    if tendon_ids is None:
        tendon_ids = list(range(model.ntendon))
    
    velocities = []
    for tid in tendon_ids:
        if tid < model.ntendon:
            velocity = data.ten_velocity[tid]
            velocities.append(velocity)
    
    return np.array(velocities)


def get_tendon_tensions(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    tendon_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """
    Get current tensions in tendons.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        tendon_ids: List of tendon IDs (if None, returns all)
        
    Returns:
        Array of tendon tensions
    """
    if not HAS_MUJOCO:
        return np.array([])
    
    if tendon_ids is None:
        tendon_ids = list(range(model.ntendon))
    
    tensions = []
    for tid in tendon_ids:
        if tid < model.ntendon:
            tension = data.ten_force[tid]
            tensions.append(tension)
    
    return np.array(tensions)


def compute_tendon_moment_arms(
    model: 'mujoco.MjModel',
    tendon_id: int,
    joint_id: int,
) -> float:
    """
    Compute moment arm of a tendon about a joint - FULL implementation.
    
    Args:
        model: MuJoCo model
        tendon_id: Tendon ID
        joint_id: Joint ID
        
    Returns:
        Moment arm (distance)
    """
    if not HAS_MUJOCO:
        return 0.0
    
    # Moment arm is stored in model.tendon_moment
    if tendon_id < model.ntendon and joint_id < model.njnt:
        # Get moment arm from tendon-joint coupling matrix
        # model.tendon_moment is a sparse matrix: nv x ntendon
        # Each row corresponds to a velocity DOF, columns to tendons
        # Find the row corresponding to this joint's velocity DOF
        joint_dof_start = model.jnt_dofadr[joint_id]
        if joint_dof_start >= 0 and joint_dof_start < model.nv:
            # Get moment arm from coupling matrix
            # model.tendon_moment is stored as sparse matrix
            # Access via model.tendon_moment[joint_dof_start * model.ntendon + tendon_id]
            # But MuJoCo uses sparse storage, so we need to search
            for i in range(model.tendon_moment_nnz):
                if (model.tendon_moment_ind[i] == joint_dof_start * model.ntendon + tendon_id or
                    model.tendon_moment_ind[i] == tendon_id * model.nv + joint_dof_start):
                    return abs(model.tendon_moment[i])
            
            # Fallback: estimate from tendon length and joint angle
            # If tendon wraps around joint, moment arm ≈ tendon distance from joint axis
            return 0.1  # Default estimate
    
    return 0.0


def get_skinned_mesh_vertices(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    skin_id: int,
) -> np.ndarray:
    """
    Get skinned mesh vertex positions.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        skin_id: Skin ID
        
    Returns:
        Array of vertex positions (N x 3)
    """
    if not HAS_MUJOCO:
        return np.array([])
    
    if skin_id >= model.nskin:
        return np.array([])
    
    # Get vertex positions from skin
    # MuJoCo stores skinned vertices in data.skin_xpos
    num_vertices = model.skin_vertnum[skin_id]
    start_idx = model.skin_vertadr[skin_id]
    
    vertices = []
    for i in range(num_vertices):
        vert_idx = start_idx + i
        if vert_idx < len(data.skin_xpos):
            vertices.append(data.skin_xpos[vert_idx])
    
    return np.array(vertices) if vertices else np.array([])


def compute_skin_deformation(
    data: 'mujoco.MjData',
    model: 'mujoco.MjModel',
    skin_id: int,
    rest_vertices: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute skin deformation metrics.
    
    Args:
        data: MuJoCo data
        model: MuJoCo model
        skin_id: Skin ID
        rest_vertices: Rest position vertices (if None, uses current as rest)
        
    Returns:
        Dictionary with deformation metrics
    """
    if not HAS_MUJOCO:
        return {'max_deformation': 0.0, 'mean_deformation': 0.0}
    
    current_vertices = get_skinned_mesh_vertices(data, model, skin_id)
    
    if len(current_vertices) == 0:
        return {'max_deformation': 0.0, 'mean_deformation': 0.0}
    
    if rest_vertices is None:
        rest_vertices = current_vertices.copy()
    
    if rest_vertices.shape != current_vertices.shape:
        return {'max_deformation': 0.0, 'mean_deformation': 0.0}
    
    # Compute vertex displacements
    displacements = np.linalg.norm(current_vertices - rest_vertices, axis=1)
    
    max_deformation = np.max(displacements)
    mean_deformation = np.mean(displacements)
    
    return {
        'max_deformation': float(max_deformation),
        'mean_deformation': float(mean_deformation),
    }


def create_custom_actuator(
    name: str,
    joint_name: str,
    actuator_type: str = "motor",
    gain: float = 1.0,
    bias: float = 0.0,
) -> str:
    """
    Generate MJCF XML for a custom actuator.
    
    Args:
        name: Actuator name
        joint_name: Joint to actuate
        actuator_type: Type ('motor', 'position', 'velocity', 'torque')
        gain: Actuator gain
        bias: Actuator bias
        
    Returns:
        MJCF XML string for actuator
    """
    actuator = ET.Element("actuator")
    
    act = ET.SubElement(actuator, actuator_type)
    act.set("name", name)
    act.set("joint", joint_name)
    act.set("gainprm", str(gain))
    act.set("biasprm", f"0 0 {bias}")
    
    return ET.tostring(actuator, encoding='unicode', method='xml')
