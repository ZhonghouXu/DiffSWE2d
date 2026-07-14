import torch
import torch.nn.functional as F

def add_ghost_cells(
    U, 
    bed, 
    ng: int, 
    boundary_left: str = "transmissive", 
    boundary_right: str = "transmissive", 
    boundary_top: str = "transmissive", 
    boundary_bottom: str = "transmissive",
    constant_value: float = 0.0, 
    constant_level: float = 0.0
):
    """
    Pad state and bed with ghost cells independently for each boundary.
    U has channels [h, hu, hv]. 
    """
    
    # 1. Base Padding (Allocate Space)
    # We use "replicate" (transmissive) as the baseline for all sides. 
    # This automatically handles any boundary set to "transmissive" and prepares the bed topology.
    Up = F.pad(U, (ng, ng, ng, ng), mode="replicate")
    zp = F.pad(bed, (ng, ng, ng, ng), mode="replicate")

    # Pre-calculate forced_h in case any boundary is set to "water_level"
    # Formula: h = Z - bed. Clamp to 0.0 to prevent negative water depths.
    forced_h = torch.clamp(constant_level - zp, min=0.0)

    # ---------------------------------------------------------
    # --- LEFT BOUNDARY (x = 0 to ng) ---
    # ---------------------------------------------------------
    if boundary_left == "periodic":
        Up[:, :, :, :ng] = U[:, :, :, -ng:]
        zp[:, :, :, :ng] = bed[:, :, :, -ng:]
    elif boundary_left == "constant":
        Up[:, :, :, :ng] = constant_value
    elif boundary_left == "water_level":
        Up[:, 0, :, :ng] = forced_h[:, 0, :, :ng]
    elif boundary_left == "wall":
        # Reverse x-momentum (channel 1)
        Up[:, 1, :, :ng] = -Up[:, 1, :, ng:2 * ng].flip(-1)

    # ---------------------------------------------------------
    # --- RIGHT BOUNDARY (x = -ng to end) ---
    # ---------------------------------------------------------
    if boundary_right == "periodic":
        Up[:, :, :, -ng:] = U[:, :, :, :ng]
        zp[:, :, :, -ng:] = bed[:, :, :, :ng]
    elif boundary_right == "constant":
        Up[:, :, :, -ng:] = constant_value
    elif boundary_right == "water_level":
        Up[:, 0, :, -ng:] = forced_h[:, 0, :, -ng:]
    elif boundary_right == "wall":
        # Reverse x-momentum (channel 1)
        Up[:, 1, :, -ng:] = -Up[:, 1, :, -2 * ng:-ng].flip(-1)

    # ---------------------------------------------------------
    # --- TOP BOUNDARY (y = 0 to ng) ---
    # ---------------------------------------------------------
    if boundary_top == "periodic":
        Up[:, :, :ng, :] = U[:, :, -ng:, :]
        zp[:, :, :ng, :] = bed[:, :, -ng:, :]
    elif boundary_top == "constant":
        Up[:, :, :ng, :] = constant_value
    elif boundary_top == "water_level":
        Up[:, 0, :ng, :] = forced_h[:, 0, :ng, :]
    elif boundary_top == "wall":
        # Reverse y-momentum (channel 2)
        Up[:, 2, :ng, :] = -Up[:, 2, ng:2 * ng, :].flip(-2)

    # ---------------------------------------------------------
    # --- BOTTOM BOUNDARY (y = -ng to end) ---
    # ---------------------------------------------------------
    if boundary_bottom == "periodic":
        Up[:, :, -ng:, :] = U[:, :, :ng, :]
        zp[:, :, -ng:, :] = bed[:, :, :ng, :]
    elif boundary_bottom == "constant":
        Up[:, :, -ng:, :] = constant_value
    elif boundary_bottom == "water_level":
        Up[:, 0, -ng:, :] = forced_h[:, 0, -ng:, :]
    elif boundary_bottom == "wall":
        # Reverse y-momentum (channel 2)
        Up[:, 2, -ng:, :] = -Up[:, 2, -2 * ng:-ng, :].flip(-2)

    return Up, zp