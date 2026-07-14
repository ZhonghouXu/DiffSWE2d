import torch.nn.functional as F


def add_ghost_cells(U, bed, ng: int, boundary: str, constant_value: float = 0.0, constant_level: float = 0.0):
    """Pad state and bed with ghost cells.

    U has channels [h, hu, hv]. A wall reverses only the momentum normal to the
    wall. A transmissive boundary copies the nearest interior cell.
    """
    if boundary == "periodic":
        return (
            F.pad(U, (ng, ng, ng, ng), mode="circular"),
            F.pad(bed, (ng, ng, ng, ng), mode="circular"),
        )
    
    if boundary == "constant":
        return (
            # Pad the fluid state (U) with the specified constant value
            F.pad(U, (ng, ng, ng, ng), mode="constant", value=constant_value),
            # Replicate the bed so you don't create an artificial cliff at the edge
            F.pad(bed, (ng, ng, ng, ng), mode="replicate"),
        )
    
    if boundary == "water_level":
        # Replicate both U and bed so water can flow freely out (transmissive momentum)
        Up = F.pad(U, (ng, ng, ng, ng), mode="replicate")
        zp = F.pad(bed, (ng, ng, ng, ng), mode="replicate")
        
        # Calculate the required water depth to force the surface to 'constant_level'
        # Formula: h = Z - bed. We clamp to 0.0 so we don't get negative water depths.
        forced_h = torch.clamp(constant_level - zp, min=0.0)
        
        # Overwrite only the water depth (channel 0) in the ghost cell regions
        # Left and Right edges
        Up[:, 0, :, :ng] = forced_h[:, 0, :, :ng]
        Up[:, 0, :, -ng:] = forced_h[:, 0, :, -ng:]
        # Top and Bottom edges
        Up[:, 0, :ng, :] = forced_h[:, 0, :ng, :]
        Up[:, 0, -ng:, :] = forced_h[:, 0, -ng:, :]
        
        return Up, zp
    
    if boundary == "transmissive":
        return (
            F.pad(U, (ng, ng, ng, ng), mode="replicate"),
            F.pad(bed, (ng, ng, ng, ng), mode="replicate"),
        )  

    if boundary == "wall":
        Up = Up.clone()
        # Left/right walls: reverse x-momentum hu.
        Up[:, 1, :, :ng] = -Up[:, 1, :, ng:2 * ng].flip(-1)
        Up[:, 1, :, -ng:] = -Up[:, 1, :, -2 * ng:-ng].flip(-1)
        # Bottom/top walls: reverse y-momentum hv.
        Up[:, 2, :ng, :] = -Up[:, 2, ng:2 * ng, :].flip(-2)
        Up[:, 2, -ng:, :] = -Up[:, 2, -2 * ng:-ng, :].flip(-2)
        zp = F.pad(bed, (ng, ng, ng, ng), mode="replicate")
        return Up, zp

    if boundary not in ("periodic", "constant", "water_level", "transmissive", "wall"):
        raise ValueError(f"Unsupported boundary type: {boundary!r}")
        # default boundary is "transmissive" if not defined
        return (
            F.pad(U, (ng, ng, ng, ng), mode="replicate"),
            F.pad(bed, (ng, ng, ng, ng), mode="replicate"),
        )   
