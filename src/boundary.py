import torch.nn.functional as F


def add_ghost_cells(U, bed, ng: int, boundary: str):
    """Pad state and bed with ghost cells.

    U has channels [h, hu, hv]. A wall reverses only the momentum normal to the
    wall. A transmissive boundary copies the nearest interior cell.
    """
    if boundary == "periodic":
        return (
            F.pad(U, (ng, ng, ng, ng), mode="circular"),
            F.pad(bed, (ng, ng, ng, ng), mode="circular"),
        )

    if boundary not in ("wall", "transmissive"):
        raise ValueError(f"Unsupported boundary type: {boundary!r}")

    Up = F.pad(U, (ng, ng, ng, ng), mode="replicate")
    zp = F.pad(bed, (ng, ng, ng, ng), mode="replicate")

    if boundary == "wall":
        Up = Up.clone()
        # Left/right walls: reverse x-momentum hu.
        Up[:, 1, :, :ng] = -Up[:, 1, :, ng:2 * ng].flip(-1)
        Up[:, 1, :, -ng:] = -Up[:, 1, :, -2 * ng:-ng].flip(-1)
        # Bottom/top walls: reverse y-momentum hv.
        Up[:, 2, :ng, :] = -Up[:, 2, ng:2 * ng, :].flip(-2)
        Up[:, 2, -ng:, :] = -Up[:, 2, -2 * ng:-ng, :].flip(-2)

    return Up, zp
