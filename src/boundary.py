from __future__ import annotations

from typing import Union

import torch
import torch.nn.functional as F


ScalarOrTensor = Union[float, int, torch.Tensor]

_VALID_BOUNDARY_TYPES = {
    "transmissive",
    "periodic",
    "constant",
    "water_level",
    "wall",
    "absorbing",
}


def _to_reference_tensor(
    value: ScalarOrTensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Convert a scalar or tensor to the dtype and device of `reference`.

    If `value` is already a differentiable tensor, using `.to()` preserves
    its connection to the autograd graph.

    Do not replace this with:

        torch.tensor(value)

    because that would detach an existing tensor from its graph.
    """
    if torch.is_tensor(value):
        return value.to(
            dtype=reference.dtype,
            device=reference.device,
        )

    return reference.new_tensor(value)


def _validate_inputs(
    U: torch.Tensor,
    bed: torch.Tensor,
    ng: int,
    boundaries: dict[str, str],
) -> None:
    """Validate state, bed, ghost-cell count, and boundary names."""
    if U.ndim != 4:
        raise ValueError(
            "U must be a four-dimensional tensor with shape "
            f"[batch, 3, ny, nx]. Received shape {tuple(U.shape)}."
        )

    if U.shape[1] != 3:
        raise ValueError(
            "U must contain three channels [h, hu, hv]. "
            f"Received {U.shape[1]} channels."
        )

    if bed.ndim != 4:
        raise ValueError(
            "bed must be a four-dimensional tensor with shape "
            f"[batch or 1, 1, ny, nx]. Received shape {tuple(bed.shape)}."
        )

    if bed.shape[1] != 1:
        raise ValueError(
            "bed must contain one elevation channel. "
            f"Received {bed.shape[1]} channels."
        )

    if U.shape[-2:] != bed.shape[-2:]:
        raise ValueError(
            "U and bed must have identical spatial dimensions. "
            f"U has {tuple(U.shape[-2:])}; "
            f"bed has {tuple(bed.shape[-2:])}."
        )

    if bed.shape[0] not in (1, U.shape[0]):
        raise ValueError(
            "The bed batch dimension must be either 1 or equal to "
            f"the state batch size. U batch={U.shape[0]}, "
            f"bed batch={bed.shape[0]}."
        )

    if not isinstance(ng, int) or isinstance(ng, bool) or ng <= 0:
        raise ValueError(
            f"ng must be a positive integer. Received {ng!r}."
        )

    ny, nx = U.shape[-2:]

    if ng > ny or ng > nx:
        raise ValueError(
            "ng cannot exceed either interior grid dimension. "
            f"Received ng={ng}, ny={ny}, nx={nx}."
        )

    for side, boundary_type in boundaries.items():
        if boundary_type not in _VALID_BOUNDARY_TYPES:
            raise ValueError(
                f"Unsupported {side} boundary type "
                f"{boundary_type!r}. Valid types are "
                f"{sorted(_VALID_BOUNDARY_TYPES)}."
            )


def _prepare_boundary_value(
    value: ScalarOrTensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """
    Prepare a boundary value for broadcasting against a 4-D model field.

    Supported forms are:

    - Python scalar
    - zero-dimensional scalar tensor
    - one-element tensor
    - one value per batch member, with shape [batch]

    The normal inverse-tsunami use case supplies a zero-dimensional tensor.
    """
    value_tensor = _to_reference_tensor(value, reference)

    if value_tensor.ndim == 0:
        return value_tensor

    if value_tensor.numel() == 1:
        return value_tensor.reshape(())

    if (
        value_tensor.ndim == 1
        and value_tensor.shape[0] == reference.shape[0]
    ):
        return value_tensor.reshape(-1, 1, 1, 1)

    raise ValueError(
        "Boundary values must be a scalar, a one-element tensor, "
        "or a tensor with one value per batch member. "
        f"Received shape {tuple(value_tensor.shape)}."
    )


def add_ghost_cells(
    U: torch.Tensor,
    bed: torch.Tensor,
    ng: int,
    boundary_left: str = "transmissive",
    boundary_right: str = "transmissive",
    boundary_top: str = "transmissive",
    boundary_bottom: str = "transmissive",
    constant_value: ScalarOrTensor = 0.0,
    constant_level: ScalarOrTensor = 0.0,
):
    """
    Pad the SWE state and bed with ghost cells.

    Parameters
    ----------
    U
        Conservative model state with shape [batch, 3, ny, nx].
        The channels are [h, hu, hv].

    bed
        Bed elevation with shape [batch or 1, 1, ny, nx].

    ng
        Number of ghost-cell layers.

    boundary_left, boundary_right, boundary_top, boundary_bottom
        Boundary condition applied on each side. Supported values:

        - ``"transmissive"``: replicate the nearest interior state.
        - ``"periodic"``: copy state from the opposite side.
        - ``"constant"``: set every state channel to constant_value.
        - ``"water_level"``: impose free-surface elevation
          constant_level while leaving momentum transmissive.
        - ``"wall"``: reflect the normal momentum component.

    constant_value
        State value used for boundaries of type ``"constant"``.

    constant_level
        Imposed free-surface elevation for boundaries of type
        ``"water_level"``.

        This may be a differentiable scalar tensor. Its autograd
        connection is preserved by this function.

    Returns
    -------
    Up
        Padded conservative state.

    zp
        Padded bed elevation.

    Notes
    -----
    The prescribed depth is calculated as:

        h = max(constant_level - bed, 0)

    Consequently, the derivative with respect to constant_level is zero
    wherever constant_level is at or below the local bed elevation.
    """
    boundaries = {
        "left": boundary_left,
        "right": boundary_right,
        "top": boundary_top,
        "bottom": boundary_bottom,
    }

    _validate_inputs(
        U=U,
        bed=bed,
        ng=ng,
        boundaries=boundaries,
    )

    # Replicate padding provides the baseline transmissive condition.
    Up = F.pad(
        U,
        pad=(ng, ng, ng, ng),
        mode="replicate",
    )

    zp = F.pad(
        bed,
        pad=(ng, ng, ng, ng),
        mode="replicate",
    )

    # ================================================================
    # 1. PERIODIC BED VALUES
    # ================================================================
    # Restrict periodic copies to the central part of each side.
    # This avoids the dimension mismatch caused by copying an unpadded
    # array into an edge that includes padded corner cells.

    if boundary_left == "periodic":
        zp[:, :, ng:-ng, :ng] = bed[:, :, :, -ng:]

    if boundary_right == "periodic":
        zp[:, :, ng:-ng, -ng:] = bed[:, :, :, :ng]

    if boundary_top == "periodic":
        zp[:, :, :ng, ng:-ng] = bed[:, :, -ng:, :]

    if boundary_bottom == "periodic":
        zp[:, :, -ng:, ng:-ng] = bed[:, :, :ng, :]

    # ================================================================
    # 2. PREPARE OPTIONAL BOUNDARY VALUES
    # ================================================================

    uses_water_level = any(
        boundary_type in ("water_level", "absorbing")
        for boundary_type in boundaries.values()
    )

    forced_h = None

    if uses_water_level:
        level = _prepare_boundary_value(
            constant_level,
            U,
        )

        # This operation preserves gradients from forced_h back to level.
        #
        # The gradient is zero in dry ghost cells because of the clamp.
        forced_h = torch.clamp(
            level - zp,
            min=0.0,
        )

    uses_constant_value = any(
        boundary_type == "constant"
        for boundary_type in boundaries.values()
    )

    prepared_constant_value = None

    if uses_constant_value:
        prepared_constant_value = _prepare_boundary_value(
            constant_value,
            U,
        )

    # ================================================================
    # 3. LEFT BOUNDARY
    # ================================================================

    if boundary_left == "periodic":
        Up[:, :, ng:-ng, :ng] = U[:, :, :, -ng:]

    elif boundary_left == "constant":
        Up[:, :, :, :ng] = prepared_constant_value

    elif boundary_left == "water_level":
        # Only impose water depth. The momentum components retain the
        # transmissive values created by replicate padding.
        Up[:, 0:1, :, :ng] = forced_h[:, :, :, :ng]

    elif boundary_left == "wall":
        # Reflect x-momentum, hu.
        Up[:, 1:2, :, :ng] = -Up[
            :, 1:2, :, ng:2 * ng
        ].flip(dims=(-1,))

    elif boundary_left == "absorbing":
        # Flather radiation condition for -x direction
        h_ghost = Up[:, 0:1, :, :ng]
        h_ref = forced_h[:, :, :, :ng]
        
        # If the reference boundary is wet, use it for celerity. 
        # If dry, use the interior depth to allow a free outfall.
        h_for_c = torch.where(h_ref > 1e-4, h_ref, h_ghost)
        c = torch.sqrt(9.81 * torch.clamp(h_for_c, min=1e-6)) # Clamped for safe autograd
        # Outward normal is -x
        Up[:, 1:2, :, :ng] = -c * (h_ghost - h_ref)

    # A transmissive boundary needs no additional operation.

    # ================================================================
    # 4. RIGHT BOUNDARY
    # ================================================================

    if boundary_right == "periodic":
        Up[:, :, ng:-ng, -ng:] = U[:, :, :, :ng]

    elif boundary_right == "constant":
        Up[:, :, :, -ng:] = prepared_constant_value

    elif boundary_right == "water_level":
        Up[:, 0:1, :, -ng:] = forced_h[:, :, :, -ng:]

    elif boundary_right == "wall":
        # Reflect x-momentum, hu.
        Up[:, 1:2, :, -ng:] = -Up[
            :, 1:2, :, -2 * ng:-ng
        ].flip(dims=(-1,))

    elif boundary_right == "absorbing":
        h_ghost = Up[:, 0:1, :, -ng:]
        h_ref = forced_h[:, :, :, -ng:]
        h_for_c = torch.where(h_ref > 1e-4, h_ref, h_ghost)
        c = torch.sqrt(9.81 * torch.clamp(h_for_c, min=1e-6))        
        # Outward normal is +x
        Up[:, 1:2, :, -ng:] = c * (h_ghost - h_ref)

    # ================================================================
    # 5. TOP BOUNDARY
    # ================================================================

    if boundary_top == "periodic":
        Up[:, :, :ng, ng:-ng] = U[:, :, -ng:, :]

    elif boundary_top == "constant":
        Up[:, :, :ng, :] = prepared_constant_value

    elif boundary_top == "water_level":
        Up[:, 0:1, :ng, :] = forced_h[:, :, :ng, :]

    elif boundary_top == "wall":
        # Reflect y-momentum, hv.
        Up[:, 2:3, :ng, :] = -Up[
            :, 2:3, ng:2 * ng, :
        ].flip(dims=(-2,))

    elif boundary_top == "absorbing":
        h_ghost = Up[:, 0:1, :ng, :]
        h_ref = forced_h[:, :, :ng, :]
        h_for_c = torch.where(h_ref > 1e-4, h_ref, h_ghost)
        c = torch.sqrt(9.81 * torch.clamp(h_for_c, min=1e-6))        
        # Outward normal is -y
        Up[:, 2:3, :ng, :] = -c * (h_ghost - h_ref)

    # ================================================================
    # 6. BOTTOM BOUNDARY
    # ================================================================

    if boundary_bottom == "periodic":
        Up[:, :, -ng:, ng:-ng] = U[:, :, :ng, :]

    elif boundary_bottom == "constant":
        Up[:, :, -ng:, :] = prepared_constant_value

    elif boundary_bottom == "water_level":
        Up[:, 0:1, -ng:, :] = forced_h[:, :, -ng:, :]

    elif boundary_bottom == "wall":
        # Reflect y-momentum, hv.
        Up[:, 2:3, -ng:, :] = -Up[
            :, 2:3, -2 * ng:-ng, :
        ].flip(dims=(-2,))

    elif boundary_bottom == "absorbing":
        h_ghost = Up[:, 0:1, -ng:, :]
        h_ref = forced_h[:, :, -ng:, :]
        h_for_c = torch.where(h_ref > 1e-4, h_ref, h_ghost)
        c = torch.sqrt(9.81 * torch.clamp(h_for_c, min=1e-6))        
        # Outward normal is +y
        Up[:, 2:3, -ng:, :] = c * (h_ghost - h_ref)

    return Up, zp