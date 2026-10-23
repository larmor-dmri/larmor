"""Reading the transforms of ANTs, FSL and MRtrix3 registrations and applying them.

Transforms map points in world coordinates (RAS+), in mm. A transform read
from a file maps points of the reference space to the input space, the
direction used to resample an image. Streamlines and surfaces move the other
way, so they need the inverse to follow the image.
"""

from larmor.registration.apply import (
    apply_transform,
    transform_file,
    transform_points,
    transform_surface,
    transform_tractogram,
    transform_tractogram_file,
    transform_volume,
)
from larmor.registration.files import (
    load_registration,
    load_transform,
    read_ants_transform,
    read_flirt_transform,
    read_fsl_warp,
    read_mrtrix_transform,
    read_mrtrix_warp,
)
from larmor.registration.transforms import (
    AffineTransform,
    DisplacementField,
    InverseDisplacementField,
    Transform,
    TransformChain,
    TransformPair,
)

__all__ = [
    "AffineTransform",
    "DisplacementField",
    "InverseDisplacementField",
    "Transform",
    "TransformChain",
    "TransformPair",
    "apply_transform",
    "load_registration",
    "load_transform",
    "read_ants_transform",
    "read_flirt_transform",
    "read_fsl_warp",
    "read_mrtrix_transform",
    "read_mrtrix_warp",
    "transform_file",
    "transform_points",
    "transform_surface",
    "transform_tractogram",
    "transform_tractogram_file",
    "transform_volume",
]
