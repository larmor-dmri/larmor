"""Spherical deconvolution methods and their defaults."""

from enum import StrEnum

from larmor.sd.rf import IsotropicRF, ResponseFunction
from larmor.sd.solvers import DRL, FRACTION_TOL, L2NNLS, SDSolver


class SDMethod(StrEnum):
    """Spherical deconvolution methods.

    GRL fits one anisotropic response function (white matter) and any number of
    isotropic ones (such as grey matter and CSF); its FOD solver is damped
    Richardson-Lucy and its inner shells are weighted 0.2. mFOD fits one or more
    anisotropic response functions, one FOD each, with all shells weighted equally;
    its FOD solver is L2-regularised NNLS.

    References:
        GRL: Guo F, Leemans A, Viergever MA, Dell'Acqua F, De Luca A. Generalized
        Richardson-Lucy (GRL) for analyzing multi-shell diffusion MRI data.
        NeuroImage 218 (2020) 116948. doi:10.1016/j.neuroimage.2020.116948

        mFOD: De Luca A, Guo F, Froeling M, Leemans A. Spherical deconvolution with
        tissue-specific response functions and multi-shell diffusion MRI to estimate
        multiple fiber orientation distributions (mFODs). NeuroImage 222 (2020)
        117206. doi:10.1016/j.neuroimage.2020.117206

        Both follow MRIToolkit's implementation (MRTTrack).
    """

    GRL = "GRL"
    MFOD = "mFOD"

    @classmethod
    def _missing_(cls, value: object) -> SDMethod:
        """Finds a method by its value in any case."""
        if isinstance(value, str):
            for member in cls:
                if member.value.lower() == value.lower():
                    return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(f"Unknown method {value!r}; valid methods are {valid}.")

    def default_solver(self) -> SDSolver:
        """Returns the FOD solver the method uses when none is given."""
        match self:
            case SDMethod.GRL:
                return DRL()
            case SDMethod.MFOD:
                return L2NNLS()

    @property
    def inner_shell_weight(self) -> float:
        """Default weight of the measurements outside the highest shell."""
        match self:
            case SDMethod.GRL:
                return 0.2
            case SDMethod.MFOD:
                return 1.0

    @property
    def _fraction_tolerance(self) -> float:
        """Change in every fraction below which a voxel leaves the multi-tissue loop.

        For mFOD it is small enough that, in practice, a voxel runs every round.
        """
        match self:
            case SDMethod.GRL:
                return FRACTION_TOL
            case SDMethod.MFOD:
                return 1e-7

    def _check_rfs(self, rfs: list[ResponseFunction]) -> None:
        """Checks the number of anisotropic response functions the method takes.

        Raises:
            ValueError: If GRL is not given exactly one anisotropic response function,
                or mFOD none.
        """
        n_aniso = sum(1 for rf in rfs if not isinstance(rf, IsotropicRF))
        match self:
            case SDMethod.GRL if n_aniso != 1:
                raise ValueError(
                    f"{self} takes exactly one anisotropic response function, "
                    f"got {n_aniso}."
                )
            case SDMethod.MFOD if n_aniso < 1:
                raise ValueError(
                    f"{self} takes at least one anisotropic response function."
                )
