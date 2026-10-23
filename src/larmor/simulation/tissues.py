"""Tissues, which are sums of water compartments, and the predefined tissues."""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

import numpy as np
from scipy.special import dawsn

# How far from 1 the compartment fractions of a tissue may sum.
_FRACTION_TOLERANCE = 1e-6

# Relaxation of brain tissue at 3 T, in ms. T1: in vivo, Wansapura et al. 1999.
# T2: frontal white and grey matter in vivo, Lu et al. 2005, Table 1. T2*: in vivo,
# the mean of the frontal and occipital values of Wansapura et al. 1999.
WM_T1_MS, WM_T2_MS, WM_T2STAR_MS = 832.0, 69.0, (44.7 + 48.4) / 2
GM_T1_MS, GM_T2_MS, GM_T2STAR_MS = 1331.0, 88.0, (51.8 + 41.6) / 2

# The diffusivity of free water at body temperature, mm^2/s, the value NODDI fixes
# for CSF (Zhang et al. 2012) and SANDI for free water at 37 C (Palombo et al. 2020).
FREE_WATER_DIFFUSIVITY = 3.0e-3


class CompartmentModel(StrEnum):
    """The diffusion models a compartment can follow, listed in `Compartment`."""

    STICK = "stick"
    ZEPPELIN = "zeppelin"
    BALL = "ball"
    POWDER_STICK = "powder_stick"
    SPHERE = "sphere"
    WATSON_STICK = "watson_stick"
    CYLINDER = "cylinder"
    DOT = "dot"
    FREE_WATER = "free_water"

    @classmethod
    def _missing_(cls, value: object) -> CompartmentModel:
        """Raises ValueError naming the valid models."""
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown compartment model {value!r}; valid models are {valid}."
        )

    @property
    def code(self) -> int:
        """Returns the integer that stands for the model in the compiled engine."""
        return _MODEL_CODES[self]

    @property
    def is_oriented(self) -> bool:
        """Returns whether the signal depends on a fibre direction."""
        return self in _ORIENTED


# The models whose signal depends on a fibre direction.
_ORIENTED = frozenset(
    {
        CompartmentModel.STICK,
        CompartmentModel.ZEPPELIN,
        CompartmentModel.WATSON_STICK,
        CompartmentModel.CYLINDER,
    }
)

# The code of each model in the compiled engine; free water is a ball there.
_MODEL_CODES = {
    CompartmentModel.STICK: 0,
    CompartmentModel.ZEPPELIN: 1,
    CompartmentModel.BALL: 2,
    CompartmentModel.POWDER_STICK: 3,
    CompartmentModel.SPHERE: 4,
    CompartmentModel.WATSON_STICK: 5,
    CompartmentModel.CYLINDER: 6,
    CompartmentModel.DOT: 7,
    CompartmentModel.FREE_WATER: 2,
}


@dataclass(frozen=True)
class Compartment:
    """One water pool of a tissue, with its diffusion model, fraction and relaxation.

    A tissue is a sum of compartments, such as the water inside axons and the
    water around them. The models and their attenuation for b-value b (s/mm^2),
    unit gradient direction g and unit fibre direction n are:

    - STICK: exp(-b D (g.n)^2), diffusion along the fibre only (Behrens et al.
      2003).
    - ZEPPELIN: exp(-b (D_perp + (D - D_perp) (g.n)^2)), an axially symmetric
      tensor (Kaden et al. 2016, Eq. 5; Novikov et al. 2019, Eq. 3.5). D_perp > D
      is allowed (an oblate tensor), although white matter has D_perp < D.
    - BALL: exp(-b D), free isotropic diffusion (Novikov et al. 2019, Eq. 3.5).
    - POWDER_STICK: sqrt(pi) / (2 sqrt(b D)) erf(sqrt(b D)), the stick averaged
      over all fibre directions (Kroenke et al. 2004; Kaden et al. 2016).
    - SPHERE: diffusion restricted to an impermeable sphere of radius R, in the
      Gaussian phase approximation for rectangular pulses, as SANDI uses for
      the soma (Palombo et al. 2020, Eq. 8). It depends on the pulse timings,
      for a gradient echo those of the bipolar pair that `Protocol` describes.
    - WATSON_STICK: sticks of diffusivity D whose directions spread about the
      fibre direction with the Watson density
      f(n) = exp(kappa (mu.n)^2) / M(1/2, 3/2, kappa), the neurite compartment of
      NODDI (Zhang et al. 2012). kappa = 0 gives the powder-averaged stick, and a
      large kappa the stick.
    - CYLINDER: diffusion restricted to an impermeable cylinder of radius R
      along the fibre, in the Gaussian phase approximation for rectangular pulses
      across it (Van Gelderen et al. 1994; De Santis et al. 2016, Eq. 3) and free
      with D along it. It depends on the pulse timings.
    - DOT: a signal of 1 at every b-value, for stationary water trapped in glial
      cells and other small structures (Panagiotaki et al. 2012). D is unused.
    - FREE_WATER: exp(-b D), a ball that the ground truth counts as free water,
      as in free-water imaging (Pasternak et al. 2009).

    A compartment's signal at b = 0 is PD (1 - exp(-TR / T1)) exp(-TE / T2) for a
    spin echo, with T2* in place of T2 for a gradient echo.

    Args:
        model: The diffusion model.
        fraction: The compartment's share of its tissue's volume, in [0, 1].
        diffusivity: D in mm^2/s; for the stick and the zeppelin, the
            diffusivity along the fibre.
        t1_ms: T1 in ms.
        t2_ms: T2 in ms.
        t2star_ms: T2* in ms, at most T2.
        proton_density: The water content relative to pure water, in [0, 1].
            Defaults to 1.0.
        perpendicular_diffusivity: The zeppelin's D_perp across the fibre, in
            mm^2/s. Defaults to 0.0; only the zeppelin uses it.
        radius_um: The radius of the sphere or the cylinder in um. Defaults to
            0.0; only the sphere and the cylinder use it.
        kappa: The Watson concentration of WATSON_STICK, >= 0. Defaults to 0.0;
            only WATSON_STICK uses it.

    Raises:
        ValueError: If the model is unknown or a value is out of range.

    References:
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical
        in vivo neurite orientation dispersion and density imaging of the human
        brain. NeuroImage 61 (2012) 1000-1016.
        Van Gelderen P, DesPres D, van Zijl PCM, Moonen CTW. Evaluation of
        restricted diffusion in cylinders. Phosphocreatine in rabbit leg muscle.
        J Magn Reson B 103 (1994) 255-260.
        De Santis S, Jones DK, Roebroeck A. Including diffusion time dependence in
        the extra-axonal space improves in vivo estimates of axonal diameter and
        density in human white matter. NeuroImage 130 (2016) 91-103.
        Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain
        microstructure with diffusion MRI: Theory and parameter estimation. NMR
        Biomed 32 (2019) e3998.
        Panagiotaki E, Schneider T, Siow B, Hall MG, Lythgoe MF, Alexander DC.
        Compartment models of the diffusion MR signal in brain white matter: A
        taxonomy and comparison. NeuroImage 59 (2012) 2241-2254.
        Pasternak O, Sochen N, Gur Y, Intrator N, Assaf Y. Free water elimination
        and mapping from diffusion MRI. Magn Reson Med 62 (2009) 717-730.
        Behrens TEJ, Woolrich MW, Jenkinson M, et al. Characterization and
        propagation of uncertainty in diffusion-weighted MR imaging. Magn Reson Med
        50 (2003) 1077-1088.
        Kroenke CD, Ackerman JJH, Yablonskiy DA. On the nature of the NAA diffusion
        attenuated MR signal in the central nervous system. Magn Reson Med 52
        (2004) 1052-1059.
        Kaden E, Kelm ND, Carson RP, Does MD, Alexander DC. Multi-compartment
        microscopic diffusion imaging. NeuroImage 139 (2016) 346-359.
        Palombo M, Ianus A, Guerreri M, et al. SANDI: a compartment-based model for
        non-invasive apparent soma and neurite imaging by diffusion MRI. NeuroImage
        215 (2020) 116835.
    """

    model: CompartmentModel
    fraction: float
    diffusivity: float
    t1_ms: float = field(kw_only=True)
    t2_ms: float = field(kw_only=True)
    t2star_ms: float = field(kw_only=True)
    proton_density: float = field(default=1.0, kw_only=True)
    perpendicular_diffusivity: float = field(default=0.0, kw_only=True)
    radius_um: float = field(default=0.0, kw_only=True)
    kappa: float = field(default=0.0, kw_only=True)

    def __post_init__(self) -> None:
        """Checks the model and the ranges of the values."""
        object.__setattr__(self, "model", CompartmentModel(self.model))
        values = {
            "fraction": self.fraction,
            "diffusivity": self.diffusivity,
            "t1_ms": self.t1_ms,
            "t2_ms": self.t2_ms,
            "t2star_ms": self.t2star_ms,
            "proton_density": self.proton_density,
            "perpendicular_diffusivity": self.perpendicular_diffusivity,
            "radius_um": self.radius_um,
            "kappa": self.kappa,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}.")
        if self.fraction > 1 or self.proton_density > 1:
            raise ValueError("fraction and proton_density must be at most 1.")
        if min(self.t1_ms, self.t2_ms, self.t2star_ms) <= 0:
            raise ValueError("T1, T2 and T2* must be positive.")
        if self.t2star_ms > self.t2_ms:
            raise ValueError(
                f"T2* ({self.t2star_ms} ms) must not exceed T2 ({self.t2_ms} ms)."
            )
        restricted = (CompartmentModel.SPHERE, CompartmentModel.CYLINDER)
        if self.model in restricted and self.radius_um <= 0:
            raise ValueError(f"A {self.model.value} needs a positive radius_um.")

    @property
    def is_oriented(self) -> bool:
        """Returns whether the signal depends on the fibre direction."""
        return self.model.is_oriented

    @property
    def engine_parameters(self) -> tuple[float, float, float, float]:
        """Returns D, D_perp, radius_um and kappa, in the compiled engine's order."""
        return (
            self.diffusivity,
            self.perpendicular_diffusivity,
            self.radius_um,
            self.kappa,
        )


@dataclass(frozen=True)
class Tissue:
    """A named tissue made of compartments whose fractions sum to 1.

    Args:
        name: The tissue's name, such as "WM"; bundles and tissue maps refer to it.
        compartments: Its compartments.

    Raises:
        ValueError: If the name is empty, there are no compartments, or the
            fractions do not sum to 1 (within 1e-6).
    """

    name: str
    compartments: tuple[Compartment, ...]

    def __post_init__(self) -> None:
        """Checks the name and that the fractions sum to 1."""
        object.__setattr__(self, "compartments", tuple(self.compartments))
        if not self.name:
            raise ValueError("A tissue needs a name.")
        if not self.compartments:
            raise ValueError(f"Tissue {self.name!r} has no compartments.")
        total = math.fsum(c.fraction for c in self.compartments)
        if abs(total - 1) > _FRACTION_TOLERANCE:
            raise ValueError(
                f"The fractions of tissue {self.name!r} sum to {total}, not 1."
            )

    @property
    def is_oriented(self) -> bool:
        """Returns whether a compartment needs a fibre direction.

        Only bundles can hold such a tissue.
        """
        return any(c.is_oriented for c in self.compartments)

    def to_dict(self) -> dict[str, object]:
        """Returns the tissue as plain values that can be saved as JSON.

        Returns:
            A dict with the name and a list of compartments.
        """
        return {
            "name": self.name,
            "compartments": [
                {k: str(v) if k == "model" else v for k, v in vars(c).items()}
                for c in self.compartments
            ],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Tissue:
        """Builds a tissue from the values `to_dict` gives.

        Args:
            data: The dict.

        Returns:
            The tissue.

        Raises:
            TypeError: If the compartments are not a list of objects.
            ValueError: If a model is unknown or a value is out of range.
        """
        raw = data["compartments"]
        if not isinstance(raw, list) or not all(isinstance(c, dict) for c in raw):
            raise TypeError("A tissue's compartments must be a list of objects.")
        compartments = tuple(
            Compartment(**(c | {"model": CompartmentModel(c.get("model"))}))
            for c in raw
        )
        return cls(str(data["name"]), compartments)


def _t2star_from_prime(t2_ms: float, t2_prime_ms: float) -> float:
    """Returns T2* from T2 and T2', 1/T2* = 1/T2 + 1/T2'."""
    return 1 / (1 / t2_ms + 1 / t2_prime_ms)


def _t2_prime(t2_ms: float, t2star_ms: float) -> float:
    """Returns T2' from T2 and T2*, 1/T2' = 1/T2* - 1/T2; infinite when T2* >= T2."""
    rate = 1 / t2star_ms - 1 / t2_ms
    return math.inf if rate <= 0 else 1 / rate


def _in_tissue(water: Compartment, fraction: float, t2_prime_ms: float) -> Compartment:
    """Places a free-water compartment in a tissue, sharing the tissue's T2'.

    Returns:
        `water` with `fraction` and T2* = 1 / (1/T2 + 1/T2').
    """
    t2star_ms = _t2star_from_prime(water.t2_ms, t2_prime_ms)
    return replace(water, fraction=fraction, t2star_ms=t2star_ms)


def standard_model_tissue(
    *,
    name: str = "WM",
    intra_fraction: float = 0.7,
    free_water_fraction: float = 0.0,
    intra_diffusivity: float = 2.3e-3,
    extra_parallel_diffusivity: float = 1.7e-3,
    extra_perpendicular_diffusivity: float = 0.5e-3,
    free_water_diffusivity: float = 3.0e-3,
    intra_t2_ms: float = 80.0,
    extra_t2_ms: float = 45.0,
    t1_ms: float = WM_T1_MS,
    t2star_ms: float = WM_T2STAR_MS,
    proton_density: float = 0.7,
) -> Tissue:
    """Makes a standard model tissue: stick, zeppelin and optional free water.

    The water inside the axons is a stick, and the water around them a zeppelin
    with the fraction 1 - intra_fraction - free_water_fraction. Free water is an
    optional FREE_WATER compartment with the parameters of `csf_tissue` apart
    from T2*. The relaxation defaults assume 3 T.

    `t2star_ms` is the tissue's T2*, not each compartment's. Every compartment
    shares the tissue's T2' = 1 / (1/T2* - 1/T2_mean), where T2_mean is the mean
    T2 of the stick and the zeppelin weighted by their fractions, and gets
    T2*_c = 1 / (1/T2_c + 1/T2').

    Sources of the defaults:

    - intra_diffusivity 2.3e-3 mm^2/s: a choice close to the in vivo
      intra-axonal diffusivity of 2.25 um^2/ms measured by Dhital et al. (2019);
      NODDI fixes 1.7e-3.
    - extra-axonal diffusivities 1.7e-3 and 0.5e-3 mm^2/s: they follow the NODDI
      constraints listed by Novikov et al. (2019, Sec. 3.4.2), D_e_par = D_a =
      1.7 um^2/ms and the tortuosity D_e_perp = (1 - f) D_e_par, about
      0.5 um^2/ms at f = 0.7.
    - intra_fraction 0.7: a choice.
    - T2 of 80 ms (intra) and 45 ms (extra): choices around the T2 of 69 ms of
      frontal white matter at 3 T in vivo (Lu et al. 2005, Table 1); that the
      intra-axonal T2 is the longer agrees with Veraart et al. (2018).
    - T1 832 ms: white matter at 3 T in vivo (Wansapura et al. 1999).
    - t2star_ms 46.6 ms: white matter at 3 T in vivo, the mean of the frontal
      and occipital values of Wansapura et al. (1999).
    - proton_density 0.7: an approximate choice for white matter water content
      (source to be added).

    Args:
        name: The tissue's name. Defaults to "WM".
        intra_fraction: The stick fraction. Defaults to 0.7.
        free_water_fraction: The free-water fraction. Defaults to 0.0, no free
            water.
        intra_diffusivity: The stick's D, mm^2/s. Defaults to 2.3e-3.
        extra_parallel_diffusivity: The zeppelin's D along the fibre, mm^2/s.
            Defaults to 1.7e-3.
        extra_perpendicular_diffusivity: The zeppelin's D across the fibre,
            mm^2/s. Defaults to 0.5e-3.
        free_water_diffusivity: The free water's D, mm^2/s. Defaults to 3.0e-3.
        intra_t2_ms: The stick's T2, ms. Defaults to 80.
        extra_t2_ms: The zeppelin's T2, ms. Defaults to 45.
        t1_ms: T1 of the stick and the zeppelin, ms. Defaults to 832.
        t2star_ms: The tissue's T2*, ms, at most T2_mean. Defaults to 46.6.
        proton_density: The proton density of the stick and the zeppelin.
            Defaults to 0.7.

    Returns:
        The tissue.

    Raises:
        ValueError: If the fractions are negative or exceed 1 together, or
            t2star_ms is not positive or exceeds T2_mean.

    References:
        Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain
        microstructure with diffusion MRI: Theory and parameter estimation. NMR
        Biomed 32 (2019) e3998.
        Dhital B, Reisert M, Kellner E, Kiselev VG. Intra-axonal diffusivity in
        brain white matter. NeuroImage 189 (2019) 543-550.
        Veraart J, Novikov DS, Fieremans E. TE dependent Diffusion Imaging (TEdDI)
        distinguishes between compartmental T2 relaxation times. NeuroImage 182
        (2018) 360-369.
        Wansapura JP, Holland SK, Dunn RS, Ball WS. NMR relaxation times in the
        human brain at 3.0 tesla. J Magn Reson Imaging 9 (1999) 531-538.
        Lu H, Nagae-Poetscher LM, Golay X, Lin D, Pomper M, van Zijl PCM. Routine
        clinical brain MRI sequences for use at 3.0 Tesla. J Magn Reson Imaging
        22 (2005) 13-22. doi:10.1002/jmri.20356
    """
    extra_fraction = 1 - intra_fraction - free_water_fraction
    if min(intra_fraction, free_water_fraction, extra_fraction) < 0:
        raise ValueError(
            "intra_fraction and free_water_fraction must be >= 0 and sum to at most 1."
        )
    weights = intra_fraction + extra_fraction
    t2_mean = (
        (intra_fraction * intra_t2_ms + extra_fraction * extra_t2_ms) / weights
        if weights > 0
        else max(intra_t2_ms, extra_t2_ms)
    )
    if not 0 < t2star_ms <= t2_mean:
        raise ValueError(
            f"t2star_ms must be positive and at most the mean T2 {t2_mean:.1f} ms, "
            f"got {t2star_ms}."
        )
    t2_prime_ms = _t2_prime(t2_mean, t2star_ms)
    compartments = [
        Compartment(
            CompartmentModel.STICK,
            intra_fraction,
            intra_diffusivity,
            t1_ms=t1_ms,
            t2_ms=intra_t2_ms,
            t2star_ms=_t2star_from_prime(intra_t2_ms, t2_prime_ms),
            proton_density=proton_density,
        ),
        Compartment(
            CompartmentModel.ZEPPELIN,
            extra_fraction,
            extra_parallel_diffusivity,
            perpendicular_diffusivity=extra_perpendicular_diffusivity,
            t1_ms=t1_ms,
            t2_ms=extra_t2_ms,
            t2star_ms=_t2star_from_prime(extra_t2_ms, t2_prime_ms),
            proton_density=proton_density,
        ),
    ]
    if free_water_fraction > 0:
        (ball,) = csf_tissue(diffusivity=free_water_diffusivity).compartments
        compartments.append(_in_tissue(ball, free_water_fraction, t2_prime_ms))
    return Tissue(name, tuple(compartments))


def sandi_tissue(
    *,
    name: str = "GM",
    neurite_fraction: float = 0.5,
    soma_fraction: float = 0.15,
    neurite_diffusivity: float = 2.3e-3,
    soma_diffusivity: float = 3.0e-3,
    soma_radius_um: float = 6.0,
    extracellular_diffusivity: float = 1.5e-3,
    t1_ms: float = GM_T1_MS,
    t2_ms: float = GM_T2_MS,
    t2star_ms: float = GM_T2STAR_MS,
    proton_density: float = 0.8,
) -> Tissue:
    """Makes a SANDI grey matter tissue: sphere, powder-averaged stick and ball.

    The neurites are sticks averaged over all directions, the cell bodies
    (somas) impermeable spheres, and the extracellular space a ball with the
    fraction 1 - neurite_fraction - soma_fraction. The tissue needs no fibre
    direction. The three compartments share T1, T2, T2* and proton density. The
    relaxation defaults assume 3 T.

    Sources of the defaults:

    - soma_diffusivity 3.0e-3 mm^2/s: the value SANDI fixes (Palombo et al. 2020).
    - soma_radius_um 6, the fractions 0.5 (neurite), 0.15 (soma) and 0.35
      (extracellular), extracellular_diffusivity 1.5e-3 mm^2/s: choices; the
      paper gives no single default.
    - neurite_diffusivity 2.3e-3 mm^2/s: a choice, the same as the white matter
      stick, close to the in vivo intra-axonal diffusivity of 2.25 um^2/ms
      measured by Dhital et al. (2019).
    - T1 1331 ms: grey matter at 3 T in vivo (Wansapura et al. 1999).
    - T2 88 ms: frontal grey matter at 3 T in vivo (Lu et al. 2005, Table 1).
    - T2* 46.7 ms: the mean of the frontal and occipital grey matter values at
      3 T (Wansapura et al. 1999).
    - proton_density 0.8: an approximate choice for grey matter water content
      (source to be added).

    Args:
        name: The tissue's name. Defaults to "GM".
        neurite_fraction: The powder-averaged stick fraction. Defaults to 0.5.
        soma_fraction: The sphere fraction. Defaults to 0.15.
        neurite_diffusivity: The stick's D, mm^2/s. Defaults to 2.3e-3.
        soma_diffusivity: The D of the water inside the sphere, mm^2/s.
            Defaults to 3.0e-3.
        soma_radius_um: The sphere radius, um. Defaults to 6.
        extracellular_diffusivity: The ball's D, mm^2/s. Defaults to 1.5e-3.
        t1_ms: T1 of every compartment, ms. Defaults to 1331.
        t2_ms: T2 of every compartment, ms. Defaults to 88.
        t2star_ms: T2* of every compartment, ms. Defaults to 46.7.
        proton_density: The proton density of every compartment. Defaults to
            0.8.

    Returns:
        The tissue.

    Raises:
        ValueError: If the fractions are negative or exceed 1 together.

    References:
        Palombo M, Ianus A, Guerreri M, et al. SANDI: a compartment-based model for
        non-invasive apparent soma and neurite imaging by diffusion MRI. NeuroImage
        215 (2020) 116835.
        Dhital B, Reisert M, Kellner E, Kiselev VG. Intra-axonal diffusivity in
        brain white matter. NeuroImage 189 (2019) 543-550.
        Wansapura JP, Holland SK, Dunn RS, Ball WS. NMR relaxation times in the
        human brain at 3.0 tesla. J Magn Reson Imaging 9 (1999) 531-538.
        Lu H, Nagae-Poetscher LM, Golay X, Lin D, Pomper M, van Zijl PCM. Routine
        clinical brain MRI sequences for use at 3.0 Tesla. J Magn Reson Imaging
        22 (2005) 13-22. doi:10.1002/jmri.20356
    """
    extracellular_fraction = 1 - neurite_fraction - soma_fraction
    if min(neurite_fraction, soma_fraction, extracellular_fraction) < 0:
        raise ValueError(
            "neurite_fraction and soma_fraction must be >= 0 and sum to at most 1."
        )
    relaxation = {
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "t2star_ms": t2star_ms,
        "proton_density": proton_density,
    }
    return Tissue(
        name,
        (
            Compartment(
                CompartmentModel.SPHERE,
                soma_fraction,
                soma_diffusivity,
                radius_um=soma_radius_um,
                **relaxation,
            ),
            Compartment(
                CompartmentModel.POWDER_STICK,
                neurite_fraction,
                neurite_diffusivity,
                **relaxation,
            ),
            Compartment(
                CompartmentModel.BALL,
                extracellular_fraction,
                extracellular_diffusivity,
                **relaxation,
            ),
        ),
    )


def free_water_tissue(
    *,
    name: str = "FW",
    diffusivity: float = FREE_WATER_DIFFUSIVITY,
    t1_ms: float = 4000.0,
    t2_ms: float = 2000.0,
    t2star_ms: float = 400.0,
    proton_density: float = 1.0,
) -> Tissue:
    """Makes a free water tissue, one FREE_WATER compartment with signal exp(-b D).

    The ground truth counts it both as a volume fraction and as a signal
    fraction, its share of the voxel's signal at b = 0, which is what free-water
    imaging estimates (Pasternak et al. 2009). With the defaults the signal
    fraction is larger than the volume fraction, because free water has a higher
    proton density and a longer T2 than tissue. The default diffusivity is the
    value NODDI fixes for its CSF compartment (Zhang et al. 2012). The proton
    density and relaxation times are choices for 3 T, without a verified source.

    Args:
        name: The tissue's name. Defaults to "FW".
        diffusivity: The diffusivity, mm^2/s. Defaults to 3.0e-3.
        t1_ms: T1, ms. Defaults to 4000.
        t2_ms: T2, ms. Defaults to 2000.
        t2star_ms: T2*, ms. Defaults to 400.
        proton_density: The proton density. Defaults to 1.0.

    Returns:
        The tissue.

    References:
        Pasternak O, Sochen N, Gur Y, Intrator N, Assaf Y. Free water elimination
        and mapping from diffusion MRI. Magn Reson Med 62 (2009) 717-730.
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical
        in vivo neurite orientation dispersion and density imaging of the human
        brain. NeuroImage 61 (2012) 1000-1016.
    """
    return Tissue(
        name,
        (
            Compartment(
                CompartmentModel.FREE_WATER,
                1.0,
                diffusivity,
                t1_ms=t1_ms,
                t2_ms=t2_ms,
                t2star_ms=t2star_ms,
                proton_density=proton_density,
            ),
        ),
    )


def csf_tissue(*, name: str = "CSF", **values: float) -> Tissue:
    """Makes a cerebrospinal fluid tissue, `free_water_tissue` named "CSF".

    Args:
        name: The tissue's name. Defaults to "CSF".
        **values: The other arguments of `free_water_tissue`.

    Returns:
        The tissue.
    """
    return free_water_tissue(name=name, **values)


def _watson_tau(kappa: float) -> float:
    """Returns tau1, the mean squared cosine of a Watson density to its mean direction.

    tau1 = int_0^1 u^2 exp(kappa u^2) du / int_0^1 exp(kappa u^2) du (Zhang et
    al. 2012, Eq. 7).

    Raises:
        ValueError: If kappa is negative or not finite.
    """
    # Up to kappa = 1 by quadrature, since the closed form with Dawson's integral
    # (Eq. 8) loses digits there.
    if not (math.isfinite(kappa) and kappa >= 0):
        raise ValueError(f"kappa must be finite and >= 0, got {kappa}.")
    if kappa <= 1:
        nodes, weights = np.polynomial.legendre.leggauss(32)
        u = 0.5 * (nodes + 1)
        density = weights * np.exp(kappa * u * u)
        return float(density @ (u * u) / density.sum())
    root = math.sqrt(kappa)
    return -1 / (2 * kappa) + 1 / (2 * float(dawsn(root)) * root)


def noddi_tissue(
    *,
    name: str = "WM",
    intra_fraction: float = 0.6,
    free_water_fraction: float = 0.05,
    kappa: float = 8.0,
    diffusivity: float = 1.7e-3,
    free_water_diffusivity: float = FREE_WATER_DIFFUSIVITY,
    t1_ms: float = WM_T1_MS,
    t2_ms: float = WM_T2_MS,
    t2star_ms: float = WM_T2STAR_MS,
    proton_density: float = 0.7,
) -> Tissue:
    """Makes a NODDI white matter tissue: Watson sticks, zeppelin and free water.

    The neurites are sticks of diffusivity d dispersed by a Watson density of
    concentration kappa (Zhang et al. 2012, Eqs. 2 and 3), and the extra-neurite
    space is a zeppelin whose diffusivities follow from d by the tortuosity model
    averaged over the same density (Eqs. 4 to 6, with tau1 from Eq. 7). Free
    water has the relaxation of `free_water_tissue`, apart from T2*. The
    orientation dispersion index is OD = (2 / pi) arctan(1 / kappa) (Eq. 9).

    `intra_fraction` (v_ic) and `free_water_fraction` (v_iso) are volume
    fractions. A NODDI fit recovers signal fractions instead, which differ with
    the defaults because free water has a higher proton density and a longer T2.

    The sticks and the zeppelin share T1, T2, T2* and proton density; the free
    water shares the tissue's T2' = 1 / (1/T2* - 1/T2) and gets
    T2*_fw = 1 / (1/T2_fw + 1/T2'). The relaxation defaults assume 3 T.

    Sources of the defaults:

    - diffusivity 1.7e-3 and free_water_diffusivity 3.0e-3 mm^2/s: the values
      NODDI fixes (Zhang et al. 2012).
    - intra_fraction 0.6, free_water_fraction 0.05 and kappa 8 (OD about 0.08):
      choices for coherent white matter.
    - T1 832 ms and T2* 46.6 ms: white matter at 3 T in vivo (Wansapura et al.
      1999); T2 69 ms: frontal white matter at 3 T in vivo (Lu et al. 2005,
      Table 1).
    - proton_density 0.7: an approximate choice for white matter water content
      (source to be added).

    Args:
        name: The tissue's name. Defaults to "WM".
        intra_fraction: v_ic, the neurites' share of the tissue outside free
            water. Defaults to 0.6.
        free_water_fraction: v_iso, the free-water share. Defaults to 0.05.
        kappa: The Watson concentration, >= 0. Defaults to 8.
        diffusivity: d, the diffusivity along the neurites, mm^2/s. Defaults
            to 1.7e-3.
        free_water_diffusivity: The free water's D, mm^2/s. Defaults to 3.0e-3.
        t1_ms: T1 of the sticks and the zeppelin, ms. Defaults to 832.
        t2_ms: T2 of the sticks and the zeppelin, ms. Defaults to 69.
        t2star_ms: T2* of the sticks and the zeppelin, ms. Defaults to 46.6.
        proton_density: The proton density of the sticks and the zeppelin.
            Defaults to 0.7.

    Returns:
        The tissue, with free water only if free_water_fraction > 0.

    Raises:
        ValueError: If a fraction is outside [0, 1] or a value is out of range.

    References:
        Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical
        in vivo neurite orientation dispersion and density imaging of the human
        brain. NeuroImage 61 (2012) 1000-1016.
        Wansapura JP, Holland SK, Dunn RS, Ball WS. NMR relaxation times in the
        human brain at 3.0 tesla. J Magn Reson Imaging 9 (1999) 531-538.
        Lu H, Nagae-Poetscher LM, Golay X, Lin D, Pomper M, van Zijl PCM. Routine
        clinical brain MRI sequences for use at 3.0 Tesla. J Magn Reson Imaging
        22 (2005) 13-22. doi:10.1002/jmri.20356
    """
    if not (0 <= intra_fraction <= 1 and 0 <= free_water_fraction <= 1):
        raise ValueError(
            "intra_fraction and free_water_fraction must each be in [0, 1]."
        )
    tau = _watson_tau(kappa)
    relaxation = {
        "t1_ms": t1_ms,
        "t2_ms": t2_ms,
        "t2star_ms": t2star_ms,
        "proton_density": proton_density,
    }
    tissue_share = 1 - free_water_fraction
    compartments = [
        Compartment(
            CompartmentModel.WATSON_STICK,
            tissue_share * intra_fraction,
            diffusivity,
            kappa=kappa,
            **relaxation,
        ),
        Compartment(
            CompartmentModel.ZEPPELIN,
            tissue_share * (1 - intra_fraction),
            diffusivity - diffusivity * intra_fraction * (1 - tau),
            perpendicular_diffusivity=(
                diffusivity - diffusivity * intra_fraction * (1 + tau) / 2
            ),
            **relaxation,
        ),
    ]
    if free_water_fraction > 0:
        (water,) = free_water_tissue(diffusivity=free_water_diffusivity).compartments
        t2_prime_ms = _t2_prime(t2_ms, t2star_ms)
        compartments.append(_in_tissue(water, free_water_fraction, t2_prime_ms))
    return Tissue(name, tuple(compartments))


def default_tissues() -> tuple[Tissue, ...]:
    """Returns the predefined tissues WM (standard model), GM (SANDI) and CSF.

    A new `Phantom` starts with these tissues.

    Returns:
        The three tissues with their default parameters, for 3 T.
    """
    return (standard_model_tissue(), sandi_tissue(), csf_tissue())
