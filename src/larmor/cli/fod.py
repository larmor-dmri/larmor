"""The `lar fod` command, which fits FODs by spherical deconvolution."""

import dataclasses
import logging
import typing
from enum import StrEnum
from pathlib import Path

import typer

from larmor._log import LarmorLogger
from larmor.cli.options import (
    BValsOpt,
    BvecFrame,
    BvecFrameOpt,
    BVecsOpt,
    DiffusionArg,
    ForceOpt,
    SHBasis,
    ThreadsOpt,
)
from larmor.cli.outputs import check_outputs, output_prefix, with_suffix
from larmor.io import load_dwi, load_volume, read_mrtrix_response, save_volume

if typing.TYPE_CHECKING:
    from larmor.sd import SDFit, SDSolver
    from larmor.types import Volume

MaskArg = typing.Annotated[
    str,
    typer.Argument(help="Path to a brain mask on the grid of the diffusion volume."),
]
OutputArg = typing.Annotated[
    str,
    typer.Argument(
        help="Prefix of the output files. 'out' writes the FOD to out.nii.gz. A "
        "fit with a NODDI grey matter response (--gm-rf noddi, the default for "
        "mFOD) has two FODs, which are merged into out.nii.gz (see --fod-merge), "
        "or written to out_fod1.nii.gz and out_fod2.nii.gz with --fod-merge none. "
        "A multi-tissue fit also writes the tissue fractions to "
        "out_tissue_fraction.nii.gz.",
    ),
]


class Method(StrEnum):
    """Spherical deconvolution methods of `lar fod`."""

    GRL = "GRL"
    MFOD = "mFOD"
    CSD = "CSD"
    MSMT_CSD = "MSMT-CSD"


class FodMerge(StrEnum):
    """How `lar fod` merges the FODs of a fit with more than one."""

    WS = "ws"
    MAJORITY = "majority"
    NONE = "none"


FodMergeOpt = typing.Annotated[
    FodMerge,
    typer.Option(
        "--fod-merge",
        help="How the FODs of a fit with a NODDI grey matter response are merged "
        "into one: 'ws' adds each FOD times its tissue fraction (mFOD-WS), "
        "'majority' keeps in each voxel the FOD of the anisotropic tissue with "
        "the largest fraction, and 'none' writes each FOD to its own file.",
    ),
]


class Solver(StrEnum):
    """FOD solvers of the GRL and mFOD methods."""

    DRL = "dRL"
    L2NNLS = "L2NNLS"
    LASSO = "LASSO"


def _solver(method: Method, name: Solver | None, reg: float | None) -> SDSolver | None:
    """Returns the solver for --solver and --reg; None keeps the method's own."""
    from larmor.sd import DRL, L2NNLS, Lasso, SDMethod

    solvers: dict[Solver, type[SDSolver]] = {
        Solver.DRL: DRL,
        Solver.L2NNLS: L2NNLS,
        Solver.LASSO: Lasso,
    }
    if method == Method.CSD:
        if name is not None or reg is not None:
            raise typer.BadParameter(
                "CSD has its own solver; --solver and --reg do not apply",
                param_hint="--solver" if name is not None else "--reg",
            )
        return None
    if reg is None:
        return None if name is None else solvers[name]()
    solver = (
        SDMethod(method.value).default_solver() if name is None else solvers[name]()
    )
    try:
        if isinstance(solver, L2NNLS):
            return L2NNLS(reg=reg)
        if isinstance(solver, Lasso):
            return Lasso(reg=reg)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="--reg") from error
    raise typer.BadParameter(
        "applies to the L2NNLS and LASSO solvers only", param_hint="--reg"
    )


MethodOpt = typing.Annotated[
    Method,
    typer.Option(
        "--method",
        case_sensitive=False,
        help="Spherical deconvolution method. GRL is generalised Richardson-Lucy, "
        "mFOD fits one FOD per anisotropic response, CSD is constrained spherical "
        "deconvolution of one shell and MSMT-CSD is multi-shell multi-tissue CSD.",
    ),
]
SolverOpt = typing.Annotated[
    Solver | None,
    typer.Option(
        "--solver",
        case_sensitive=False,
        help="Solver of GRL and mFOD. dRL is damped Richardson-Lucy, L2NNLS is "
        "non-negative least squares with an L2 penalty, and LASSO has an L1 "
        "penalty. Defaults to dRL for GRL and L2NNLS for mFOD.",
        show_default=False,
    ),
]
RegOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--reg",
        help="Weight of the penalty of the L2NNLS and LASSO solvers. Defaults to "
        "the solver's own value.",
        show_default=False,
    ),
]
InnerShellWeightOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--inner-shell-weight",
        help="Weight, above 0 and at most 1, of the measurements outside the "
        "highest shell in a multi-tissue GRL or mFOD fit; the highest shell has "
        "weight 1. Defaults to 0.2 for GRL and 1 for mFOD.",
    ),
]

ResponseOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--response",
        help="White matter response for CSD, a text file of zonal SH coefficients "
        "with one row per shell, as MRtrix3's dwi2response writes it. Without it, "
        "the response is estimated from the single-fibre voxels in the mask.",
        show_default=False,
        exists=True,
        dir_okay=False,
    ),
]

LmaxOpt = typing.Annotated[
    int,
    typer.Option(
        "--lmax",
        help="Maximum spherical harmonic (SH) degree of the FOD, even and at "
        "least 2. GRL and mFOD lower it when the highest shell has too few "
        "directions, and MSMT-CSD lowers it to that of the white matter response; "
        "CSD keeps it.",
    ),
]
WMRFOpt = typing.Annotated[
    tuple[float, float, float] | None,
    typer.Option(
        "--wm-rf",
        help="Eigenvalues in mm^2/s of the white matter response of GRL and mFOD, "
        "the first along the fibre, such as 1.7e-3 0.2e-3 0.2e-3. Without it, the "
        "response is estimated from the data in the mask.",
    ),
]

BasisOpt = typing.Annotated[
    SHBasis,
    typer.Option(
        "--basis",
        help="SH basis to write the FOD in. The file does not record it, so give "
        "the same basis to the commands that read the FOD.",
    ),
]

MKOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--mk",
        help="Mean kurtosis of the white matter response of GRL and mFOD. "
        "Defaults to the kurtosis estimated with the response (0 with fewer than "
        "two shells above b = 500), or to 0 with --wm-rf.",
    ),
]


def _diffusivity_or_none(value: str | float) -> float | None:
    """Parses an isotropic diffusivity, or "none" (any case) for no tissue."""
    # Typer also passes the default, a float, through the parser.
    if isinstance(value, str) and value.lower() == "none":
        return None
    return float(value)


# The --gm-rf value that models grey matter with the NODDI response.
GM_NODDI = "noddi"
# The --gm-rf default: NODDI for mFOD, an isotropic response for GRL.
GM_AUTO = "auto"
GM_ISOTROPIC_DEFAULT = "0.7e-3"


def _gm_response(value: str) -> str | None:
    """Parses --gm-rf, a diffusivity, "none", "noddi" or "auto" (any case).

    The value stays text, since the option takes a word or a number.

    Returns:
        The diffusivity as text, None for "none", or the NODDI or auto marker.
    """
    if value.lower() in (GM_NODDI, GM_AUTO):
        return value.lower()
    diffusivity = _diffusivity_or_none(value)
    return None if diffusivity is None else str(diffusivity)


GMRFOpt = typing.Annotated[
    str | None,
    typer.Option(
        "--gm-rf",
        parser=_gm_response,
        metavar="<float|none|noddi|auto>",
        help=(
            "Grey matter response of GRL and mFOD: an isotropic diffusivity in "
            "mm^2/s, 'none' to leave grey matter out, or 'noddi' for an anisotropic "
            "NODDI response with its own FOD. 'auto' means 'noddi' for mFOD and "
            "0.7e-3 for GRL."
        ),
    ),
]
CSFRFOpt = typing.Annotated[
    float | None,
    typer.Option(
        "--csf-rf",
        parser=_diffusivity_or_none,
        metavar="<float|none>",
        help="Isotropic diffusivity in mm^2/s of the CSF response of GRL and mFOD, "
        "or 'none' to leave CSF out.",
    ),
]


_RESPONSE_HELP = (
    "response for MSMT-CSD, a text file with one row per shell, as MRtrix3's "
    "dwi2response writes it. Without any of the three responses, all three are "
    "estimated from the data in the mask; --gm-response and --csf-response need "
    "--wm-response."
)


WMResponseOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--wm-response",
        help=f"White matter {_RESPONSE_HELP}",
        show_default=False,
        exists=True,
        dir_okay=False,
    ),
]
GMResponseOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--gm-response",
        help=f"Grey matter {_RESPONSE_HELP}",
        show_default=False,
        exists=True,
        dir_okay=False,
    ),
]
CSFResponseOpt = typing.Annotated[
    Path | None,
    typer.Option(
        "--csf-response",
        help=f"CSF {_RESPONSE_HELP}",
        show_default=False,
        exists=True,
        dir_okay=False,
    ),
]


# The options of SDModel's methods, refused with --method CSD and MSMT-CSD.
SD_MODEL_OPTIONS = {
    "wm_rf": "--wm-rf",
    "mk": "--mk",
    "inner_shell_weight": "--inner-shell-weight",
    "gm_rf": "--gm-rf",
    "csf_rf": "--csf-rf",
    "fod_merge": "--fod-merge",
}
# The options that only one method takes.
METHOD_OPTIONS = {
    "response": ("--response", "CSD"),
    "wm_response": ("--wm-response", "MSMT-CSD"),
    "gm_response": ("--gm-response", "MSMT-CSD"),
    "csf_response": ("--csf-response", "MSMT-CSD"),
}


def _given(context: typer.Context, name: str) -> bool:
    """Returns whether an option was given on the command line."""
    source = context.get_parameter_source(name)
    return source is not None and source.name != "DEFAULT"


def _refuse_sd_model_options(context: typer.Context, method: Method) -> None:
    """Refuses the options of GRL and mFOD given on the command line with method."""
    for name, flag in SD_MODEL_OPTIONS.items():
        if _given(context, name):
            raise typer.BadParameter(
                f"does not apply to --method {method.value}", param_hint=flag
            )


def _refuse_other_methods_options(context: typer.Context, method: Method) -> None:
    """Refuses the options of CSD or MSMT-CSD given with another method."""
    for name, (flag, owner) in METHOD_OPTIONS.items():
        if owner != method.value and _given(context, name):
            raise typer.BadParameter(
                f"applies to --method {owner} only", param_hint=flag
            )


def _merges(context: typer.Context, fod_merge: FodMerge, gm_noddi: bool) -> bool:
    """Returns whether the FODs of the fit are merged into one.

    Raises:
        typer.BadParameter: If --fod-merge is given for a fit with one FOD.
    """
    if not gm_noddi and _given(context, "fod_merge"):
        raise typer.BadParameter(
            "applies only to a fit with two FODs (--gm-rf noddi)",
            param_hint="--fod-merge",
        )
    return gm_noddi and fod_merge != FodMerge.NONE


def _fod_paths(prefix: Path, n_aniso: int) -> list[Path]:
    """Returns <prefix>.nii.gz for one FOD, <prefix>_fod1.nii.gz, ... for several."""
    if n_aniso == 1:
        return [with_suffix(prefix, ".nii.gz")]
    return [with_suffix(prefix, f"_fod{i}.nii.gz") for i in range(1, n_aniso + 1)]


def fod(
    context: typer.Context,
    volume: DiffusionArg,
    mask: MaskArg,
    output: OutputArg,
    bvals: BValsOpt = None,
    bvecs: BVecsOpt = None,
    method: MethodOpt = Method.GRL,
    solver: SolverOpt = None,
    reg: RegOpt = None,
    inner_shell_weight: InnerShellWeightOpt = None,
    bvec_frame: BvecFrameOpt = BvecFrame.fsl,
    lmax: LmaxOpt = 8,
    wm_rf: WMRFOpt = None,
    gm_rf: GMRFOpt = GM_AUTO,
    csf_rf: CSFRFOpt = 3.0e-3,
    mk: MKOpt = None,
    response: ResponseOpt = None,
    wm_response: WMResponseOpt = None,
    gm_response: GMResponseOpt = None,
    csf_response: CSFResponseOpt = None,
    basis: BasisOpt = SHBasis.tournier,
    fod_merge: FodMergeOpt = FodMerge.WS,
    force: ForceOpt = False,
    jobs: ThreadsOpt = -1,
) -> None:
    """Estimate fibre orientation distributions (FODs) by spherical deconvolution."""
    # Imported here, so that starting lar and Tab completion do not load scipy.
    from larmor.sd import (
        CSDModel,
        SDModel,
        build_csd_response,
        build_response_functions,
        merge_fods,
    )

    _refuse_other_methods_options(context, method)
    if method == Method.MSMT_CSD:
        _refuse_sd_model_options(context, method)
        if solver is not None or reg is not None:
            raise typer.BadParameter(
                "MSMT-CSD has its own solver; --solver and --reg do not apply",
                param_hint="--solver" if solver is not None else "--reg",
            )
        _fod_msmt(
            volume,
            mask,
            output,
            bvals=bvals,
            bvecs=bvecs,
            bvec_frame=bvec_frame,
            lmax=lmax,
            wm_response=wm_response,
            gm_response=gm_response,
            csf_response=csf_response,
            basis=basis,
            force=force,
            jobs=jobs,
        )
        return

    fod_solver = _solver(method, solver, reg)
    if method == Method.CSD:
        _refuse_sd_model_options(context, method)
        gm_rf = csf_rf = None
    if gm_rf == GM_AUTO:
        gm_rf = GM_NODDI if method == Method.MFOD else GM_ISOTROPIC_DEFAULT
    gm_noddi = gm_rf == GM_NODDI
    merged = _merges(context, fod_merge, gm_noddi)
    gm_diffusivity = None if gm_rf is None or gm_noddi else float(gm_rf)
    prefix = output_prefix(output)
    # One FOD for white matter, and one for grey matter when it is NODDI, unless
    # the two are merged into one.
    sh_outs = _fod_paths(prefix, n_aniso=2 if gm_noddi and not merged else 1)
    frac_out = with_suffix(prefix, "_tissue_fraction.nii.gz")

    outs = list(sh_outs)
    # Fractions are written only by a multi-tissue fit (any isotropic response).
    if gm_rf is not None or csf_rf is not None:
        outs.append(frac_out)

    check_outputs(outs, force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    mask_volume = load_volume(mask)

    sd: CSDModel | SDModel
    if method == Method.CSD:
        rf = (
            None
            if response is None
            else build_csd_response(read_mrtrix_response(response), gtab)
        )
        sd = CSDModel(rf, lmax=lmax, basis=basis.value)
    else:
        rfs = build_response_functions(
            image,
            gtab,
            mask_volume,
            wm_eigen_vals=wm_rf,
            mean_kurtosis=mk,
            gm_diffusivity=gm_diffusivity,
            csf_diffusivity=csf_rf,
            gm_noddi=gm_noddi,
        )
        sd = SDModel(
            rfs=rfs,
            method=method.value,
            solver=fod_solver,
            inner_shell_weight=inner_shell_weight,
            lmax=lmax,
            basis=basis.value,
            progress=LarmorLogger.isEnabledFor(logging.WARNING),
        )
    results = sd.fit(image, gtab, mask=mask_volume, n_jobs=jobs)
    if merged:
        assert results.tissue_fractions is not None  # a NODDI fit is multi-tissue
        mode: typing.Literal["ws", "majority"] = (
            "majority" if fod_merge == FodMerge.MAJORITY else "ws"
        )
        fod = merge_fods(results.fods, results.tissue_fractions, mode)
        results = dataclasses.replace(results, fods=[fod])

    _save_fit(image, results, sh_outs, frac_out, basis)


def _save_fit(
    image: Volume,
    results: SDFit,
    sh_outs: list[Path],
    frac_out: Path,
    basis: SHBasis,
) -> None:
    """Writes the FODs and, for a multi-tissue fit, the fractions."""
    for fod, sh_out in zip(results.fods, sh_outs, strict=True):
        save_volume(image.with_data(fod), sh_out)
        LarmorLogger.info(f"Saved fODF to {sh_out} ({basis.value} basis)")
    if results.tissue_fractions is not None:
        save_volume(image.with_data(results.tissue_fractions), frac_out)
        LarmorLogger.info(f"Saved fractions to {frac_out}")


def _fod_msmt(
    volume: str,
    mask: str,
    output: str,
    *,
    bvals: str | None,
    bvecs: str | None,
    bvec_frame: BvecFrame,
    lmax: int,
    wm_response: Path | None,
    gm_response: Path | None,
    csf_response: Path | None,
    basis: SHBasis,
    force: bool,
    jobs: int,
) -> None:
    """Runs `lar fod --method MSMT-CSD`, with the given or estimated responses."""
    from larmor.sd import MSMTCSDModel, build_shell_response

    responses = [r for r in (wm_response, gm_response, csf_response) if r is not None]
    if responses and wm_response is None:
        raise typer.BadParameter(
            "is needed with --gm-response and --csf-response",
            param_hint="--wm-response",
        )
    prefix = output_prefix(output)
    sh_outs = _fod_paths(prefix, n_aniso=1)
    frac_out = with_suffix(prefix, "_tissue_fraction.nii.gz")
    # Three tissues when estimated; a single response gives a single-tissue fit.
    multi_tissue = not responses or len(responses) > 1
    check_outputs([*sh_outs, frac_out] if multi_tissue else sh_outs, force)

    image, gtab = load_dwi(volume, bvals=bvals, bvecs=bvecs, frame=bvec_frame.value)
    mask_volume = load_volume(mask)
    rfs = None
    if responses:
        rfs = [build_shell_response(read_mrtrix_response(p), gtab) for p in responses]
    sd = MSMTCSDModel(
        rfs,
        lmax=lmax,
        basis=basis.value,
        progress=LarmorLogger.isEnabledFor(logging.WARNING),
    )
    results = sd.fit(image, gtab, mask=mask_volume, n_jobs=jobs)
    _save_fit(image, results, sh_outs, frac_out, basis)
