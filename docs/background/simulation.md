---
myst:
  html_meta:
    description: "What Larmor's diffusion MRI simulation computes, from compartment models, tissues and partial volume to relaxation and Rician noise, and what it does not model."
---

# Simulating diffusion MRI

This page describes what {py:func}`~larmor.simulation.simulate` computes, so that you know what the simulated
images contain and what they leave out. The simulation uses the instant engine, which evaluates the signal of every
voxel in closed form, as if the whole acquisition happened at one instant. The sections below describe the signal
of a voxel as a sum over compartments, how tissues share a voxel, how relaxation weights the signal, and how noise
is added. The [simulation guide](../guides/simulation.md) shows how to run a simulation. The simulation builds on
SymDWI, a simulator by Larmor's author, relicensed under the LGPL-3.0-only. Where this page gives a source for a
model or a default value, it is listed under [References](#references); a value marked as a choice has no source.

## The signal

The signal of a voxel is the sum of the signals of its compartments, each weighted by the fraction of the voxel it
fills and by its relaxation. For echo time e and gradient n, it is

```
S(e, n) = sum over compartments c of  f_c * A_c(b_n, g_n) * w_c(e)
```

where `f_c` is the fraction of the voxel that compartment c fills, `A_c` its diffusion attenuation for the
b-value `b_n` and the unit gradient direction `g_n`, and `w_c(e)` its relaxation weight. The sum runs over all
compartments of all tissues in the voxel. An oriented compartment (a stick, a zeppelin, Watson-dispersed sticks or
a cylinder) is evaluated along each fibre direction in the voxel, and the results are weighted by the share of the
voxel each fibre fills.

## Compartments

A compartment is one pool of water that follows a diffusion model. The table below gives the attenuation of each
model, where D is the diffusivity, `b` the b-value in s/mm^2, `g` the gradient direction and `n` the fibre
direction.

| Model | Attenuation | Source |
| --- | --- | --- |
| Stick | `exp(-b D (g.n)^2)` | Behrens et al. 2003 |
| Zeppelin | `exp(-b (D_perp + (D - D_perp) (g.n)^2))` | Kaden et al. 2016, Eq. 5; Novikov et al. 2019, Eq. 3.5 |
| Ball | `exp(-b D)` | Novikov et al. 2019, Eq. 3.5 |
| Powder-averaged stick | `sqrt(pi) / (2 sqrt(b D)) * erf(sqrt(b D))`, the stick averaged over all directions | Kroenke et al. 2004; Kaden et al. 2016 |
| Sphere | Diffusion restricted to an impermeable sphere, in the Gaussian phase approximation for rectangular pulses | Palombo et al. 2020, Eq. 8 |
| Watson-dispersed sticks | The stick averaged over directions spread about the fibre by a Watson density | Zhang et al. 2012 |
| Cylinder | Diffusion restricted to an impermeable cylinder across the fibre, in the Gaussian phase approximation for rectangular pulses, and free along it | Van Gelderen et al. 1994; De Santis et al. 2016, Eq. 3 |
| Dot | `1` | Panagiotaki et al. 2012 |
| Free water | `exp(-b D)`, with D = 3e-3 mm^2/s in the presets | Pasternak et al. 2009; Zhang et al. 2012 |

The stick has diffusion along the fibre only, and the zeppelin is an axially symmetric tensor. The signals of the
sphere and the cylinder depend on the pulse timings and the radius, while the others do not. Compartments do not
exchange water.

The Watson-dispersed sticks describe neurites that do not all point the same way. Their directions n follow the
Watson density `f(n) = exp(kappa (mu.n)^2) / M(1/2, 3/2, kappa)`, centred on the fibre direction mu, where M is
the confluent hypergeometric function that makes f integrate to 1. The concentration kappa sets how tightly the
sticks follow mu. At kappa = 0 they spread evenly over all directions and give the powder-averaged stick, and as
kappa grows they gather along mu and tend to the stick. The signal is the stick's attenuation averaged over f
(Zhang et al. 2012, Eqs. 2 and 3).

The cylinder models an axon of radius R. Along the axon, water diffuses freely, so the cylinder attenuates like
the stick, `exp(-b D (g.n)^2)`. Across it, water is restricted by the wall, and in the Gaussian phase
approximation the logarithm of the signal is a sum over the roots of the derivative of the Bessel function J1
(Van Gelderen et al. 1994, written as in De Santis et al. 2016, Eq. 3). The gradient across the axon carries the
share `1 - (g.n)^2` of the b-value. As R shrinks, the cylinder becomes the stick, and with short pulses far apart
its signal across the axon tends to `exp(-q^2 R^2 / 4)`, with `q = gamma G delta`.

The dot is stationary water trapped in glial cells and other small structures, so its signal is 1 at every b-value.
It was introduced as one of the compartments of a taxonomy of white matter models (Panagiotaki et al. 2012). Free
water is a ball with the diffusivity of water at body temperature, 3e-3 mm^2/s, the value NODDI fixes for its CSF
compartment (Zhang et al. 2012) and SANDI for free water at 37 C (Palombo et al. 2020). It differs from a ball only
in that the ground truth of a simulation counts it. `free_water_fraction.nii.gz` gives the share of each voxel's
volume that free water fills, and `free_water_signal_fraction.nii.gz` (one per echo time) its share of the voxel's
signal at b = 0, in which each compartment is weighted by its proton density and relaxation. The signal fraction is
the quantity that free-water imaging estimates (Pasternak et al. 2009).

## Tissues

A tissue is a set of compartments with fractions that sum to 1, and each compartment has its own relaxation.
Three are predefined, with defaults for 3 T:

- **White matter** follows the standard model, with a stick for the intra-axonal water, a zeppelin for the
  extra-axonal water, and optionally a ball of free water (Novikov et al. 2019).
- **Grey matter** follows SANDI, with a sphere for the somas, a powder-averaged stick for the neurites and a ball
  for the extracellular space (Palombo et al. 2020). It does not depend on the gradient direction.
- **CSF** is free water alone (Pasternak et al. 2009).

Two more presets can be used in their place. **NODDI** describes white or grey matter with three pools (Zhang et
al. 2012, Eq. 1). The neurites are Watson-dispersed sticks with fraction `(1 - v_iso) v_ic`. The space around them
is a zeppelin with fraction `(1 - v_iso) (1 - v_ic)`, and free water has fraction `v_iso`. The zeppelin's
diffusivity across a neurite follows a tortuosity model, `d (1 - v_ic)`, so denser neurites hinder the water around
them more. Averaged over the Watson density, it becomes a zeppelin along the fibre with
`d_par = d - d v_ic (1 - tau1)` and `d_perp = d - d v_ic (1 + tau1) / 2`, where tau1 is the mean squared cosine
between a neurite and the fibre direction (Eqs. 5 to 7). More dispersion therefore lowers the anisotropy of both
the neurites and the space around them. The diffusivities default to the values NODDI fixes, 1.7e-3 mm^2/s along
the neurites and 3e-3 mm^2/s for free water. **Free water** is one free-water compartment, a tissue of its own.

The docstrings of `standard_model_tissue`, `sandi_tissue`, `noddi_tissue`, `free_water_tissue` and `csf_tissue`
give the source or the reason for each default. The standard model's extra-axonal diffusivities follow the NODDI
constraints listed by Novikov et al. (2019, Sec. 3.4.2), which are 1.7e-3 mm^2/s along the fibre and the
tortuosity `(1 - f) 1.7e-3`, about 0.5e-3 mm^2/s, across it at the intra-axonal fraction f = 0.7. A few defaults
are choices without a source, namely the intra-axonal fraction and the SANDI fractions and radius.

## Tissues in a voxel

A voxel is shared between tissues in three steps, and a fraction is never scaled up.

1. Each bundle fills a fraction of the voxel, its occupancy, which is the length of its streamlines in the voxel
   times the cross-section each streamline stands for, divided by the voxel's volume, and at most 1. Where bundles
   together fill more than 1, as in a crossing, their occupancies are scaled down to sum to 1.
2. The tissue maps share what the bundles leave. Where their values add up to more than that, they are scaled down.
3. What remains goes to the background tissue, or gives no signal.

A voxel filled by one tissue therefore has exactly that tissue's signal, and a voxel with a grey matter map value of
0.05 and nothing else is 5% grey matter and 95% background. This is how partial volume is modelled: the signals of
the tissues add in proportion to what they fill, each with its own relaxation weight.

The fibres of a bundle in a voxel are the directions of its streamlines there, grouped in cells of about 7
degrees, each with its share of the bundle's length. A bundle that fans or bends in a voxel therefore has several
fibre directions, and its signal is their average.

## Relaxation

Each compartment has a proton density PD, a T1, a T2 and a T2*, which together set how much it contributes to the
signal. Its relaxation weight at echo time TE, with repetition time TR, is

```
w_c = PD * (1 - exp(-TR / T1)) * exp(-TE / T2)
```

for a spin echo, with T2* in place of T2 for a gradient echo. The proton density is the water content relative to
pure water, and its defaults are approximate choices for white matter, grey matter and CSF
(source to be added).

T2* is shorter than T2 because of reversible dephasing, T2', with `1/T2* = 1/T2 + 1/T2'`. Dephasing comes from the
field in the voxel, so the compartments of a tissue share one T2', derived from the tissue's T2* and the mean T2
of its stick and zeppelin (or from the shared T2* and T2 of the NODDI sticks and zeppelin), and each compartment's
T2* follows from its own T2. This includes the free water of the white matter tissues, whose T2* is therefore much
shorter inside a tissue than in CSF. The T2* of a tissue cannot exceed its mean T2.

The defaults are in vivo values at 3 T: T1 and T2* from Wansapura et al. (1999), and T2 from frontal white and
grey matter in Lu et al. (2005, Table 1).
The longer intra-axonal than extra-axonal T2 of white matter agrees with Veraart et al. (2018). The T1, T2 and T2*
of CSF are choices.

Several echo times share one diffusion block: the attenuation is the same at every echo, and only the relaxation
weight changes.

## The scan

The diffusion weighting is a pair of rectangular gradient pulses of strength G, each delta long and Delta apart, so
that the b-value is `b = gamma^2 G^2 delta^2 (Delta - delta / 3)`. A gradient echo has no refocusing pulse, so it
is assumed to use a bipolar pair instead: a lobe of +G and one of -G, each delta long and Delta apart. Its
effective waveform is the same, so its b-value and its restricted signal are those of the spin echo; only the
relaxation uses T2*.

A protocol is checked when it is made: delta must not exceed Delta, Delta + delta must be shorter than the echo
time, and TR must be longer than the last echo time. It warns when the largest b-value is outside 100 to 50 000
s/mm^2, which is probably a wrong unit. Gradient directions made by `create_gradient_table` are spread by
electrostatic repulsion (Jones et al. 1999).

## Noise

With `snr` set, noise is added to the noise-free signal. Each value S becomes `sqrt((S + n1)^2 + n2^2)`, with n1
and n2 independent and normal with standard deviation sigma. This is the magnitude of a complex signal with
Gaussian noise, which follows a Rician distribution (Gudbjartsson and Patz 1995). It is the noise of a single
coil, so it describes an image combined from several coils only roughly.

There is one sigma for the whole image and all volumes and echoes: `sigma = S0_ref / snr`, with S0_ref the b = 0
signal of a voxel filled with the reference tissue ("WM" unless given) at the first echo time. Thermal noise does
not depend on the signal, so the SNR of a voxel falls with its signal. Where there is no signal the magnitude has
a mean of `sigma * sqrt(pi / 2)`, so the background is not zero, and later echoes are noisier relative to their
signal.

## What the instant engine does not model

Because the instant engine evaluates the signal of each voxel as if the whole acquisition happened at one instant,
it does not model the following effects.

- k-space: sampling, the readout and its duration, partial Fourier, parallel imaging and coil sensitivities.
- Gibbs ringing and the blurring along the phase-encoding direction.
- Distortion from susceptibility, B0 inhomogeneity and eddy currents, ghosting, and motion.
- Exchange of water between compartments, and the dependence of the Gaussian compartments on the diffusion time.
- The slice profile.

## References

- Behrens TEJ, Woolrich MW, Jenkinson M, et al. Characterization and propagation of uncertainty in
  diffusion-weighted MR imaging. Magn Reson Med 50 (2003) 1077-1088.
- Kroenke CD, Ackerman JJH, Yablonskiy DA. On the nature of the NAA diffusion attenuated MR signal in the central
  nervous system. Magn Reson Med 52 (2004) 1052-1059.
- Kaden E, Kelm ND, Carson RP, Does MD, Alexander DC. Multi-compartment microscopic diffusion imaging.
  NeuroImage 139 (2016) 346-359.
- Palombo M, Ianus A, Guerreri M, et al. SANDI: a compartment-based model for non-invasive apparent soma and
  neurite imaging by diffusion MRI. NeuroImage 215 (2020) 116835.
- Zhang H, Schneider T, Wheeler-Kingshott CA, Alexander DC. NODDI: Practical in vivo neurite orientation
  dispersion and density imaging of the human brain. NeuroImage 61 (2012) 1000-1016.
- Van Gelderen P, DesPres D, van Zijl PCM, Moonen CTW. Evaluation of restricted diffusion in cylinders.
  Phosphocreatine in rabbit leg muscle. J Magn Reson B 103 (1994) 255-260.
- De Santis S, Jones DK, Roebroeck A. Including diffusion time dependence in the extra-axonal space improves in
  vivo estimates of axonal diameter and density in human white matter. NeuroImage 130 (2016) 91-103.
- Panagiotaki E, Schneider T, Siow B, Hall MG, Lythgoe MF, Alexander DC. Compartment models of the diffusion MR
  signal in brain white matter: A taxonomy and comparison. NeuroImage 59 (2012) 2241-2254.
- Novikov DS, Fieremans E, Jespersen SN, Kiselev VG. Quantifying brain microstructure with diffusion MRI: Theory
  and parameter estimation. NMR Biomed 32 (2019) e3998.
- Pasternak O, Sochen N, Gur Y, Intrator N, Assaf Y. Free water elimination and mapping from diffusion MRI. Magn
  Reson Med 62 (2009) 717-730.
- Wansapura JP, Holland SK, Dunn RS, Ball WS. NMR relaxation times in the human brain at 3.0 tesla. J Magn Reson
  Imaging 9 (1999) 531-538.
- Veraart J, Novikov DS, Fieremans E. TE dependent Diffusion Imaging (TEdDI) distinguishes between compartmental
  T2 relaxation times. NeuroImage 182 (2018) 360-369.
- Lu H, Nagae-Poetscher LM, Golay X, Lin D, Pomper M, van Zijl PCM. Routine clinical brain MRI sequences for use
  at 3.0 Tesla. J Magn Reson Imaging 22 (2005) 13-22.
- Gudbjartsson H, Patz S. The Rician distribution of noisy MRI data. Magn Reson Med 34 (1995) 910-914.
- Jones DK, Horsfield MA, Simmons A. Optimal strategies for measuring diffusion in anisotropic systems by
  magnetic resonance imaging. Magn Reson Med 42 (1999) 515-525.
