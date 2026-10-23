---
myst:
  html_meta:
    description: "The public diffusion MRI datasets that Larmor downloads for its examples, with their sizes, licences and citations, and where they are stored."
---

# Example data

The module `larmor.data` downloads public diffusion MRI datasets. You can use them to try Larmor, and the examples
in this documentation use them too. Each `fetch_` function downloads its dataset the first time it is called,
checks every file against its SHA-256 checksum, and returns the paths of the files in a dictionary. Later calls
return the cached copy straight away.

```python
from larmor.data import fetch_ismrm2015
from larmor.io import load_dwi

files = fetch_ismrm2015()
dwi, gtab = load_dwi(files["dwi"], files["bvals"], files["bvecs"])
```

## Datasets

| Function | Contents | Size | Licence |
| --- | --- | --- | --- |
| `fetch_ismrm2015` | The ISMRM 2015 Tractography Challenge data, simulated from bundles traced in a real brain: DWI at 2 mm with one b=0 image and 32 directions at b=1000 s/mm^2, with noise and artefacts, and a T1-like image at 1 mm | 41 MB | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| `fetch_cfin_multishell` | One healthy volunteer at 3 T: DWI at 2.5 mm with one b=0 image and 33 directions at each of 15 b-values from 200 to 3000 s/mm^2 (raw), and a defaced 1 mm T1 | 197 MB | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) |
| `fetch_stanford_hardi` | One healthy volunteer at 3 T: DWI at 2 mm with 10 b=0 images and 150 directions at b=2000 s/mm^2 (motion corrected), a T1 and a FreeSurfer parcellation on the same grid | 93 MB | [PDDL 1.0](https://opendatacommons.org/licenses/pddl/1-0/) |

The datasets store their b-vectors in different frames, so they are not all loaded the same way. Loaded in the
wrong frame, the fibre directions come out mirrored.

- **CFIN** stores them as FSL bvecs, the default of `load_dwi`, so they load as in the code above.
- **ISMRM 2015** also stores them as FSL bvecs, but the published b-vectors have the wrong sign on their x
  components, as the challenge's data page notes. `fetch_ismrm2015` flips that sign when it extracts the file, so
  the `dwi.bvecs` in the cache is correct for any program that reads FSL bvecs, and it loads as in the code above.
  The other files are as published, only renamed.
- **Stanford** stores them in world coordinates, so load them with `frame="world"`:
  `load_dwi(files["dwi"], files["bvals"], files["bvecs"], frame="world")`.

When in doubt, the direction of the fibres beside the corpus callosum shows which frame is right.

Please cite the dataset you use:

- ISMRM 2015 Tractography Challenge: Maier-Hein KH, Neher PF, Houde J-C, et al. The challenge of mapping the human
  connectome based on diffusion tractography. Nature Communications 8 (2017) 1349.
  [doi:10.1038/s41467-017-01285-x](https://doi.org/10.1038/s41467-017-01285-x). The data is published on Zenodo,
  [doi:10.5281/zenodo.572345](https://doi.org/10.5281/zenodo.572345), under CC BY 4.0. The brain images on this
  website are made from it.
- CFIN multi-shell: Hansen B, Jespersen SN. Data for evaluation of fast kurtosis strategies, b-value optimization and
  exploration of diffusion MRI contrast. Scientific Data 3 (2016) 160072.
  [doi:10.1038/sdata.2016.72](https://doi.org/10.1038/sdata.2016.72). The data is published on Dryad,
  [doi:10.5061/dryad.9bc43](https://doi.org/10.5061/dryad.9bc43), and Larmor downloads the copy in the University
  of Washington's ResearchWorks archive ([hdl:1773/38488](http://hdl.handle.net/1773/38488)).
- Stanford HARDI: Rokem A, Yeatman J, Pestilli F, Wandell B. High angular resolution diffusion MRI. Stanford Digital
  Repository (2013). [purl.stanford.edu/yx282xq2090](https://purl.stanford.edu/yx282xq2090). The repository asks
  that the data never be used to identify the volunteer.

Each dataset's licence and citation are also saved in a `README.txt` next to its files.

## The cache folder

The datasets go into the folder named by the `LARMOR_DATA_DIR` environment variable. Without it, they go into
Larmor's folder in your cache: `~/Library/Caches/larmor` on macOS, `~/.cache/larmor` on Linux (or under
`$XDG_CACHE_HOME`) and `%LOCALAPPDATA%\larmor\Cache` on Windows. `larmor.data.data_folder()` returns this folder,
and each `fetch_` function also accepts another folder as its argument. Deleting the folder removes the data,
which is downloaded again when needed.

If the connection drops, the function retries the download from the beginning, up to three attempts in all. A
file is moved into the cache only when it is complete and its checksum matches, so the cache never holds a
partial or wrong file.

None of the datasets includes a tractogram, but you can make a whole-brain tractogram from any of them with
Larmor's tracking.

## For contributors

An example that uses `larmor.data` carries the line `# needs: example-data`. The tests run it only when
`LARMOR_DATA_DIR` is set, so a plain `pytest` run downloads nothing. CI sets it to a folder it keeps between runs,
keyed on `src/larmor/data/files.py`, the table of URLs, checksums and sizes. A new dataset needs a licence that
allows redistribution and use in documentation (CC0, CC-BY, PDDL or similar), its exact licence and citation, and
a stable host from which a script can download it.
