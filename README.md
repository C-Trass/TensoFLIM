# TensoFLIM

TensoFLIM is a Python toolkit for processing, normalising, and analysing
fluorescence lifetime imaging microscopy (FLIM) data from molecular
tension sensors.

The package is designed for lifetime-based force readouts, where changes
in fluorescence lifetime or FRET efficiency are interpreted as relative
mechanical tension rather than absolute force.

TensoFLIM focuses on reproducible, quantitative workflows for analysing
tension-sensitive FLIM data across cells, conditions, and experiments.

---

## Features

- Processing of lifetime images from FLIM-based tension sensors
- Offset correction and zero-force referencing using tensionless controls
- Conversion between lifetime, FRET efficiency, and relative force
- Pixel-wise and region-based analysis
- Generation of publication-ready plots and summary statistics
- Designed to integrate with FLIM, FRET, and TCSPC pipelines

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/C-Trass/TensoFLIM.git
cd TensoFLIM
pip install -r requirements.txt
For development use:
```

---

## Who is this for?
TensoFLIM is intended for:

- Researchers using FLIM-based molecular tension sensors
- Users analysing vinculin, talin, or other mechanosensitive probes
- Scientists interested in relative force changes rather than absolute
calibration
-FLIM and FRET users who require transparent, scriptable analysis

---

## Citation
If you use TensoFLIM in academic work, please cite:

Dr Conor A. Treacy, TensoFLIM: Quantitative analysis of FLIM-based molecular
tension sensor data,
Biomedical Optics Express, forthcoming.

A DOI will be added here once the associated methods paper is published.

Until then, you may cite this repository directly using the GitHub URL
and the software version.

---

## License
This project is released under the MIT License.
You are free to use, modify, and distribute the code with attribution.

---

## Contact
Maintained by Conor Treacy
📫 Contact: [LinkedIn](https://www.linkedin.com/in/conor-treacy-phd-2aa7406a/) | [Email](mailto:conor.treacy@brunel.ac.uk)
