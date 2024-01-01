# `MHSMS_analysis`

The MHSMS (Mental Health Services Monthly Statistics) are reported as spreadsheets by the NHS (the UK's National Health Service). This repository contains Python codes to parse and analyse these datasets.

## Installation

Requires the Python packages `matplotlib`, `numpy`, `pandas`, and `xarray`, and optionally `scipy` (for statistics) and `mplcursors` (for interactive plot labels). These packages can be installed using a package manager such as Anaconda.

## Usage

There are two main scripts, which should be called from the command line:

```
python3 process_data.py
python3 plot_results.py
```
