# 02-{5,7}12 Final Project: Simulating Latent Single-Cell Gene Expression of COVID-19 Patients with Variational Autoencoder and Implications for Augmenting Classification Models

## Abstract

> We present a Variational Autoencoder-based simulation of single cell gene expression from healthy and COVID-19 patients' PBMC cells. To demonstrate the utility of this simulation, we build a synthetic dataset seeded from rare cell types. We then train a classifier of ventilation severity and demonstrate that, by augmenting with simulated data, predictive performance declines slightly. However, based on phylogeny tree analysis, the simulated rare cell data does not differ dramatically from the original distribution, as desired. Regardless of these mixed results, we consider our simulation as a solid baseline and a promising future direction to aiding in single cell research of COVID-19.

## Setup

Pull the code properly

```bash
git clone --recursive https://github.com/simonlevine/singlecell-sim-by-VAE.git
```

Install dependencies

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install dvc
```

Grab the data

```bash
source venv/bin/activate && source .envrc && make download_data
```

Reproduce
```bash
source venv/bin/activate && source .envrc && dvc repro
```
