# 02-{5,7}18 Final Project

## Setup

Pull the code properly

```bash
git clone --recursive https://github.com/simonlevine/02-712-project.git
```

Install dependencies

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/pip install dvc
```

Grab the data

```bash
dvc pull
```