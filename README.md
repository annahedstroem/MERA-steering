<br/><br/>
<p align="center">
  <img width="300" src="logo.png">
<h3 align="center"><b>Inference-time Steering for Language Models</b></h3>
<p align="center">PyTorch</p>
<br/><br/>

This repository contains the code and experiments for the paper **["To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models"]([Link](https://openreview.net/pdf?id=ukLxqA8zXj))** by Hedström et al., (2025).

[![Getting started!](https://colab.research.google.com/assets/colab-badge.svg)](anonymous)
![Python version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!--[![PyPI version](https://badge.fury.io/py/metaquantus.svg)](https://badge.fury.io/py/metaquantus)-->
<!--[![Python package](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)](https://github.com/annahedstroem/MetaQuantus/actions/workflows/python-publish.yml/badge.svg)-->
<!--[![Launch Tutorials](https://mybinder.org/badge_logo.svg)](anonymous)-->

Please note that this repository is under active development!
<!--
## Citation

If you find this work interesting or useful in your research, use the following Bibtex annotation to cite us:

```bibtex
@article{mera2025steering,
    title={To Steer or Not to Steer? Mechanistic Error Reduction with Abstention for Language Models},
    author={},
    journal={},
    year={},
    url={},
}
```
-->
<!--This work has been published ...........-->

## Repository overview

The repository is organised as follows:
- The `src/` folder contains all necessary functions.
- The `nbs/` folder includes notebooks for generating the plots in the paper and for benchmarking experiments.
- The `tests/` folder contains the tests.
<!--- The `assets/` folder contains all files to reproduce the experiments.-->

## Paper highlights 📚

Our approach consists of three main steps. First, we use linear probes to obtain an effective direction for minimising the predicted error. Second, this direction is then scaled using the closed-form solution at both the token and layer levels. Third, we calibrate the steering threshold against the probe's error on a calibration dataset, informed by the user's tolerance for uncertainty.
</p>
<p align="center">
  <img width="750" src="summary.png"> 
</p>

The main benefits of MERA are:
- **Selective steering** — steer only if the probe's estimated error is larger than a calibrated threshold α
- **Adaptive strength** — the steering intensity λ scales with the probe's estimated error 
- **Global abstention** — steer only if when confident, such that performance is at least ε with probability 1-δ

## Installation

Install the necessary packages using the provided [requirements.txt](https://github.com/annahedstroem/MERA-steering/blob/main/requirements.txt):

```bash
conda create --name mera python==3.10
conda activate mera 
pip install torch --extra-index-url https://download.pytorch.org/whl/cu122  
pip install -r requirements.txt
```

If you want to run SAE experiments, also install:
```bash
pip install -e git+https://github.com/jbloomAus/SAELens.git#egg=sae-lens
pip install --force-reinstall --no-cache-dir cffi
```

## Package requirements 

Required packages are:

```setup
python
torch
transformers
datasets
huggingface_hub
accelerate
wandb
```

## Getting started

If you want to try using MERA with your own dataset and model, go to the following notebook `nbs/getting_started.py`.

To steer with MERA on any of the existing datasets and models (see supported datasets and models [here](#supported-models-and-datasets)), run the following script:
```bash
python mera.py --dataset_names yes_no_question --model_name google/gemma-2-2b
```

## How to reproduce experimental results

In the following, we describe how to reproduce the results in the paper. It requires 
- access to `wandb` (and that you pass your API [key](https://docs.wandb.ai/support/find_api_key/)
- that datasets are downloaded and saved to a `MERA-steering/hf_cache` folder [here](#how-to-download-datasets))
- that you have a `runs/` folder to save results

Create a  `runs/` folder at the root and go to `src/` folder.

```bash
mkdir runs/
cd src
```

<details> <summary><b>Step 1.</b> Prepare datasets for probe training</summary>
For each model, to prepare datasets for probe training (see supported datasets and models [here](#supported-models-and-datasets)) run the following script:

<br>
```bash
python -m cache.cache_run --dataset_names sentiment_analysis yes_no_question mmlu_high_school sms_spam --nr_samples 3000 --model_name meta-llama/Llama-3.2-1B-Instruct --hf_token INSERT_KEY
python -m cache.cache_run --dataset_names mmlu_professional --nr_samples 2601 --model_name meta-llama/Llama-3.2-1B --hf_token INSERT_KEY
```
</br>
Just rerun with the different models (see supported datasets and models [here](#supported-models-and-datasets)).

Next, post-processes the cache data (i.e., subselect activation values based on token positions ("last" of the prompt and "exact" of the answer)), making the cached files significantly smaller in size in preparation for probe training.

<br>
```bash
python -m cache.cache_postprocess --dataset_names sms_spam
```
</br>

</details>

<details> <summary><b>Step 2.</b> Train linear probes</summary>
For each model, to train linear probes (error estimators), run the following script:

<br>
```bash
python -m probes.probes_train --dataset_names sentiment_analysis yes_no_question mmlu_high_school sms_spam --model_name meta-llama/Llama-3.2-1B-Instruct --save_name custom --transform
```
</br>
if you want to change any of the hyperparameters, please edit the script `probes_train.py` directly.

To analyse the performance of the probes, go to the following notebook `nbs/evaluate_probes.py`.
</details>

<details> <summary><b>Step 3.</b> Benchmark steering methods</summary>
For each model, to benchmark steering methods, run the following script:

<br>
```bash
python -m steering.steering_run --steering_methods optimal_probe --dataset_names sms_spam --model_names "meta-llama/Llama-3.2-1B-Instruct" --fname custom_experiment --probe_token_pos exact --wandb_key INSERT_KEY
```
</br>

To analyse the performance of the steering methods, go to the following notebook `nbs/evaluate_steering.py`.
</details>



## Dataset and models

#### How to download datasets

To download the datasets, please follow the instructions [here](src/how-to-download-datasets.md). If you want to include additional datasets, please follow the guide [here](src/how-extend-datasets.md)).

#### Supported datasets and models

Currently, we support these datasets:
- `sentiment_analysis`
- `yes_no_question`
- `mmlu_high_school`
- `sms_spam`

Our experiments work with these models:
- `google/gemma-2-2b`
- `google/gemma-2-2b-it`
- `Qwen/Qwen2.5-3B`
- `Qwen/Qwen2.5-3B-Instruct`
- `meta-llama/Llama-3.2-1B`
- `meta-llama/Llama-3.2-1B-Instruct`
- 
... but other `HuggingFace` decoder-only LM models that contain blocks with a residual stream would be compatible with our current implementation (see `register_hooks`  in [here](https://github.com/annahedstroem/MERA-steering/blob/main/src/cache/cache_utils.py#L217)).

### Thank you

We hope our repository is beneficial to your work and research. If you have any feedback, questions, or ideas, please feel free to raise an issue in this repository. Alternatively, you can reach out to us directly via email for discussions or suggestions. 

📧 Contact us: 
- Anna Hedström: [hedstroem.anna@gmail.com](mailto:hedstroem.anna@gmail.com)
- Salim Amoukou: [salimamoukou@gmail.com](mailto:salimamoukou@gmail.com)

Thank you for your interest and support!