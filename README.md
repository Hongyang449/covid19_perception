## Text-based prediction of COVID-19

NLP models are used in predictiong COVID-19 based on texts about pereceptual ability changes in smell, taste, and chemesthesis.

![Figure1](figure/fig1.jpg?raw=true "Title")

---

## Installation
Git clone a copy of code:
```
git clone https://github.com/Hongyang449/covid19_perception.git
```
## Required dependencies

* [python](https://www.python.org) (3.9.7)
* [transformers](https://huggingface.co/docs/transformers/main/en/index) (4.21.0)
* [shap](https://github.com/slundberg/shap) (0.39.0)

## 1. Preprocess data 
* directory data/
* python preprocess.py

## 2. Train NLP models and make predictions 
* directories code/class5/ and code/class6/
* python train.py

## 3. SHAP analysis 
* directories code/class5/ and code/class6/
* python shap_analysis.py
* python shap_consensus.py


