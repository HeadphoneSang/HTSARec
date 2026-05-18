# HTSARec
This is the main implementation code of HTSARec for sequential recommender systems.

## 🛠️Installation
First you need to install `RecBole` framework to conduct the experiments. Then, move `htsarec.py` to the `sequential_recommender` directory and modify the `__init__.py` to make sure the `HTSARec` model can be integrated into `RecBole` framework.

In addition, the `.recbole/` directory provided in this repository contains modified source files for the original `RecBole` framework to ensure compatibility with `HTSARec`. Before running the experiments, please replace the corresponding files in the original `RecBole` source code with the files provided in this directory.

Install `RecBole` and other required packages:
```bash
pip install recbole, ray, ray[tune], protobuf==3.20.0
```
## 🚀Configuration & Training
First, replace the original `overall.yaml` file in the RecBole framework with the `overall.yaml` file provided in the root directory of this repository.

Next, copy `HTSARec.yaml` into the `.recbole/properties/model/` directory of the RecBole framework.

In addition, all datasets and corresponding experimental configuration files used in this paper are provided in the `.datasets/` directory of this repository. Please select and import the appropriate dataset files according to your experimental requirements.

Train HTSARec:
```bash
python run.py
```
Some basic configurations are set in `run.py`. 



