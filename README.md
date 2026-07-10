# Example

Research scratchpad of Python scripts for training and evaluating 3D deep-learning
segmentation models on CT images of the liver and liver ablation regions. Despite the
repo name, this is not a template — it is a working collection of experiment drivers
from a 2020-era liver disease ablation project. One-off research code, not maintained
(last commit 2021).

## What it does

- Runs learning-rate range tests and hyperparameter sweeps (layers, filters, max
  filters) for 3D UNet-style and hybrid DenseNet121 models, logging trials to
  TensorBoard HParams and Excel.
- Trains and evaluates the selected models (TF1 and TF2 variants exist side by side,
  e.g. `Main.py` vs `Main_TF2.py`, `Run_Model.py` vs `Run_Model_TF2.py`).
- Converts RayStation exports (`Examination.mhd`, `Liver.mhd`, `Ablation.mhd`) to
  NIfTI-style images with SimpleITK for direct model testing
  (`Direct_Testing_From_Raystation/`).

## Layout

- `Main*.py` — flag-driven experiment drivers; toggle booleans inside the script to
  pick the stage (find LR, plot LR, train ~200 epochs, tabulate results).
- `Optimization/` — LR finding and optimization-result plotting.
- `Return_Train_Validation_Generators*.py` — data generators, model builders, and
  hyperparameter definitions.
- `Utils/` — pretrained DenseNet121 loading, metrics-to-Excel, path helpers.

## Requirements

TensorFlow 1.x/2.x, SimpleITK, pandas, openpyxl, TensorBoard. Imports a sibling
`Base_Deeplearning_Code` package (separate repo) and uses hard-coded local drive
paths, so the scripts will not run as-is outside the original environment.
