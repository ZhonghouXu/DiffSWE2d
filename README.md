# DiffSWE2d

A PyTorch-based, differentiable two-dimensional shallow-water
equations solver for end-to-end tsunami and flood modelling. The solver uses a
structured square-cell model grid and Basilisk B-Flood/BG-Flood style finite-volume method.

## How to install
```bash
git clone https://github.com/ZhonghouXu/DiffSWE2d.git
cd DiffSWE2d
conda create -n diffswe2d python=3.11
conda activate diffswe2d
pip install -r requirements.txt
```

## How to run an example scenario
Forward modelling (inference mode)
```bash
python ./Examples/forward/Monai_Valley/Monai_Valley_res001/example_run.py
```
Inverse modelling
```bash
python ./Examples/inverse/Monai_Valley/Monai_Velley_BC_inversion_spline_lr002_6simg/bc_inversion_train.py
```

## Citation
If you find this useful, consider citing: Xu, Z (2026). DiffSWE2d: a differentiable Shallow Water Equations solver for
end-to-end flood and tsunami modelling. 










