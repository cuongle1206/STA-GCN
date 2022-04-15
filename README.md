# STA-GCN
This repository is the implementation for "Spatio-temporal attention graph neural network for skeleton-based action recognition".

## Architecture of CTR-GC
![image](architecture.png){:height="36px" width="36px"}

# Prerequisites
- Python = 3.6.8
- PyTorch >= 1.1.0
- `pip install -r requirements.txt `
- `pip install -e torchlight`

# Datasets
Download skeleton-based dataset: https://github.com/shahroudy/NTURGB-D
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)

Put downloaded data into the following directory structure:

```
- data/
  - ntu60/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
  - ntu120/
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
```

