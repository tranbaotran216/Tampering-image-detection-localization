# Seminar CS406 UIT

This repository is created for the **CS406 Seminar** at the  
**University of Information Technology (UIT)**.

The project focuses on experimenting with different **CNN / Transformer backbones**
on multiple **image forensics datasets**.

## Create Data

1. Create a folder named `data` in the root of this repository:

2. Download the **CASIA dataset** from Kaggle:
   [https://www.kaggle.com/datasets/sophatvathana/casia-dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)

3. Extract and place the dataset into the `data/` folder with the following structure:

   ```text
   data/
   ├── CASIA1/
   └── CASIA2/
   ```

⚠️ The dataset is **not included** in this repository due to size and license constraints.

---
## How to train
```bash
python train_swint2.py 
```
## How to Run `test.py`

```bash
python test.py --backbone swin --dataset columbia
```

### Available Options

* **Datasets**:

  ```text
  [columbia, coverage, casia]
  ```

* **Backbones**:

  ```text
  [swin, resnet, mobilenet]
  ```
