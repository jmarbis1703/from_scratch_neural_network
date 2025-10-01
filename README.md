# From-Scratch Neural Network

**From-scratch NumPy implementation of a feedforward neural network, applied to Titanic survival prediction.**  
This project demonstrates backpropagation coded by hand (no ML frameworks), mini-batch gradient descent with Adam, early stopping, and reproducibility checks.  

Highlights:
- Emphasizes the math and statistics behind neural nets. 
- From-scratch NN (NumPy) with tested backprop (gradient check).
- No data leakage: split, fit scaler on train, transform val&test.
- Mini-batch, shuffling, early stopping, Adam optimizer.
- Fast demo notebook runs in < 2 minutes on CPU (subset).

## Quickstart (fast)
```bash
python -m final_project.cli fast-titanic
```
If `Titanic Dataset.csv` is not present, the script generates a small synthetic dataset so the pipeline still runs.






