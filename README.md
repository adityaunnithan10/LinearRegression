# Linear Regression from Scratch

Simple linear regression implemented from scratch in Python using gradient descent, with no machine learning libraries. The model predicts a student's `score` from their `studytime`, and its result is checked against the exact least-squares solution.

## Why I built this

I wanted to understand what happens inside `LinearRegression().fit()`: how a loss function is defined, how its gradient points toward a minimum, and how the learning rate and feature scale affect convergence.

## How it works

The model is a line `y = m·x + b`, trained by minimizing the mean squared error:

$$L(m, b) = \frac{1}{n}\sum_{i=1}^{n}\big(y_i - (m x_i + b)\big)^2$$

The gradients with respect to each parameter are:

$$\frac{\partial L}{\partial m} = -\frac{2}{n}\sum_i x_i\big(y_i - (m x_i + b)\big), \qquad \frac{\partial L}{\partial b} = -\frac{2}{n}\sum_i \big(y_i - (m x_i + b)\big)$$

On every epoch, both parameters take a step opposite to their gradient:

```
m = m - L * dL/dm
b = b - L * dL/db
```

where `L` is the learning rate.

## Results

Trained on 100 data points with a learning rate of `0.00035` for 200,000 epochs:

| | m | b |
|---|---|---|
| Gradient descent (this project) | 1.3257 | 6.0599 |
| Exact solution (`np.polyfit`) | 1.3257 | 6.0599 |

The two agree to four decimal places, which confirms the gradient formulas and update step are correct.

## What I learned

- **The learning rate has a hard limit.** On the raw data, any rate above roughly `0.00038` makes the loss grow instead of shrink, and the parameters overflow to `nan`. I hit this at `0.001`.
- **Unscaled features converge slowly.** `studytime` ranges from about 25 to 75, so the gradient for `m` is much larger than the gradient for `b`. The slope settles quickly while the intercept crawls, which is why 200,000 epochs were needed.
- **Standardizing the input fixes this.** Scaling `x` to zero mean and unit variance allows a learning rate around `0.1` and reaches the same answer in a few hundred epochs.

## Project structure

```
LinearRegression/
├── main.py            # gradient descent, training function, and plot
├── analysis.ipynb     # notebook that imports main.py and shows plots
├── data.csv           # dataset (studytime, score)
├── requirement.txt    # dependencies
├── .gitignore
└── README.md
```

## Getting started

```bash
git clone https://github.com/adityaunnithan10/LinearRegression.git
cd LinearRegression

python -m venv myenv
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # Mac/Linux

pip install -r requirement.txt
python main.py
```

The script prints the parameters as training progresses, then shows a scatter plot of the data with the fitted line.

The training loop is not vectorized, so the default 200,000 epochs can take several minutes. For a quick run, lower `epochs` in `main.py`.

## Using the notebook

Open `analysis.ipynb` in VS Code or Jupyter and select the `myenv` kernel. The notebook imports the model from `main.py`, so you can choose the training settings yourself:

```python
import pandas as pd
import main

data = pd.read_csv("data.csv")
m, b = main.train(data, epochs=5000, L=0.00035)
```

## Data

`data.csv` is synthetic: 100 points generated from a line with slope about 1.3 and intercept about 7, plus random noise. It has two columns, `studytime` and `score`.

## Possible improvements

- Vectorize the training loop with NumPy
- Add feature scaling inside the model
- Plot loss against epoch to show convergence
- Evaluate with R² and RMSE on a held-out test set
- Extend to multiple features and a real dataset

## Tech

Python, Pandas, Matplotlib
