# LoopICL

LoopICL is a looped in-context learning foundation model for tabular data, submitted for reproducibility at ICLR.

## Installation

```bash
pip install -e .
```

For OpenML benchmark evaluation:
```bash
pip install -e ".[openml]"
```

## Usage

### Inference

```python
from loopicl.interface import LoopICLClassifier

clf = LoopICLClassifier()
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)
```

### Training

```bash
python -m loopicl.train.run <config_name>
```

### Generating prior data

```bash
genload --config <prior_config> --output <output_dir>
```

## Reproducibility

All configs used in the paper are in `src/loopicl/configs/`.
Benchmark evaluation scripts are in `paper_experiments`.
