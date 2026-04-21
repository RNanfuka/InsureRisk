# Contributing to InsureRisk

Thanks for contributing to InsureRisk.

This project focuses on insurance risk prediction, model evaluation, and a Streamlit dashboard for decision support. The goal of this guide is to keep contributions clear, reproducible, and easy to review.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/RNanfuka/InsureRisk.git
cd InsureRisk
```

### 2. Create the environment

Using conda:

```bash
conda env create -f environment.yml
conda activate insurerisk
```

### 3. Run the project

Train the model:

```bash
python src/train_model.py
```

Evaluate the model:

```bash
python -m src.evaluate_model
```

Launch the app:

```bash
streamlit run app/app.py
```

## Contribution Areas

Contributions are welcome in areas such as:
- data preprocessing improvements
- model training and evaluation improvements
- explainability enhancements
- Streamlit UI improvements
- documentation cleanup
- test coverage

## Project Conventions

Please follow these guidelines when contributing:

- keep changes focused and small where possible
- prefer readable code over clever shortcuts
- update documentation when behavior changes
- keep file structure consistent with the current project layout
- reuse the existing training and preprocessing logic unless intentionally refactoring it

## Code Style

- use clear function names
- add docstrings for non-trivial functions
- keep comments concise and meaningful
- avoid introducing unnecessary dependencies

## Working With Models and Reports

This repository may generate files such as:
- `models/model.pkl`
- `reports/evaluation_metrics.json`
- `reports/confusion_matrix.png`
- `reports/feature_importance.png`

If your change affects outputs, make sure:
- the generated files match the current code
- the README stays consistent with the workflow
- large or unnecessary temporary files are not committed

## Before Submitting Changes

Please check the following:

1. The code runs without errors in the project environment.
2. Training, evaluation, or app commands still work if your change touches them.
3. Documentation is updated if commands, outputs, or features changed.
4. Generated outputs are only committed when they are intentionally part of the project.

## Pull Request Guidance

When opening a pull request, include:
- a short summary of what changed
- why the change was made
- any files or outputs reviewers should pay attention to
- screenshots if the Streamlit dashboard UI changed

## Questions

If something is unclear, open an issue or document the assumption in your pull request description.
