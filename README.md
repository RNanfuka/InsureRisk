# InsureRisk

InsureRisk is a machine learning project that predicts whether an insurance customer is likely to fall into a higher-cost or lower-cost segment based on demographic and health-related inputs.

The project includes:
- data preparation and feature encoding
- model training with logistic regression and random forest
- model evaluation with classification metrics and plots
- a Streamlit dashboard for prediction, explanation, and decision support

## Project Goal

The goal is to help translate insurance profile data into a practical risk signal that can be used in a simple decision-support workflow.

In this project, the target is a binary label:
- `0` = lower-cost customer
- `1` = high-cost customer

The label is created by splitting insurance charges around the dataset median.

## Tech Stack

- Python
- pandas
- scikit-learn
- matplotlib
- Altair
- Streamlit
- joblib

## Project Structure

```text
InsureRisk/
├── app/
│   └── app.py
├── data/
│   ├── processed/
│   │   └── clean_data.csv
│   └── raw/
│       └── insurance.csv
├── models/
│   └── model.pkl
├── notebooks/
├── reports/
│   ├── confusion_matrix.png
│   ├── evaluation_metrics.json
│   └── feature_importance.png
├── src/
│   ├── data_preprocessing.py
│   ├── evaluate_model.py
│   └── train_model.py
├── environment.yml
└── requirements.txt
```

## Data and Features

The model uses the following fields:
- `age`
- `sex`
- `bmi`
- `children`
- `smoker`
- `region`

During preprocessing:
- `sex` is encoded as binary
- `smoker` is encoded as binary
- `region` is one-hot encoded
- `high_cost` is created from the median of `charges`

## Model Training

Training is handled in [src/train_model.py](/Users/rebeccarosettenanfuka/Desktop/MY-_PROJECTS/InsureRisk/src/train_model.py:1).

The script:
1. loads the processed dataset
2. preprocesses the features
3. splits the data into train and test sets
4. trains logistic regression and random forest models
5. prints classification reports
6. saves the random forest model to `models/model.pkl`

Run training with:

```bash
python src/train_model.py
```

## Model Evaluation

Evaluation is handled in [src/evaluate_model.py](/Users/rebeccarosettenanfuka/Desktop/MY-_PROJECTS/InsureRisk/src/evaluate_model.py:1).

The evaluation script:
- reloads the saved model
- reconstructs the same preprocessing and test split
- computes accuracy, precision, and recall
- saves metrics to `reports/evaluation_metrics.json`
- saves a confusion matrix plot
- saves a feature importance plot for the trained random forest

Run evaluation with:

```bash
python -m src.evaluate_model
```

### Current Evaluation Results

Based on the current saved model:
- Accuracy: `0.948`
- Precision: `0.984`
- Recall: `0.910`

## Streamlit App

The interactive dashboard is in [app/app.py](/Users/rebeccarosettenanfuka/Desktop/MY-_PROJECTS/InsureRisk/app/app.py:1).

The app provides:
- sidebar inputs for customer profile data
- prediction summary cards
- confidence and risk-level messaging
- plain-English explanations of what influenced the prediction
- a local driver view for the current profile
- a global feature-influence chart
- suggested action guidance

Run the app with:

```bash
streamlit run app/app.py
```

## Setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate insurerisk
```

### Option 2: pip

If you prefer pip, install the required packages manually or populate `requirements.txt` with the same dependencies listed in `environment.yml`.

## Example Workflow

```bash
conda activate insurerisk
python src/train_model.py
python -m src.evaluate_model
streamlit run app/app.py
```

## Outputs

After running the project, you should see:
- `models/model.pkl`
- `reports/evaluation_metrics.json`
- `reports/confusion_matrix.png`
- `reports/feature_importance.png`

## Notes

- The current model used by the dashboard is the saved random forest classifier.
- The Streamlit app still includes `sex` and `region` because the trained model depends on them.
- `src/data_preprocessing.py` is currently a placeholder and the active preprocessing logic lives in `src/train_model.py`.

## Future Improvements

- move preprocessing into `src/data_preprocessing.py`
- add model persistence for preprocessing steps
- add tests for training and evaluation
- add deployment instructions
- improve explainability with SHAP or a more formal local explanation method

## Author

Rebecca Rosette Nanfuka  
Master of Data Science - University of British Columbia

- GitHub: https://github.com/RNanfuka
- LinkedIn: https://linkedin.com/in/rebecca-rosette-nanfuka-657b3920b
