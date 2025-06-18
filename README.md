# Insurance Risk Analytics Project
## Task 1 and Task 2
## Project Overview
This project analyzes car insurance data for AlphaCare Insurance Solutions to:
1. Identify low-risk customer segments
2. Optimize premium pricing
3. Improve marketing strategy

## Data
Historical insurance claim data from Feb 2014 to Aug 2015

## Project Structure
- `data/`: Contains all data files
- `scripts/`: Python scripts for data processing and analysis
- `notebooks/`: Jupyter notebooks for exploratory analysis
- `reports/`: Contains analysis reports and visualizations

## Setup
1. Clone repository
2. Install requirements: `pip install -r requirements.txt`

## Final work(Task 3 and Task 4)
## Overview
Predictive modeling and risk analysis for auto insurance premiums using:
- **Data Version Control (DVC)**
- **Machine Learning (XGBoost)**
- **SHAP explainability**
- **Flask API**
  ## Setup
```bash
git clone https://github.com/your-username/Insurance-risk-analytics.git
cd Insurance-risk-analytics
pip install -r requirements.txt
```

## Repository Structure
```
.
├── data/               # Raw and processed data (DVC-tracked)
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for EDA
├── scripts/            # Processing/training scripts
├── api/                # Flask prediction API
├── reports/            # Visualizations and metrics
├── dvc.yaml            # Pipeline definition
└── requirements.txt    # Python dependencies
```

## Key Findings
1. **Top Risk Factors**:
   - Smoker status (4.7× higher costs)
   - BMI (especially for smokers)
   - Age (non-linear relationship)

2. **Model Performance**:
   - R² Score: 0.87
   - MAE: $2,449

3. **Regional Variations**:
   - Southeast has 18% higher claims

## How to Use
### Run the API
```bash
cd api
flask run
```

### Make a Prediction
```bash
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"age":35, "bmi":28, "smoker":"no", "children":2}'
```

## Results
![Regional Analysis](reports/regional_costs.png)
*Figure: Insurance costs by region*

## License
MIT
## DVC Tracking 
bash
# Track large files
dvc add data/insurance.csv models/insurance_model.pkl

# Push data to storage
dvc push

# Commit DVC metadata
git add data/insurance.csv.dvc models/insurance_model.pkl.dvc
git commit -m "Track data and models with DVC"
git push
