# Brent Oil Price Analysis & Risk Monitoring System

A Python-based analytical system for detecting structural changes in Brent oil prices, computing financial risk metrics, and generating interactive dashboards for decision support.

## Business Problem

Brent oil price volatility significantly impacts energy markets, investment portfolios, and policy decisions.

Organizations often lack:

- Automated risk metric calculation
- Reliable detection of structural regime shifts
- Integrated dashboards for quick interpretation
- Reproducible and tested analytical pipelines

Manual analysis is slow, inconsistent, and difficult to scale.

## Solution Overview

This project implements a modular analytical pipeline that:

- Loads and validates historical Brent oil price data
- Detects structural change points in time series
- Computes key financial risk metrics
- Generates clear visualizations and dashboards
- Ensures reliability through testing and CI/CD automation

The system is built with clean architecture principles and separation of concerns across modules.

## Key Results

- **Automated Risk Analysis**: Financial metrics computed instantly instead of manual spreadsheet work
- **Regime Shift Detection**: Identifies structural changes in oil price behavior
- **Improved Data Reliability**: Built-in validation and preprocessing
- **Reduced Manual Effort**: End-to-end pipeline execution via a single entry point
- **Quality Assurance**: Automated testing, linting, and security scanning


## Quick Start
```
git clone https://github.com/Eyu12/brent-oil-analysis
cd brent-oil-analysis
pip install -r requirements.txt
python src/main.py

```
## Project Structure
```
brent-oil-analysis/
│
├── src/
│   ├── config.py           # Configuration parameters and paths
│   ├── data_loader.py      # Data loading and preprocessing logic
│   ├── change_point.py     # Change point detection algorithms
│   ├── risk_metrics.py     # Financial and risk metric calculations
│   ├── visualization.py    # Plotting and chart generation
│   ├── dashboard.py        # Dashboard assembly and reporting
│   ├── utils.py            # Utilities (logging, validation, helpers)
│   └── main.py             # Application entry point
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_change_point.py
│   ├── test_metrics.py
│
├── data/                   # Raw or sample datasets
│
├── requirements.txt        # Python dependencies
├── README.md
└── .github/workflows/ci.yml # CI/CD pipeline
```
## Technical Details

### Data

Historical Brent oil price data (CSV format)

Preprocessing includes:
- Missing value handling
- Outlier detection
- Index validation
- Date formatting

### Model & Algorithms

**Change Point Detection**
- Binary Segmentation
- PELT (if using ruptures)
- Sliding window methods
- Consensus detection (if implemented)

**Risk Metrics**
- Daily returns
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Rolling beta

All computations use vectorized NumPy and pandas operations for efficiency.

### Evaluation & Validation

- Unit tests using pytest
- Coverage reporting via pytest-cov
- Linting with flake8
- Formatting validation with black
- Type checking with mypy
- Security scanning using safety and bandit
- Automated CI/CD pipeline via GitHub Actions

## Future Improvements

With additional development time, the system could be extended to include:

- Machine learning-based price forecasting
- Real-time API integration
- Multi-commodity analysis (WTI, natural gas)
- Automated PDF/Excel report export
- Docker containerization for deployment
- Cloud deployment (AWS/GCP/Azure)

## Author

**Eyayaw Zewdu Ejigu**  
Communication Engineer

- LinkedIn: [https://www.linkedin.com/in/eya-z-87a0b613b/]
- Email: eyazew1@gmail.com