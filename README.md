# Advanced Data Analytics INS401602

## Overview

Official code repository for Advanced Data Analytics INS401602.

Financial information of 364 Vietnamese's companies are manually collected through annual end-of-year financial report.

The Project objective is to classify whether the company end-of-year financial statement have modified or not based on company's financial metrics.

| **Variable** | **Description** |
|-----------------------|---------------------------------------------------------------------------------|
| **EBIT** | Earnings Before Interest and Taxes |
| **IE** | Interest Expense |
| **EBIT - IE** | EBIT minus Interest Expense |
| **WC/TA** | Working Capital to Total Assets ratio |
| **TL/TA** | Total Liabilities to Total Assets ratio |
| **EBIT/IE** | Interest Coverage Ratio |
| **NOCF/CL** | Net Operating Cash Flow to Current Liabilities ratio (Operating Cash Flow Ratio)|
| **RE/TA** | Retained Earnings to Total Assets ratio |
| **NP/S** | Net Profit Margin |
| **EBIT/TA** | Return on Assets (before interest and taxes) |
| **S/TA** | Asset Turnover Ratio |
| **TA(t)/TA(t-1)** | Total Asset Growth Rate |
| **NP(t)/NP(t-1)** | Net Profit Growth Rate |
| **MVE/TL** | Market Value of Equity to Total Liabilities ratio |

## Research method
in progress
In this project we compare the performance of traditional ML algorithm with Deep Learning model which specialize for tabular data.
We devided into 3 types methods: Traditional model, Ensemble model, Deep Learning model.
| **Traditional**           | **Ensemble**      | **Deep Learning**  |
|---------------------------|-------------------|--------------------|
| Logistic Regression       | Bagging           | NODE               |
| Decision Tree             | Boosting          | TabNet             |
| Random Forest             | Ensemble Voting   | AutoInt            |
| Support Vector Machine    |                   | TabTransformer     |
| Naive Bayes               |                   | GATE               |
| Artificial Neural Network |                   | GANDAF             |
|                           |                   | DANETs             |

In addition for each Deep Learning model, we experiment with 6 type of Loss Function:
* DiceLoss
* BCEDiceLoss
* IoULoss
* FocalLoss
* TverskyLoss
* FocalTverskyLoss

We then compare the performance of these model using: Accuracy, Precision, Recall, F1, AUC.

## Result evaluation
![alt text](images/ml-result.png)

![alt text](images/dl-result.png)

## Steps to run code

### Build from the source
Clone the repo
```sh
git clone https://github.com/PhungQuan-business/Advanced-Data-Analytics.git
```

Create the environment using Anaconda
```sh
conda env create -f environment.yml
```

Activate the conda environment
```sh
conda activate financial-restatement
```

Move to the source code
```
cd /src
```

Start Streamlit app
```sh
streamlit run app.py
```

### Build using Docker
pull the image
```sh
docker pull 
```

## Contributor
Vũ Thu Huyền - 21070237 \
Huỳnh Minh Quân - 21070801 \
Hoàng Ngọc Khoa - 21070330 \
Phùng Hồng Quân - 2107053
