# Starbucks Capstone Project

This is my Capstone project for Udacity Machine Learning Engineer nanodegree. All source files and final report will be available in this repo soon.

## Installation

To recreate this project you will need:

- Data from Udacity and Starbucks
- Python 3.7.5
- To install requirements:

```bash
pip install pip_requirements.txt
```


## Project Overview

Udacity partnered with Starbucks to provide a real-world business problem and simulated data mimicking their customer behavior.  This project is focused on tailoring the personalized offers sent as part of the Starbucks Rewards Program to the customers who are most likely to use them. The Machine Learning terminology for this is "propensity modeling".

We want to determine which kind of offer, if any, to send to each customer based on their purchases and interaction with the previously sent offers. Some customers do not want to receive offers and might be turned off by them, so we want to avoid sending offers to those customers.

## Data overview

All ownership of used data is belong to Udacity and Starbucks. It consist of the following path:

- **profile.json**: Rewards program users (17000 users x 5 fields)
  * gender: (categorical) M, F, O, or null
  * age: (numeric) missing value encoded as 118
  * id: (string/hash)
  * became_member_on: (date) format YYYYMMDD
  * income: (numeric)

- **portfolio.json**: Offers sent during 30-day test period (10 offers x 6 fields)
  * reward: (numeric) money awarded for the amount spent
  * channels: (list) web, email, mobile, social
  * difficulty: (numeric) money required to be spent to receive reward
  * duration: (numeric) time for offer to be open, in days
  * offer_type: (string) bogo, discount, informational
  * id: (string/hash)

- **transcript.json**: Event log (306648 events x 4 fields)
  * person: (string/hash)
  * event: (string) offer received, offer viewed, transaction, offer completed
  * value: (dictionary) different values depending on event type
  * offer id: (string/hash) not associated with any "transaction"
  * amount: (numeric) money spent in "transaction"
  * reward: (numeric) money gained from "offer completed"
  * time: (numeric) hours after start of test

## Files Description

- **requirements.txt** - Python packages required to run this program.
- **proposal.pdf** - Proposal for this project.
- **README.md** - README file for this project.
- **report.pdf** - Report for this project.
- **notebook.ipynb** - Jupyter Notebook containing all project steps.
- **src/model.py** - Neural network model code (required for hyperparameter tuning).

## Results Summary

The Neural Network model performed the best with an F₂ Score of 0.84863 on the Test Set.

|                  Model                   | Accuracy | F1 Score | F2 Score |  TP   |  FP   |  TN   |  FN   |
| :--------------------------------------: | :------: | :------: | :------: | :---: | :---: | :---: | :---: |
|   Logistic Regression __\[test set\]__   | 0.71208  | 0.79208  | 0.83016  | 4838  | 1737  | 1444  |  803  |
| Support Vector Machines __\[test set\]__ | 0.72463  | 0.78873  | 0.80353  | 4534  | 1391  | 1858  | 1038  |
| Neural Network (Final) __\[test set\]__  | 0.71163  | 0.79726  | 0.84863  | 5002  | 1905  | 1276  |  639  |
