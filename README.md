# Data Job Compensation Analysis 
 - Donghang Wu ©2022

### Project Goals

While switching to a data related career, I was intrigued in gauging into salary data of such a career to gain insights of how different roles are valued in employers' perspectives

The goal of this project is to gain insight into the possble **factors** that affect data related job salaries, and build ***easy to interpret*** **predictive models** *(ie. not having a 99% R2 Score model with 50 features of improbable interpretability)*

The project used many **EDA plots/graphs** to gain insights of each feature's relationship to target (salary). To minimize the amount of feature included in model construction without compromising data and model integrity, I used **empirical evidence (plots and hypothesis testings)** in determining outliers and feature relations

It is important to note that **Linear Regression (OLS)** is the **base model** for ***coefficient interpretation*** and ***R2 Score reference***

In later parts, I utilize **regularized** and **ensemble models** in effort of enhancing model performance without changing or adding features, comparing models' performance and advantages/disadvantages


### Process Notes

As of October, 2022. To use the **TWO** modules in the [**Appendix**](#A-1), make sure to import related **Python files (or copy related code)** I created. ***Note*** that these two modules are **greedy** in finding the best parameters for the two tree ensemble models, you may need to manually fiddle some codes in order to find the optimal parameters.

***I will improve the BestForest/BestGB module (A module that displays parameter vs. R2 Score) upon demand.*** Personally I find tunning for optimal hyperparameter to be better this way; it can give me valuable insights of how R2 Score changes as parameters change. However, it can be very time consuming at times, so please use at own discretion

Here are a few questions that this project has sought to answer:
- What's the ideal size of company and employment type for a data job?
- Does some of the categorical different given in the data actually make a difference in salary?
- What are the most important factor (feature) in determinig data job salary?
- Does the model predicted make sense comparing to data job salaries claimed on the internet?
- What is someone like me's predicted salary? (An entry level Data Scientist living in a developing country)


### Data sources

The csv file is a popular data salary dataset on [**Kaggle**](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries). 

According to author, the original data is aggregated and provided by [ai-jobs.net](https://www.ai-jobs.net)
