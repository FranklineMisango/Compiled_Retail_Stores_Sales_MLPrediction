# Rossman store sales prediction

## Business Problem
Rossmann operates over 3,000 drug stores in 7 European countries. Currently, Rossmann
store managers are tasked with predicting their daily sales for up to six weeks in advance.
Store sales are influenced by many factors, including promotions, competition, school and
state holidays, seasonality, and locality. With thousands of individual managers predicting
sales based on their unique circumstances, the accuracy of results can be quite varied.


## Data Description
We have been provided data with the following data fields.

* Id - an Id that represents a (Store, Date) duple within the test set
* Store - a unique Id for each store
* Sales - the turnover for any given day (this is what you are predicting)
* Customers - the number of customers on a given day
* Open - an indicator for whether the store was open: 0 = closed, 1 = open
* StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are
closed on state holidays. Note that all schools are closed on public holidays and
weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
* SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public
schools
* StoreType - differentiates between 4 different store models: a, b, c, d
* Assortment - describes an assortment level: a = basic, b = extra, c = extended
* CompetitionDistance - distance in meters to the nearest competitor store
* CompetitionOpenSince[Month/Year] - gives the approximate year and month of the
time the nearest competitor was opened
* Promo - indicates whether a store is running a promo on that day
* Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 =
store is not participating, 1 = store is participating
* Promo2Since[Year/Week] - describes the year and calendar week when the store
started participating in Promo2
* PromoInterval - describes the consecutive intervals Promo2 is started, naming the
months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round
starts in February, May, August, November of any given year for that store
Src : https://www.kaggle.com/c/rossmann-store-sales/dataAim
We are provided with historical sales data for 1,115 Rossmann stores. The task is to
forecast the "Sales" column.

### Approach

Exploratory data analysis is the process of analysing the dataset to understand its
characteristics. In this step, we will figure out the following.
a. Identifying the number of unique users
b. Platforms used by the users the most
c. Correlations
d. Checking for null / inconsistent values and various other insights are drawn.
2. Imputation
This step involves the process of filling the missing values in appropriate ways so
that the data is not lost.
3. Outliers Detection and removal
Outliers are nothing but points that are abnormally distant from the other points.
These kinds of outliers present in the data are detected and eliminated.
4. Further Exploratory Data Analysis
Further EDA is performed to understand the data more and find out a few exceptional
cases.
5. Label Encoding and One hot encoding
Machine learning algorithms for regression can understand the input only in the form
of numbers and hence it is highly essential to convert the non - numeric data that we
have to numeric data by providing them labels.
6. Model Building and evaluation
Various regression algorithms are applied on the dataset and the model that suits
best for the dataset is selected. The models that we apply for this dataset are
a. Linear Regression
b. SGD Regression
c. Random Forest Regressor
d. Decision Tree Regressor
7. Feature Importance Analysis
