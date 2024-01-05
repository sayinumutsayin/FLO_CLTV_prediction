##############################################################
# CLTV Prediction with BG-NBD and Gamma-Gamma:
##############################################################

###############################################################
# About the Dataset:
###############################################################

# The dataset consists of information derived from the past shopping behaviors of
# customers who engaged in OmniChannel shopping, both online and offline, during the years 2020-2021.

# FEATURES:

# master_id: Unique customer id
# order_channel : Shopping platforms: Android, ios, Desktop, Mobile, Offline
# last_order_channel : Last channel of the order
# first_order_date :
# last_order_date :
# last_order_date_online : date of the last online order
# last_order_date_offline : date of the last offline order
# order_num_total_ever_online : The total number of purchases made by the customer on the online platform.
# order_num_total_ever_offline : The total number of purchases made by the customer on the offline platform.
# customer_value_total_ever_offline : The total amount spent by the customer in offline purchases.
# customer_value_total_ever_online : The total amount spent by the customer in online purchases.
# interested_in_categories_12 : The list of categories in which the customer has made purchases in the last 12 months.


###############################################################

# Problem:
# FLO aims to outline a roadmap for its sales and marketing activities.
# In order for the company to develop medium to long-term plans,
# it is essential to forecast the potential value that existing customers
# will bring to the company in the future.



# Libraries needed:

import pandas as pd
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_rows", 20)
pd.set_option("display.float_format", lambda x: "%.4f" % x)




###############################################################
# Data Preparation:

df_ = pd.read_csv("FLOCLTVPrediction/flo_cltv_prediction_dataset.csv")
df = df_.copy()
df.describe([0.05, 0.50, 0.95, 0.99]).T
df.isnull().any()
df.describe().T

# Building represser function to repress the outliers
# Outlying range has been chosen out of the range %0.01-%0.99
def outlier_thresholds(dataframe, column):
    q1 = dataframe[column].quantile(0.01)
    q3 = dataframe[column].quantile(0.99)
    iqr = q3 - q1
    up_limit = round(q3 + 1.5 * iqr)
    low_limit = round(q1 - 1.5 * iqr)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, column):
    low_limit, up_limit = outlier_thresholds(dataframe, column)
    # dataframe.loc[(dataframe[column] < low_limit), column] = low_limit
    dataframe.loc[(dataframe[column] > up_limit), column] = up_limit

# Repressing the outliers:
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# Creating new features for total purchase number(total_purchase) and total expense(total_value)
df["total_purchase"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_value"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head(10)

# Converting date columns to date type

df.info()
dates = ["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline"]
# OR dates = [col for col in df.columns if "date" in col]
for col in dates:
    df[col] = pd.to_datetime(df[col]) # .astype("datetime64[ms]")


# Creating the data structure to predict CLTV
# analize date: 2 days after the last purchase
df["last_order_date"].max() # 30.05.2021
today_date = dt.datetime(2021, 6, 1)

# Creating a new dataframe with columns
# customer_id, recency(weekly recency), T(weekly tenure), frequency and monetary(average monetary)
df.columns
cltv_avg = pd.DataFrame()

cltv_avg["recency"] = (df["last_order_date"] -  df["first_order_date"]).dt.days
cltv_avg["T"] = (today_date -  df["first_order_date"]).dt.days
cltv_avg["frequency"] = df["total_purchase"]
cltv_avg["monetary"] = df["total_value"] / df["total_purchase"]


cltv_avg = (cltv_avg[cltv_avg["frequency"] > 1])
cltv_avg["recency"] = cltv_avg["recency"] / 7
cltv_avg["T"] = cltv_avg["T"] / 7

cltv_avg.describe().T
cltv_avg.info()

# Setting BG-NBD and Gamma-Gamma models, calculating the customer Lifetime Value(CLTV)

# Fitting the BG-NBD model

bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_avg["frequency"],
        cltv_avg["recency"],
        cltv_avg["T"])

# # Making the number of next 3-month-purchase predictions: new column exp_sales_3_month
cltv_avg["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                                                        cltv_avg["frequency"],
                                                                                        cltv_avg["recency"],
                                                                                        cltv_avg["T"])

# OR

cltv_avg["exp_sales_3_month"] = bgf.predict(4*3,
                                            cltv_avg["frequency"],
                                            cltv_avg["recency"],
                                            cltv_avg["T"])

# # Making the next 6-month-purchase predictions: new column exp_sales_6_month:
cltv_avg["exp_sales_6_month"] = bgf.predict(4*6,
                                            cltv_avg["frequency"],
                                            cltv_avg["recency"],
                                            cltv_avg["T"])
# Fitting the Gamma-Gamma model:
ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_avg["frequency"], cltv_avg["monetary"])

# Predicting the average benefit from each customer:
cltv_avg["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_avg["frequency"],
                                                                        cltv_avg["monetary"])

cltv_avg.sort_values("exp_average_value", ascending=False).head(20)

# Calculating the CLTV for 6 months:
cltv_avg["cltv"] = ggf.customer_lifetime_value(bgf,
                                               cltv_avg["frequency"],
                                               cltv_avg["recency"],
                                               cltv_avg["T"],
                                               cltv_avg["monetary"],
                                               time=6, #month
                                               freq="W", # week
                                               discount_rate=0.01)

# Max 20 cltv values:
cltv_avg.sort_values("cltv", ascending=False).head(20)

# Segmenting the customers according the their cltv values for 6 months:
cltv_avg["segment"] = pd.qcut(cltv_avg["cltv"], 4, labels=["D","C","B","A"])

# creating the dataset with the expected sales in 3 & 6 months,
# expected  average value, cltv and the segments
new_df = cltv_avg.drop(columns = ["recency", "T", "frequency","monetary"])
new_df.groupby("segment").agg({"cltv": ["mean", "min", "max"]})

# Creating the dataset of the customers in segment A and segment C
# to give possible suggestions to the company about the customers of segment A and C
suggestion_dataset = cltv_avg[(cltv_avg["segment"] == "A") | (cltv_avg["segment"] == "C")]

suggestion_dataset.sort_values("cltv", ascending = False)

# mean, min, max values of recency, T, frequency, monetary and cltv values according to the segments
suggestion_dataset.groupby("segment").agg({"recency": ["mean", "min", "max"],
                                           "T": ["mean", "min", "max"],
                                           "frequency":["mean", "min", "max"],
                                           "monetary":["mean", "min", "max"],
                                           "cltv": ["mean", "min", "max"]})
suggestion_dataset.describe().T

