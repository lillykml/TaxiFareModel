import pandas as pd
from sklearn.model_selection import train_test_split

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"


def get_data(nrows=10_000):
    '''returns a DataFrame with nrows from s3 bucket'''
    df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def clean_data(df, test=False):

    #Drop Nans
    df = df.dropna(how='any', axis='rows')

    #Drop lat and lons if they are 0
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]

    #Keep Fare amount between 0 and 4k
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]

    #Keep passengers between 1 and 8
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]

    #Keep lat and lon between a certain range
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]

    return df


def holdout(df):
    y = df["fare_amount"]
    X = df.drop(columns="fare_amount")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    return (X_train, X_test, y_train, y_test)


if __name__ == '__main__':
    df = get_data()
