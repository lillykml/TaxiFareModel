import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.data import get_data, clean_data, holdout
from TaxiFareModel.utils import compute_rmse


class Trainer():
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),
                              ('stdscaler', StandardScaler())])

        time_pipe = Pipeline([('time_enc', TimeFeaturesEncoder('pickup_datetime')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude','dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])],
                                         remainder="drop")

        pipe = Pipeline([('preproc', preproc_pipe),
                         ('linear_model', LinearRegression())])
        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""

        self.X_train, self.X_test, self.y_train, self.y_test = holdout(self.X, self.y)

        self.set_pipeline()

        self.pipeline.fit(self.X_train, self.y_train)


    def evaluate(self, ):
        """evaluates the pipeline on df_test and return the RMSE"""
        self.y_pred = self.pipeline.predict(self.X_test)

        return compute_rmse(self.y_pred, self.y_test)


if __name__ == "__main__":
    df = get_data()
    df_clean = clean_data(df)
    X = df_clean.drop(columns="fare_amount")
    y = df_clean["fare_amount"]
    trainer = Trainer(X,y)
    trained_pipe = trainer.run()
    trainer.evaluate()
