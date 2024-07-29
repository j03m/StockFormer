import datetime

import numpy as np
import pandas as pd
from Transformer.utils.yahoodownloader import YahooDownloader
from Transformer import config
from stockstats import StockDataFrame as Sdf


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            user user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature

    def preprocess_data(self, df, type=0):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            if type != 0:
                df = self.add_technical_indicator2(df)
            else:
                df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        return df

    def add_technical_indicator2(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicators = []
            for i in range(len(unique_ticker)):

                temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = unique_ticker[i]
                temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                    "date"
                ].to_list()
                for s in range(1,3):
                    temp_indicator[indicator+'_'+str(s)] = temp_indicator[indicator].shift(s)
                indicators.append(temp_indicator)
            indicator_df = pd.concat(indicators, ignore_index=True)
            df = df.merge(
                indicator_df[["tic", "date", indicator, indicator+'_'+str(1), indicator+'_'+str(2)]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in self.tech_indicator_list:
            indicators = []
            for i in range(len(unique_ticker)):

                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["date"] = df[df.tic == unique_ticker[i]][
                        "date"
                    ].to_list()
                    indicators.append(
                        temp_indicator)
            indicator_df = pd.concat(indicators, ignore_index=True)
            df = df.merge(
                indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left"
            )
        df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df_vix = YahooDownloader(
            start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        ).fetch_data()
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]

        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    # Calculate turbulence index based on Dow 30
    def calculate_turbulence(self, data):
        """Calculate turbulence index based on Dow 30"""

        # Make a copy of the input data
        df = data.copy()

        # Create a pivot table with dates as rows and tickers as columns, values are closing prices
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")

        # Calculate daily returns (percentage change)
        df_price_pivot = df_price_pivot.pct_change()

        # Get unique dates in the dataset
        unique_date = df.date.unique()

        # Start calculation after the first 252 trading days (one year of data)
        start = 252

        # Initialize the turbulence index list with zeros for the first 252 days
        turbulence_index = [0] * start

        # Counter to keep track of non-zero turbulence values
        count = 0

        # Loop over each date starting from the 252nd day to the end of the unique dates
        for i in range(start, len(unique_date)):

            # Get the price data for the current date
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]

            # Get the historical price data for the past 252 days (one year)
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i]) &
                (df_price_pivot.index >= unique_date[i - 252])
                ]

            # Filter historical data to remove columns (tickers) with too many missing values
            filtered_hist_price = hist_price.iloc[
                                  hist_price.isna().sum().min():
                                  ].dropna(axis=1)

            # Calculate the covariance matrix of the filtered historical prices
            cov_temp = filtered_hist_price.cov()

            # Calculate the deviation of the current price from the historical mean
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )

            # Calculate the turbulence value using the Mahalanobis distance
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )

            # Check if the turbulence value is positive
            if temp > 0:
                count += 1
                # Avoid large outliers by initializing with zeros for the first few values
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    turbulence_temp = 0
            else:
                turbulence_temp = 0

            # Append the calculated turbulence value to the list
            turbulence_index.append(turbulence_temp)

        # Create a DataFrame with the dates and corresponding turbulence index values
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )

        # Return the DataFrame containing the turbulence index
        return turbulence_index

