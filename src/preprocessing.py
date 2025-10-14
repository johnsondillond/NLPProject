"""
This file is for preprocessing our data.
That way it is ready for our model to use it as input.
"""
import polars as pl

def join_data():
    """
    This function joins the training and validation data
    so that we can do cross-fold validation with our own splits
    instead of having one designated validation set.
    """
    train_df = pl.scan_csv('data/train.csv')
    valid_df = pl.scan_csv('data/valid.csv')
    total_df = pl.concat(
        [
            train_df,
            valid_df
        ],
        how='vertical'
    )
    total_df.sink_csv('data/total.csv')
    total_df.head()

join_data()
