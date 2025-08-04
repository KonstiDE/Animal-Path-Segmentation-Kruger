import os

from preprocessing.dataframe_builder_stolsnek import walk as walk_stolsnek
from preprocessing.dataframe_builder_kogeretal import walk as walk_koger
from utils.dataframe_viewer import view_dataframe
from utils.distributor import move_into_split


if __name__ == '__main__':
    # Build data
    walk_stolsnek(["part1", "part8"])

    # Put data into training, validation, and test split
    move_into_split()

    # view_dataframe()
    # view_dataframe("df_pkl_part1_3_15")
    # view_dataframe("df_pkl_part1_4_15")
