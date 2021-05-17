import argparse
#import nestargs

from datetime import datetime
from pathlib import Path


parser = argparse.ArgumentParser(description='Spine decompression')
parser.add_argument(
    "--model_name", default="random_forest", type=str, help="Model to use", choices=['random_forest', 'linear_regression', 'gradient_boost', 'decision_tree']
)
#Random Forest params
# parser.add_argument(
#     "--rf_n_estimators",
#     default=10,
#     type=int,
#     help="Number of estimators to use in Random Forest",
# )
# parser.add_argument(
#     "--rf_random_state",
#     default=42,
#     type=int,
#     help="Random seed",
# )
# #Gradient Boost params
# parser.add_argument(
#     "--gb_n_estimators",
#     default=10,
#     type=int,
#     help="Number of estimators to use in Gradient Boost",
# )
# parser.add_argument(
#     "--gb_learning_rate",
#     default=0.1,
#     type=float,
#     help="Learning rate to use in Gradient Boost",
# )

parser.add_argument(
    "--dataset_path", default="data/final_data.csv", type=str, help="Dataset dir to use"
)
parser.add_argument(
    "--post_preproc_data_path", default="data/no_categorical_data.csv", type=str, help="Dataset dir to use to store the dataset after preprocessing"
)
def setup_dir(args):
    # now = datetime.now()
    # directory = "models/{}/model".format(args.model_name)
    # Path(directory).mkdir(parents=True, exist_ok=True)
    # return directory
    return "models"


def get_args():
    # namespace = main_parser.parse_args()
    args = parser.parse_args()
    args.output_dir = setup_dir(args)
    return args
