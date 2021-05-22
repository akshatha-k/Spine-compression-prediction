import argparse

# import nestargs

from datetime import datetime
from pathlib import Path


parser = argparse.ArgumentParser(description="Spine decompression")
parser.add_argument(
    "--model_name",
    default="random_forest",
    type=str,
    help="Model to use",
    choices=["random_forest", "linear_regression", "gradient_boost", "decision_tree"],
)
parser.add_argument(
    "--Age",
    default=62,
    type=int,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--Heightcm",
    default=170,
    type=int,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--Weightkg",
    default=72,
    type=int,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--BMI",
    default=26,
    type=int,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--level",
    default="L3L4",
    type=str,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--Gender",
    default="m",
    type=str,
    help="Number of estimators to use in Random Forest",
)
# Random Forest params
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
    "--parent_path",
    default=".",
    type=str,
    help="Dataset dir to use",
)
parser.add_argument(
    "--dataset_path", default="data/final_data.csv", type=str, help="Dataset dir to use"
)
parser.add_argument(
    "--post_preproc_data_path",
    default="data/no_categorical_data.csv",
    type=str,
    help="Dataset dir to use to store the dataset after preprocessing",
)
parser.add_argument(
    "--model_path", default="models", type=str, help="Dataset dir to use"
)


def get_args():
    # namespace = main_parser.parse_args()
    args = parser.parse_args()
    args.dataset_path = "{}/{}".format(args.parent_path, args.dataset_path)
    args.post_preproc_data_path = "{}/{}".format(
        args.parent_path, args.post_preproc_data_path
    )
    args.model_path = "{}/{}".format(args.parent_path, args.model_path)
    return args
