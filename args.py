import argparse
import nestargs

from datetime import datetime
from pathlib import Path

parser = nestargs.NestedArgumentParser(description="Surgery effect prediction")
parser.add_argument(
    "--model_name", default="random_forest", type=str, help="Model to use"
)
parser.add_argument(
    "--random_forest.n_estimators",
    default=10,
    type=int,
    help="Number of estimators to use in Random Forest",
)
parser.add_argument(
    "--random_forest.random_state",
    default=42,
    type=int,
    help="Random seed",
)
parser.add_argument(
    "--output_dir", default="saved_models", type=str, help="Output dir to use"
)
# parser.add_argument(
#     "--linear_regression.foo",
#     default=10,
#     type=int,
#     help="Number of estimators to use in Random Forest",
# )
# parser.add_argument(
#     "--linear_regression.bar",
#     default=42,
#     type=int,
#     help="Random seed",
# )

# main_parser = argparse.ArgumentParser(description="Surgery effect prediction")
# # Main Training Parameters
# main_parser.add_argument(
#     "--model_name", default="random_forest", type=str, help="Model to use"
# )

# # sub commands
# subparsers = main_parser.add_subparsers(dest="model_params")

# # Random Forest Params
# random_forest_parser = subparsers.add_parser("random_forest")
# random_forest_parser.add_argument(
#     "--n_estimators",
#     default=10,
#     type=int,
#     help="Number of estimators to use in Random Forest",
# )
# random_forest_parser.add_argument(
#     "--random_state",
#     default=42,
#     type=int,
#     help="Random seed",
# )
# random_forest_parser.set_defaults(func="random_forest")

# # Linear Regression params
# linear_regression_parser = subparsers.add_parser("linear_regression")
# linear_regression_parser.add_argument(
#     "--boo",
#     default=10,
#     type=int,
#     help="Number of estimators to use in Random Forest",
# )
# linear_regression_parser.add_argument(
#     "--foo",
#     default=42,
#     type=int,
#     help="Random seed",
# )
# linear_regression_parser.set_defaults(func="linear_regression")
def setup_dir(args):
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y-%H_%M_%S")
    directory = "saved_models/{}/{}".format(args.model_name, date_time)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_args():
    # namespace = main_parser.parse_args()
    args = parser.parse_args()
    args.output_dir = setup_dir(args)
    return args
