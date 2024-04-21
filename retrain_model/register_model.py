import mlflow
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--model_path", required=True)
args = parser.parse_args()

mlflow.register_model(
    model_uri=args.model_path,
    name=args.model_name
)