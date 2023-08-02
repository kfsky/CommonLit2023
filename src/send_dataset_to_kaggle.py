import argparse
import json
import os

from kaggle.api.kaggle_api_extended import KaggleApi

# 定数
COMPETITION_NAME = "CommonLit2023"

# config情報を読み込み
# https://blog.katty.in/9994
f = open("./conf/kaggle.json", "r")
KAGGLE_CONFIG = json.load(f)
os.environ["KAGGLE_USERNAME"] = KAGGLE_CONFIG["username"]
os.environ["KAGGLE_KEY"] = KAGGLE_CONFIG["key"]


def dataset_create_new(dataset_name: str, upload_dir: str):
    dataset_metadata = {}
    dataset_metadata["id"] = f"{os.environ['KAGGLE_USERNAME']}/{dataset_name}"
    dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
    dataset_metadata["title"] = dataset_name

    with open(os.path.join(upload_dir, "dataset-metadata.json"), "w") as f:
        json.dump(dataset_metadata, f)

    api = KaggleApi()
    api.authenticate()

    if f"{os.environ['KAGGLE_USERNAME']}/{dataset_name.lower()}" not in [
        str(d) for d in api.dataset_list(user=os.environ["KAGGLE_USERNAME"], search="commonlit2023")
    ]:
        print("New Dataset")
        api.dataset_create_new(folder=upload_dir, dir_mode="tar", convert_to_csv=False)

    else:
        print("Update Dataset")
        api.dataset_create_version(
            folder=upload_dir, version_notes="update", convert_to_csv=False, delete_old_versions=True, dir_mode="tar"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, default="all")

    args = parser.parse_args()

    # upload dir
    if args.exp_name == "all":
        upload_dir = "./outputs"
    else:
        upload_dir = os.path.join("./outputs", args.exp_name)

    dataset_create_new(dataset_name=COMPETITION_NAME + "-" + args.exp_name, upload_dir=upload_dir)


if __name__ == "__main__":
    main()
