import argparse
import sys
from logging import getLogger
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation, construct_transform
from recbole.utils import init_seed, init_logger, get_model, get_flops, set_color, get_trainer, get_environment


def run_model(model=None, dataset=None, config_files=None, config_dict=None, saved=True, mode="train", model_file=None):
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_files,
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])

    init_logger(config)

    # logger initialization
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)
    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)
    res_dict = config.final_config_dict.copy()

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    logger.info(model)
    if model_file:
        checkpoint = torch.load(model_file, map_location=config["device"])
        model.load_state_dict(checkpoint['state_dict'])
    transform = construct_transform(config)
    """
    preprocess training data
    """
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")

    # trainer loading and initialization
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_result = None
    best_valid_score = None
    # model training
    if mode == "train":
        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data, saved=saved, show_progress=config["show_progress"]
        )
    if mode == "plot":
        interaction = next(iter(train_data))
        interaction = interaction.to("cuda")
        with torch.no_grad():
            output = model.calculate_loss(interaction)
    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"],model_file=model_file
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )
    if mode == "train":
        logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")
    best_epoch = trainer.best_epoch
    result = {
        "best_epoch": best_epoch,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }
    return result  # for the single process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )

    parser.add_argument("--config_files", type=str, default=None, help="config files")

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    run_model(
        model="AHRec",
        dataset=args.dataset,
        config_files=config_file_list,
    )
