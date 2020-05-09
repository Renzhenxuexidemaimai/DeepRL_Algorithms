#!/usr/bin/env python
# Created at 2020/3/14
import time

import click
import yaml
from tqdm import tqdm

from Algorithms.pytorch.GAIL.gail import GAIL


@click.command()
@click.option("--eval_model_epoch", type=int, default=1000, help="Intervals for evaluating model")
@click.option("--save_model_epoch", type=int, default=1000, help="Intervals for saving model")
@click.option("--save_model_path", type=str, default="../model_pkl", help="Path for saving trained model")
@click.option("--load_model", type=bool, default=False, help="Indicator for whether load trained model")
@click.option("--load_model_path", type=str, default="../model_pkl/MAGAIL_Train_2020-05-08_16:31:56",
              help="Path for loading trained model")
def main(eval_model_epoch, save_model_epoch, save_model_path, load_model,
         load_model_path):
    config_path = "../config/config.yml"

    exp_name = f"MAGAIL_Train_{time.strftime('%Y-%m-%d_%H:%M:%S', time.localtime())}"

    config = config_loader(path=config_path)  # load model configuration
    training_epochs = config["general"]["training_epochs"]

    mail = GAIL(config=config, log_dir="../log", exp_name=exp_name)

    if load_model:
        print(f"Loading Pre-trained MAGAIL model from {load_model_path}!!!")
        mail.load_model(load_model_path)

    for epoch in tqdm(range(1, training_epochs + 1)):
        mail.train(epoch)

        if epoch % eval_model_epoch == 0:
            mail.eval(epoch)

        if epoch % save_model_epoch == 0:
            mail.save_model(save_model_path)


def config_loader(path=None):
    with open(path) as f:
        config = yaml.full_load(f)
    return config


if __name__ == '__main__':
    main()
