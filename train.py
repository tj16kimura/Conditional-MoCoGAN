import os
import argparse
import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import logging

from src import data, model


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    # ---------------
    # meta info
    # ---------------
    parser.add_argument("--memo", type=str, default="")
    parser.add_argument("--sample_interval", type=int, default=300)
    parser.add_argument("--model_save_interval", type=int, default=0)
    parser.add_argument("--save_top_k", type=int, default=1)

    # ---------------
    # hyper params
    # ---------------
    parser.add_argument("--epoch", type=int, default=50000)
    parser.add_argument("--gpus", type=int, nargs='+', default=[0])
    parser.add_argument("--lr", type=float, default=2e-4)

    # ---------------
    # model conf
    # ---------------
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--use_noise", type=bool, default=True)
    parser.add_argument("--noise_sigma", type=float, default=0.1)
    parser.add_argument("--norm_mode", type=str, default="spectral")
    parser.add_argument("--dim_z_content", type=int, default=30)
    parser.add_argument("--dim_z_motion", type=int, default=30)
    parser.add_argument("--num_content", type=int, default=9)
    parser.add_argument("--num_motion", type=int, default=4)

    # ---------------
    # dataset conf
    # ---------------
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--cache_path", type=str, default=".cache/processed_data.pkl")
    parser.add_argument("--attr_path", type=str, default="attr_limited.txt")
    parser.add_argument('--video_length', type=int, default=16)
    parser.add_argument('--every_nth', type=int, default=2)

    # ---------------
    # transforms
    # ---------------
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--image_batch_size', type=int, default=32)
    parser.add_argument('--video_batch_size', type=int, default=32)
    parser.add_argument('--mean', type=float, nargs='+', default=[0.5, 0.5, 0.5])
    parser.add_argument('--std', type=float, nargs='+', default=[0.5, 0.5, 0.5])

    args = parser.parse_args()

    date = (datetime.datetime.utcnow() + datetime.timedelta(hours=9)).strftime("%Y%m%d_%H%M%S")
    logger_pl = TensorBoardLogger(save_dir="logs", name="", version=date)
    model_save_dir = os.path.join("logs", date, "models")
    os.makedirs(model_save_dir, exist_ok=True)

    dict_args = vars(args)

    dm = data.ImageVideoLDM(**dict_args)
    dm.prepare_data()
    dm.setup()

    checkpoint_callback = ModelCheckpoint(
        dirpath=model_save_dir,
        filename="{epoch:08d}",
        save_top_k=args.save_top_k,
        period=args.model_save_interval
    )

    lm = model.GANLM(**dict_args)
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epoch,
        logger=logger_pl,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(lm, dm)


if __name__ == "__main__":
    main()
