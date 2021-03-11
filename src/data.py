import os
import tqdm
import pickle
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
import logging

logger = logging.getLogger(__name__)


class VideoFolderDataset(Dataset):
    def __init__(
        self,
        folder,
        attr_path,
        transform=None,
        cache_path=".cache/processed_data.pkl",
    ):
        self.transform = transform if transform is not None else lambda x: x
        self.length = []
        self.images = []

        if os.path.exists(cache_path):
            with open(cache_path, "rb") as fp:
                self.images, self.length = pickle.load(fp)
            print(f"cache file {cache_path} was loaded !!")
        else:
            self.attr2idx = {}
            self.idx2attr = {}

            attr_path = os.path.join(folder, attr_path)
            lines = [line.rstrip() for line in open(attr_path, "r")]
            all_attr_names = lines[0].split()
            for i, attr_name in enumerate(all_attr_names):
                self.attr2idx[attr_name] = i
                self.idx2attr[i] = attr_name

            lines = lines[1:]

            for i, line in enumerate(tqdm.tqdm(lines)):
                splited = line.split()
                filename = os.path.join(folder, splited[0])
                values = [int(v) for v in splited[1:]]

                video = PIL.Image.open(filename).convert("RGB")
                video = np.array(video)
                horizontal = video.shape[1] > video.shape[0]
                shorter, longer = (
                    min(video.shape[0], video.shape[1]),
                    max(video.shape[0], video.shape[1]),
                )
                video_len = longer // shorter
                video = np.split(video, video_len, axis=1 if horizontal else 0)
                video = [self.transform(v) for v in video]
                # video = torch.cat(video).reshape(len(video), *video[0].shape)
                video = torch.stack(video)

                self.images.append((video, values))
                self.length.append(video_len)

            if cache_path is not None:
                with open(cache_path, "wb") as fp:
                    pickle.dump((self.images, self.length), fp)
                print(f"cache file {cache_path} was saved.")

        self.cumsum = np.cumsum([0] + self.length)
        print(f"Total number of videos {len(self.images)}")

    def __getitem__(self, item):
        im, attr = self.images[item]
        # im = torch.load(path)
        return im, attr

    def __len__(self):
        return len(self.images)


class ImageDataset(Dataset):
    def __init__(
        self,
        dataset,
    ):
        self.dataset = dataset

    def __getitem__(self, item):
        if item != 0:
            video_id = np.searchsorted(self.dataset.cumsum, item) - 1
            frame_num = item - self.dataset.cumsum[video_id] - 1
        else:
            video_id = 0
            frame_num = 0

        video, attr = self.dataset[video_id]
        return {"images": video[frame_num], "categories": attr}

    def __len__(self):
        return self.dataset.cumsum[-1]


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset,
        video_length,
        every_nth=1,
    ):
        self.dataset = dataset
        self.video_length = video_length
        self.every_nth = every_nth

    def __getitem__(self, item):
        video, attr = self.dataset[item]

        video_len = len(video)
        if video_len >= self.video_length * self.every_nth:
            needed = self.every_nth * (self.video_length - 1)
            gap = video_len - needed
            start = 0 if gap == 0 else np.random.randint(0, gap, 1)[0]
            subsequence_idx = np.linspace(
                start, start + needed, self.video_length, endpoint=True, dtype=np.int32
            )
        elif video_len >= self.video_length:
            subsequence_idx = np.arrange(0, self.video_length)
        else:
            raise Exception(f"length is too short id - {self.dataset[item]}, len - {video_len}")

        selected = video[subsequence_idx].permute(1, 0, 2, 3)  # -> (C, T, H, W)
        return {"images": selected, "categories": attr}

    def __len__(self):
        return len(self.dataset)


class ImageVideoLDM(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        attr_path,
        image_batch_size,
        video_batch_size,
        video_length,
        every_nth,
        img_size,
        mean,
        std,
        cache_path,
        **kwargs,
    ):
        self.train_dir = train_dir
        self.attr_path = attr_path
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size
        self.video_length = video_length
        self.every_nth = every_nth
        self.img_size = img_size
        self.mean = mean
        self.std = std
        self.cache_path = cache_path

        self.transform = transforms.Compose(
            [
                PIL.Image.fromarray,
                transforms.Resize([self.img_size, self.img_size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = VideoFolderDataset(
                self.train_dir, self.attr_path, self.transform, self.cache_path
            )
            self.video_dataset = VideoDataset(dataset, self.video_length, self.every_nth)

    def train_dataloader(self):
        video_loader = DataLoader(
            self.video_dataset, self.video_batch_size, shuffle=True, num_workers=4
        )
        return video_loader
