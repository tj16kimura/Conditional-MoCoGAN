import time
import datetime
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import pytorch_lightning as pl
import torch.optim as optim
from torch.nn.utils import spectral_norm

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch

from . import loss_function


class Noise(nn.Module):
    def __init__(self, use_noise, sigma=0.2):
        super(Noise, self).__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * T.FloatTensor(x.size()).normal_()
        return x


class MyConv(nn.Module):
    def __init__(
        self,
        conv_dim,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        norm_mode=None,
    ):
        super(MyConv, self).__init__()
        self.norm_mode = norm_mode

        if conv_dim == 2:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            if norm_mode == "batch":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_mode == "spectral":
                self.layer = spectral_norm(self.conv)
        elif conv_dim == 3:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
            if norm_mode == "batch":
                self.norm = nn.BatchNorm3d(out_channels)
            elif norm_mode == "spectral":
                self.layer = spectral_norm(self.conv)

    def forward(self, x):
        if self.norm_mode is None:
            return self.conv(x)
        elif self.norm_mode == "batch":
            return self.norm(self.conv(x))
        elif self.norm_mode == "spectral":
            return self.layer(x)


class PatchImageDiscriminator(nn.Module):
    def __init__(
        self,
        n_channels,
        n_output_neurons=1,
        ndf=64,
        norm_mode="batch",
        use_noise=False,
        noise_sigma=None,
    ):
        super(PatchImageDiscriminator, self).__init__()

        self.use_noise = use_noise

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            MyConv(2, n_channels, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            MyConv(2, ndf, ndf * 2, 4, 2, 1, norm_mode=norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            MyConv(2, ndf * 2, ndf * 4, 4, 2, 1, norm_mode=norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            MyConv(2, ndf * 4, n_output_neurons, 4, 2, 1),
        )

    def forward(self, input):
        h = self.main(input).squeeze()
        return h, None


class PatchVideoDiscriminator(nn.Module):
    def __init__(
        self,
        n_channels,
        n_output_neurons=1,
        bn_use_gamma=True,
        norm_mode="batch",
        use_noise=False,
        noise_sigma=None,
        ndf=64,
    ):
        super(PatchVideoDiscriminator, self).__init__()

        self.n_channels = n_channels
        self.n_output_neurons = n_output_neurons
        self.use_noise = use_noise
        self.bn_use_gamma = bn_use_gamma

        self.main = nn.Sequential(
            Noise(use_noise, sigma=noise_sigma),
            MyConv(3, n_channels, ndf, 4, stride=(1, 2, 2), padding=(0, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            MyConv(3, ndf, ndf * 2, 4, stride=(1, 2, 2), padding=(0, 1, 1), norm_mode=norm_mode),
            nn.LeakyReLU(0.2, inplace=True),
            Noise(use_noise, sigma=noise_sigma),
            MyConv(
                3, ndf * 2, ndf * 4, 4, stride=(1, 2, 2), padding=(0, 1, 1), norm_mode=norm_mode
            ),
            nn.LeakyReLU(0.2, inplace=True),
            MyConv(3, ndf * 4, n_output_neurons, 4, stride=(1, 2, 2), padding=(0, 1, 1)),
        )

    def forward(self, input):
        h = self.main(input)
        h = h.squeeze()
        return h, None


class VideoGenerator(nn.Module):
    # z_contentにはラベル情報をコンキャットしたものを入れる
    def __init__(
        self,
        n_channels,
        dim_z_content,
        dim_z_motion,
        num_motion,
        video_length,
        ngf=64,
        img_size=96,
    ):
        super(VideoGenerator, self).__init__()

        self.n_channels = n_channels
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.num_motion = num_motion
        self.video_length = video_length

        self.init_size = int(img_size / 16)

        # za -> G
        #         dim_z = dim_z_motion + dim_z_category + dim_z_content
        #         self.recurrent = nn.GRUCell(dim_z_motion, dim_z_motion)

        # za -> Rm
        dim_z = dim_z_motion + dim_z_content
        self.recurrent = nn.GRUCell(dim_z_motion + num_motion, dim_z_motion)

        self.main = nn.Sequential(
            # (dim_z) x 1 x 1
            nn.ConvTranspose2d(dim_z, ngf * 8, self.init_size, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, self.n_channels, 4, 2, 1, bias=False),
            # (ngf) x 64 x 64
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.main(z)
        return h

    def gru(self, e, h):
        return self.recurrent(e, h)


class GANLM(pl.LightningModule):
    def __init__(
        self,
        sample_interval,
        n_channels,
        use_noise,
        noise_sigma,
        norm_mode,
        dim_z_content,
        dim_z_motion,
        video_length,
        num_content,
        num_motion,
        image_batch_size,
        video_batch_size,
        img_size,
        mean,
        std,
        lr=1e-4,
        beta1=0.9,
        beta2=0.999,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.video_length = video_length
        self.num_content = num_content
        self.num_motion = num_motion
        self.num_fix_samples = num_content * num_motion
        self.dim_z_content = dim_z_content
        self.dim_z_motion = dim_z_motion
        self.image_batch_size = image_batch_size
        self.video_batch_size = video_batch_size

        self.G = VideoGenerator(
            n_channels,
            dim_z_content + num_content,
            dim_z_motion,
            num_motion,
            video_length,
            img_size=img_size,
        )
        self.D_i = PatchImageDiscriminator(
            n_channels + num_content,
            norm_mode=norm_mode,
            use_noise=use_noise,
            noise_sigma=noise_sigma,
        )
        self.D_v = PatchVideoDiscriminator(
            n_channels + num_motion,
            norm_mode=norm_mode,
            use_noise=use_noise,
            noise_sigma=noise_sigma,
        )
        self.configure_criterions()

        self.start_time = time.time()

    def forward(self, z):
        return self.G(z)

    def configure_criterions(self):
        self.g_criterion = loss_function.loss_hinge_gen
        self.d_criterion = loss_function.loss_hinge_dis

    def configure_optimizers(self):
        opt_G = optim.Adam(
            self.G.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_D_i = optim.Adam(
            self.D_i.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_D_v = optim.Adam(
            self.D_v.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        return [opt_G, opt_D_i, opt_D_v], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        # batch = {"images":, "categories":,}
        videos, labels = batch.values()

        # train generator
        if optimizer_idx == 0:
            image_fake, _, i_label_fake = self.sample_image(self.image_batch_size)
            d_out, _ = self.D_i(self.concat_image_label(image_fake, i_label_fake))
            g_i_loss = self.g_criterion(d_out)

            video_fake, v_label_fake, _ = self.sample_video(self.video_batch_size)
            d_out, _ = self.D_v(self.concat_video_label(video_fake, v_label_fake))
            g_v_loss = self.g_criterion(d_out)

            g_loss = g_i_loss + g_v_loss

            output = {
                "loss": g_loss,
                "log_name": "loss/g_loss",
            }
            return output

        # train image discriminator
        if optimizer_idx == 1:
            i = torch.randint(0, self.video_length, (1,), dtype=torch.int32)[0]
            images = torch.stack([v[:, i, :, :] for v in videos])
            labels = labels[1]

            d_out, _ = self.D_i(self.concat_image_label(images, labels))
            real_loss = self.d_criterion(d_out, fake=False)

            image_fake, _, i_label_fake = self.sample_image(self.image_batch_size)
            d_out, _ = self.D_i(self.concat_image_label(image_fake.detach(), i_label_fake))
            fake_loss = self.d_criterion(d_out, fake=True)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            output = {
                "loss": d_loss,
                "log_name": "loss/d_i_loss",
            }
            return output

        # train video discriminator
        if optimizer_idx == 2:
            videos = videos
            labels = labels[0]

            d_out, _ = self.D_v(self.concat_video_label(videos, labels))
            real_loss = self.d_criterion(d_out, fake=False)

            video_fake, v_label_fake, _ = self.sample_video(self.video_batch_size)
            d_out, _ = self.D_v(self.concat_video_label(video_fake.detach(), v_label_fake))
            fake_loss = self.d_criterion(d_out, fake=True)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            output = {
                "loss": d_loss,
                "log_name": "loss/d_v_loss",
            }
            return output

    def training_epoch_end(self, outputs):
        log_dict = {}
        elapced_time = time.time() - self.start_time
        elapced_time = str(datetime.timedelta(seconds=elapced_time))[:-7]
        self.log("et", elapced_time, prog_bar=True, logger=False)
        for output in outputs:
            loss_avg = torch.stack([o["loss"] for o in output]).mean()
            log_dict[output[0]["log_name"]] = loss_avg
        self.log_dict(log_dict, prog_bar=True)

        if self.current_epoch % self.hparams.sample_interval == 0:
            with torch.no_grad():
                image_fake, _, _ = self.sample_image(self.num_fix_samples, random=False)
                video_fake, _, _ = self.sample_video(self.num_fix_samples, random=False)
                image_fake = make_grid(self.denormalize(image_fake), nrow=self.num_content)
                video_fake = self.denormalize(video_fake).permute(2, 0, 1, 3, 4)
                video_fake = torch.stack([make_grid(v, nrow=self.num_content) for v in video_fake])
                tbd = self.logger.experiment
                tbd.add_images("Image", image_fake.unsqueeze(0), self.current_epoch)
                tbd.add_video("Video", video_fake.unsqueeze(0), self.current_epoch)

    def denormalize(self, x):
        x[x > 1] = 1
        x[x < -1] = -1
        x = (x + 1) / 2
        return x

    def get_noise(self, num_samples):
        return T.FloatTensor(num_samples, self.dim_z_motion).normal_()

    def sample_category(self, num_samples, dim_categ, video_len=None, random=None):
        """
        zで生成するクラスを生成．ワンホットベクトルとラベルのリストを返す．
        """
        video_len = video_len if video_len is not None else self.video_length

        if random is None:
            classes_to_generate = np.random.randint(dim_categ, size=num_samples)
        else:
            if num_samples != self.num_content * self.num_motion:
                exit()
            if random == "content":
                classes_to_generate = np.array(
                    [i for i in range(self.num_content)] * self.num_motion
                )
            elif random == "motion":
                classes_to_generate = np.array(
                    [[i] * self.num_content for i in range(self.num_motion)]
                ).flatten()

        one_hot = np.zeros((num_samples, dim_categ), dtype=np.float32)
        one_hot[np.arange(num_samples), classes_to_generate] = 1
        one_hot = np.repeat(one_hot, video_len, axis=0)

        return (
            one_hot,
            torch.from_numpy(classes_to_generate).to(self.device),
        )  # requires_grad_()いらないのでは．

    def sample_z_m(self, num_samples, video_len=None, random=None):
        """
        GRUユニットにランダムノイズとカテゴリラベルを入力し，z_mを得る
        """
        video_len = video_len if video_len is not None else self.video_length
        random = random if random is None else "motion"

        z_category, z_labels = self.sample_category(num_samples, self.num_motion, 1, random=random)
        z_category = torch.from_numpy(z_category).to(self.device)

        h_t = [self.get_noise(num_samples)]

        if z_category is not None:
            for _ in range(video_len):
                e_t = torch.cat([self.get_noise(num_samples), z_category], dim=1)
                h_t.append(self.G.gru(e_t, h_t[-1]))
        else:
            for _ in range(video_len):
                e_t = self.get_noise(num_samples)
                h_t.append(self.G.gru(e_t, h_t[-1]))

        z_m_t = [h_k.view(-1, 1, self.dim_z_motion) for h_k in h_t]
        z_m = torch.cat(z_m_t[1:], dim=1).view(-1, self.dim_z_motion)

        # z_m.size() -> (num_samples * video_len, dim_z_motion)
        return z_m, z_labels

    def sample_z_c(self, num_samples, video_len=None, random=None):
        """
        Gに入力するz_cを用意する
        """
        video_len = video_len if video_len is not None else self.video_length
        random = random if random is None else "content"

        z_category, z_labels = self.sample_category(num_samples, self.num_content, random=random)

        content = np.random.normal(0, 1, (num_samples, self.dim_z_content)).astype(np.float32)
        content = np.repeat(content, video_len, axis=0)
        content = np.concatenate([content, z_category], axis=1)

        z_c = torch.from_numpy(content).to(self.device)

        return z_c, z_labels

    def sample_z(self, num_samples, video_len=None, random=None):
        """
        Gに入力するzを用意する
        """
        video_len = video_len if video_len is not None else self.video_length

        z_c, z_c_labels = self.sample_z_c(num_samples, video_len, random)
        z_m, z_m_labels = self.sample_z_m(num_samples, video_len, random)

        z = torch.cat([z_c, z_m], dim=1)

        return z, z_m_labels, z_c_labels

    def sample_video(self, num_samples, video_len=None, random=None):
        """"""
        video_len = video_len if video_len is not None else self.video_length

        z, z_m_labels, z_c_labels = self.sample_z(num_samples, video_len, random)

        z = z.view(z.size(0), z.size(1), 1, 1)
        h = self.G(z)
        h = h.view(h.size(0) // video_len, video_len, self.hparams.n_channels, h.size(3), h.size(3))

        video = h.permute(0, 2, 1, 3, 4)  # D input shape -> (N, C, T, W, H)

        return video, z_m_labels, z_c_labels

    def sample_image(self, num_samples, video_len=None, random=None):
        """"""
        video_len = video_len if video_len is not None else self.video_length

        z, z_m_labels, z_c_labels = self.sample_z(num_samples, video_len, random)

        j = np.array([i * video_len for i in range(num_samples)]) + np.random.choice(video_len)
        z = z.view(z.size(0), z.size(1), 1, 1)
        image = self.G(z)
        image = image[j, ::]

        return image, z_m_labels, z_c_labels

    def concat_image_label(self, image, label):
        B, C, H, W = image.shape
        oh = torch.eye(self.num_content)[label].view(-1, self.num_content, 1, 1)
        oh = oh.expand(B, self.num_content, H, W).to(self.device)
        return torch.cat((image, oh), dim=1)

    def concat_video_label(self, video, label):
        B, C, T, H, W = video.shape
        oh = torch.eye(self.num_motion)[label].view(-1, self.num_motion, 1, 1, 1)
        oh = oh.expand(B, self.num_motion, T, H, W).to(self.device)
        return torch.cat((video, oh), dim=1)

    def generate_fake_label(self, labels, kind):
        div = self.num_content if kind == "content" else self.num_motion
        fake_labels = (labels + torch.randint(1, div, labels.size()).to(self.device)) % div
        return fake_labels
