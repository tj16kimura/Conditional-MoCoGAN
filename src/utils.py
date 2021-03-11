import os
import numpy as np
import matplotlib.pyplot as plt

from torchvision import utils as vu

from PIL import Image


def save_videos(video, epoch, save_dir=None):
    video = video.data.cpu().numpy().transpose(0, 2, 3, 4, 1)
    video[video < -1] = -1
    video[video > 1] = 1
    video = (video + 1) / 2 * 255
    video = video.astype('uint8')
    for i, v in enumerate(video):
        path = os.path.join(save_dir, '{}_{}.jpg'.format(epoch, i))
        get_concat(v, path)


def get_concat(im_arr, path):
    '''
    Args:
        - im_arr (ndarray(frame, height, width, channel))
        - path (save_path)
    '''
    height, width = im_arr[0].shape[:2]
    shorter, longer = min(width, height), max(width, height)
    ims = []
    for im in im_arr:
        ims.append(Image.fromarray(im).resize((shorter, shorter)))

    im0 = ims[0]
    for im in ims[1:]:
        dst = Image.new('RGB', (im0.width + im.width, im0.height))
        dst.paste(im0, (0, 0))
        dst.paste(im, (im0.width, 0))
        im0 = dst

    im0.save(path)


def show_batch(batch):
    normed = batch * 0.5 + 0.5
    is_video_batch = len(normed.size()) > 4

    if is_video_batch:
        rows = [vu.make_grid(b.permute(1, 0, 2, 3), nrow=b.size(1)).numpy() for b in normed]
        im = np.concatenate(rows, axis=1)
    else:
        im = vu.make_grid(normed).numpy()

    im = im.transpose((1, 2, 0))

    plt.imshow(im)
    plt.show(block=True)


def concat_h(im_arr, interval=0):
    '''
    Args:
        - im_arr (ndarray(frame, height, width, channel))
        - path (save_path)
    '''
#     height, width = im_arr[0].shape[:2]
#     shorter, longer = min(width, height), max(width, height)
#     ims = []
#     for im in im_arr:
#         ims.append(Image.fromarray(im))
    ims = im_arr
    im0 = ims[0]
    w, h = im0.width, im0.height
    for im in ims[1:]:
        dst = Image.new('RGB', (im0.width + w + interval, im0.height), (255, 255, 255))
        dst.paste(im0, (0, 0))
        dst.paste(im, (im0.width + interval, 0))
        im0 = dst

    return im0


def concat_v(im_arr, interval=0):
    '''
    Args:
        - im_arr (ndarray(frame, height, width, channel))
        - path (save_path)
    '''
    ims = im_arr
    im0 = ims[0]
    w, h = im0.width, im0.height
    for im in ims[1:]:
        dst = Image.new('RGB', (im0.width, im0.height + h + interval), (255, 255, 255))
        dst.paste(im0, (0, 0))
        dst.paste(im, (0, im0.height + interval))
        im0 = dst

    return im0


def marge_images(ims_list):
    num_gifs = len(ims_list)
    rows = 4
    cols = 8

    gif_srcs = []
    for k in ims_list:
        ims = ims_list[k]
        frames = []
        for f in range(len(ims[0])):
            gen_frames = []
            for v in ims:
                gen_frames.append(v[f])
            row = []
            for y in range(rows):
                column = []
                for x in range(cols):
                    column.append(gen_frames[y*cols + x])
                row.append(concat_h(column))
            frames.append(concat_v(row))
        gif_srcs.append(frames)

    return gif_srcs


def save_gif(src_path, trg_path):
    img = Image.open(src_path)
    height, width = img.size
    shorter, longer = min(width, height), max(width, height)

    ims = []
    for i in range(longer//shorter):
        p1 = i * shorter
        p2 = (i+1) * shorter
        ims.append(img.crop((p1, 0, p2, shorter)))
    ims[0].save(trg_path, save_all=True, append_images=ims[1:], optimize=False, duration=100, loop=0)


def split_frames(src_path):
    img = Image.open(src_path)
    height, width = img.size
    shorter, longer = min(width, height), max(width, height)

    ims = []
    for i in range(longer//shorter):
        p1 = i * shorter
        p2 = (i+1) * shorter
        ims.append(img.crop((p1, 0, p2, shorter)))

    return ims
