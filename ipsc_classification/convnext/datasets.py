# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms

from timm.data.constants import \
    IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        if is_train:
            print(f'loading training data from {root}')
        else:
            print(f'loading validation data from {root}')

        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes

        print(f'found {len(dataset.imgs)} images with classes: {dataset.classes}')
    else:
        raise NotImplementedError()
    print("Number of classes: %d" % nb_classes)

    return dataset, nb_classes


IPSC_EXT_REORG_ROI_MEAN = (0.513, 0.513, 0.513)
IPSC_EXT_REORG_ROI_STD = (0.06, 0.06, 0.06)

def _pytorch_interp(method):
    if method == 'bicubic':
        return transforms.InterpolationMode.BICUBIC
    elif method == 'lanczos':
        return transforms.InterpolationMode.LANCZOS
    elif method == 'hamming':
        return transforms.InterpolationMode.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return transforms.InterpolationMode.BILINEAR


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IPSC_EXT_REORG_ROI_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IPSC_EXT_REORG_ROI_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:
            t.append(
                transforms.Resize((args.input_size, args.input_size),
                                  interpolation=transforms.InterpolationMode.BICUBIC),
            )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
