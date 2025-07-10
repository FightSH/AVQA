import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


import timm
import tome
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
import numpy as np
import argparse
from open_clip  import create_model_from_pretrained


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# We use the timm augreg models here, but you can use any supported implementation.
# model_name = "vit_large_patch16_384"

# model = timm.create_model(model_name, pretrained=True)

# model.head = Identity()
# model.global_pool = None

# Source tracing is necessary for visualization!
# tome.patch.timm(model, trace_source=True)

# input_size = model.default_cfg["input_size"][1]


# Make sure the transform is correct for your model!
# transform_list = [
#     transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
#     transforms.CenterCrop(input_size)
#     # transforms.Resize([384,384])
# ]

# # The visualization and model need different transforms
# transform_vis  = transforms.Compose(transform_list)
# transform_norm = transforms.Compose(transform_list + [
#     transforms.ToTensor(),
#     transforms.Normalize(model.default_cfg["mean"], model.default_cfg["std"]),
# ])


def TransformImage(img, input_size, mean, std):
    transform_list = [transforms.Resize(int((256 / 224) * input_size), interpolation=InterpolationMode.BICUBIC),
                      transforms.CenterCrop(input_size)
                      # transforms.Resize([384,384])
                      ]

    # The visualization and model need different transforms
    transform_vis = transforms.Compose(transform_list)
    transform_norm = transforms.Compose(transform_list + [transforms.ToTensor(),
                                                          transforms.Normalize(mean, std),
                                                          ])

    return transform_vis(img), transform_norm(img)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", dest='gpu', type=str, default='3',
                        help='Set CUDA_VISIBLE_DEVICES environment variable, optional')
    parser.add_argument("--dir_path", type=str, default='/mnt/sda/shenhao/datasets/MUSIC-AVQA/frames/',
                        help='sec path')
    parser.add_argument("--dst_path", type=str, default='/mnt/sda/shenhao/datasets/siglip2/MUSIC-AVQA/tome/',
                        help='sec save path')
    parser.add_argument("--sample_frames", type=int, default=60, help='sample frames')
    parser.add_argument("--tokens", type=int, default=14, help='merge tokens numbers')
    parser.add_argument("--layers", type=int, default=23, help='merge tokens layers, 25, 23, 22, 20, 16, 8')
    parser.add_argument("--model_name", type=str, default="vit_so400m_patch16_siglip_384", help='model_name')
    parser.add_argument("--global_pool", type=str, default="None", help='global_pool')
    
    # import timm
    # vit_models = [m for m in timm.list_models() if 'vit' in m.lower()]
    # print(vit_models)


    torch.hub.set_dir('/mnt/sda/shenhao/models/hub')
    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    params = vars(args)


    model_name = args.model_name
    model = timm.create_model(model_name, pretrained=True,cache_dir='/mnt/sda/shenhao/models/hub').to(device)
# 
    model.head = Identity()
    model.global_pool = None
    if hasattr(model, 'fc_norm'):
        model.fc_norm = Identity()
    tome.patch.timm(model, trace_source=True)
    # 检查是否正确 patch
    # print("Model successfully patched:", hasattr(model, '_tome_info'))

    input_size = model.default_cfg["input_size"][1]
    mean = model.default_cfg["mean"]
    std = model.default_cfg["std"]

    # model = nn.DataParallel(model).cuda()
    model = model.to(device)

    video_list = os.listdir(args.dir_path)
    video_list.sort()
    video_nums = len(video_list)
    cnt = 0

    for video_name in video_list:

        cnt += 1
        print("\n-->: ", cnt, "/", video_nums, " --- ", video_name)

        output_file_path = os.path.join(args.dst_path, video_name + ".npy")
        if os.path.exists(output_file_path):
            print("This file is already processed!")
            continue

        img_list_path = os.path.join(args.dir_path, video_name)
        img_list_all = os.listdir(img_list_path)

        # samples = np.linspace(0, len(img_list_all) - 2, args.sample_frames, dtype=int)
        # samples = np.round(np.linspace(0, args.params_frames - 1, args.params_frames))

        # img_list = [img_list_all[int(sample)] for sample in samples]
        img_list = img_list_all
        token_feat = torch.zeros([args.sample_frames, args.tokens, 1152])

        idx = 0
        for img_name in img_list:
            img_file = os.path.join(img_list_path, img_name)
            img = Image.open(img_file)
            # img_vis = transform_vis(img)
            # img_tensor = transform_norm(img)[None, ...]
            img_vis, img_tensor = TransformImage(img, input_size, mean, std)
            img_tensor = img_tensor[None, ...]

            model.r = [25] * args.layers

            with torch.no_grad():
                # output = model(img_tensor.to('cuda'))
                # output = model.vision_model(img_tensor.to(device))
                output = model(img_tensor.to(device))
                
               

            # output = model(img_tensor.cuda())
            print("output: ", output.shape)
            assert output.shape == (1, args.tokens, 1152)
            token_feat[idx, :, :] = output[:, :, :]
            idx += 1

            # source = model._tome_info["source"]
            # vis_img = tome.make_visualization(img_vis, source, patch_size=16, class_token=True)
            # img_name = img_name.replace(".png", "_")
            # save_name = os.path.join(save_path, img_name + str(source.shape[1]) + '.png')
            # save_name_org = os.path.join(save_path, img_name + '.png')
            # vis_img.save(save_name)
            # img_vis.save(save_name_org)

        token_feat = token_feat.detach().cpu().numpy()
        assert token_feat.shape == (args.sample_frames, args.tokens, 1152)

        np.save(output_file_path, token_feat)

    print("\n------------> finished <-----------------------")