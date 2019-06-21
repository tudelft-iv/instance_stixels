# This file is part of Instance Stixels:
# https://github.com/tudelft-iv/instance-stixels
#
# Original:
# Copyright (c) 2017, Fisher Yu
# BSD 3-Clause License
#
# Modifications:
# Copyright (c) 2019 Thomas Hehn.
#
# Instance Stixels is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Instance Stixels is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Instance Stixels. If not, see <http://www.gnu.org/licenses/>.

import argparse

import numpy as np

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import transforms

from matplotlib import pyplot as plt
import PIL.Image

from datasets import Cityscapes
from datasets.transforms import ModeDownsample

from models.DRNDownsampled import DRNRegressionDownsampled
from visualization import visualize_positionplusoffset, visualize_semantics
from visualization import visualize_offsethsv
from utils import check_mkdir

def main(model_name):
    # TODO: parse args.
    n_classes = 19
    #batch_size = 2
    batch_size = 1 #24
    n_workers = 12
    n_semantic_pretrain = 0 # 500 # First train only on semantics.
    n_epochs = 500
    validation_step = 15
    # TODO: implement resize as pil_transform
    resize = None # (256, 512)
    cityscapes_directory = "/home/<someuser>/cityscapes"
    output_directory = "tmp/"
    # Uncomment next line when you've set all directories.
    raise ValueError("Please set the input/output directories.")
    checkpoint = None
    #checkpoint = (
    #        "weights/...pth",
    #        <fill in epoch>)

    # --- Setup loss functions.
    classification_loss = nn.CrossEntropyLoss(ignore_index=255)
    regression_loss = nn.MSELoss(reduction='elementwise_mean')

    print("--- Load model.")
    if model_name == 'DRNRegressionDownsampled':
        classification_loss = None
        regression_loss = nn.MSELoss(reduction='elementwise_mean')
        dataset_kwargs = {
                'pil_transforms' : None,
                'gt_pil_transforms' : [ModeDownsample(8)],
                'fit_gt_pil_transforms' : [transforms.Resize(
                                                size=(784//8, 1792//8), 
                                                interpolation=2)],
                'input_transforms' : [transforms.Normalize(
                                         mean=[0.290101, 0.328081, 0.286964],
                                         std=[0.182954, 0.186566, 0.184475])],
                'tensor_transforms' : None }
        model = DRNRegressionDownsampled(
                       model_name='drn_d_22',
                       classes=n_classes,
                       pretrained_dict=torch.load('./weights/drn_d_22_cityscapes.pth'))
        model.cuda()
        parameters = model.parameters()
    else:
        raise ValueError("Model \"{}\" not found!".format(model_name))

    optimizer = optim.Adam(parameters)
    start_epoch = 0

    if checkpoint is not None:
        print("Loading from checkpoint {}".format(checkpoint))
        model.load_state_dict(torch.load(checkpoint[0]))
        optimizer.load_state_dict(torch.load(checkpoint[1]))
        start_epoch = checkpoint[2]+1

    print("--- Setup dataset and dataloaders.")
    train_set = Cityscapes(data_split='subtrain', 
                           cityscapes_directory=cityscapes_directory,
                           **dataset_kwargs)
    train_loader = data.DataLoader(train_set, batch_size=batch_size,
                                   num_workers=n_workers, shuffle=True)

    val_set = Cityscapes(data_split='subtrainval',
                         cityscapes_directory=cityscapes_directory,
                         **dataset_kwargs)
    val_loader = data.DataLoader(val_set, batch_size=batch_size,
                                 num_workers=n_workers)

    # Sample 10 validation indices for visualization.
    #validation_idxs = np.random.choice(np.arange(len(val_set)), 
    #                                   size=min(9, len(val_set)),
    #                                   replace=False)
    # Nah, let's pick them ourselves for now.
    validation_idxs = [ 17, 241, 287, 304, 123, 
                       458,   1,  14, 139, 388]

    if True:
        print("--- Setup visual validation.")
        # Save them for comparison.
        check_mkdir('{}/validationimgs'.format(output_directory))
        check_mkdir('{}/offsets_gt'.format(output_directory))
        check_mkdir('{}/semantic_gt'.format(output_directory))
        for validation_idx in validation_idxs:
            img_pil, _, _ = val_set.load_fit_gt_PIL_images(validation_idx)
            img, semantic_gt, offset_gt = val_set[validation_idx]
            img_pil.save("{}/validationimgs/id{:03}.png"
                         .format(output_directory, validation_idx))
            visualize_semantics(
                    img_pil, semantic_gt,
                    "{}/semantic_gt/id{:03}"
                    .format(output_directory, validation_idx))
            visualize_positionplusoffset(
                    offset_gt,
                    "{}/offsets_gt/id{:03}_mean"
                    .format(output_directory, validation_idx),
                    groundtruth=offset_gt)
            visualize_offsethsv(
                    offset_gt,
                    "{}/offsets_gt/id{:03}"
                    .format(output_directory, validation_idx))

    print("--- Training.")
    rlosses = []
    closses = []
    for epoch in range(start_epoch, n_epochs):
        model.train()
        total_rloss = 0
        total_closs = 0
        for batch_idx, batch_data in enumerate(train_loader):
            img = batch_data[0].cuda()
            semantic_gt = batch_data[1].cuda()
            instance_offset_gt = batch_data[2].cuda()
            del batch_data

            optimizer.zero_grad()
            outputs = model(img)

            batch_rloss = 0
            batch_closs = 0
            loss = 0
            closs = 0
            rloss = 0
            if regression_loss is not None:
                predicted_offset = outputs[:,-2:]

                rloss = regression_loss(predicted_offset, instance_offset_gt)

                batch_rloss += int(rloss.detach().cpu())
                total_rloss += batch_rloss

                loss += rloss

            if classification_loss is not None:
                closs = classification_loss(outputs[:,:n_classes], 
                                            semantic_gt)

                batch_closs += int(closs.detach().cpu())
                total_closs += batch_closs

                loss += closs

            loss.backward()
            optimizer.step()

            if batch_idx % 30 == 0 and batch_idx != 0:
                print('\t[batch {}/{}], [batch mean - closs {:5}, rloss {:5}]'
                      .format(batch_idx, len(train_loader), 
                              batch_closs/img.size(0),
                              batch_rloss/img.size(0)))
        del img, semantic_gt, instance_offset_gt, outputs, rloss, closs, loss
        total_closs /= len(train_set)
        total_rloss /= len(train_set)

        print('[epoch {}], [mean train - closs {:5}, rloss {:5}]'
              .format(epoch, total_closs, total_rloss))
        rlosses.append(total_rloss)
        closses.append(total_closs)
        plt.plot(np.arange(start_epoch, epoch+1), rlosses) 
        plt.savefig('{}/rlosses.svg'.format(output_directory))
        plt.close('all')
        plt.plot(np.arange(start_epoch, epoch+1), closses) 
        plt.savefig('{}/closses.svg'.format(output_directory))
        plt.close('all')
        plt.plot(np.arange(start_epoch, epoch+1), np.add(rlosses,closses))
        plt.savefig('{}/losses.svg'.format(output_directory))
        plt.close('all')
         
        # --- Visual validation.
        if (epoch % validation_step) == 0:
            # Save model parameters.
            check_mkdir('{}/models'.format(output_directory))
            torch.save(model.state_dict(),
                       '{}/models/Net_epoch{}.pth'
                       .format(output_directory, epoch))
            torch.save(optimizer.state_dict(),
                       '{}/models/Adam_epoch{}.pth'
                       .format(output_directory, epoch))

            # Visualize validation imgs.
            check_mkdir('{}/offsets'.format(output_directory))
            check_mkdir('{}/offsets/means'.format(output_directory))
            check_mkdir('{}/semantics'.format(output_directory))
            check_mkdir('{}/semantics/overlay'.format(output_directory))
            model.eval()
            for validation_idx in validation_idxs:
                img_pil, _, _ = val_set.load_PIL_images(validation_idx)
                img, _, offset_gt = val_set[validation_idx]
                img = img.unsqueeze(0).cuda()
                with torch.no_grad():
                    outputs = model(img)
                epoch_filename = 'id{:03}_epoch{:05}'\
                                 .format(validation_idx, epoch)
                if classification_loss is not None:
                    visualize_semantics(
                            img_pil, outputs,
                            "{}/semantics/{}"
                            .format(output_directory, epoch_filename),
                            "{}/semantics/overlay/{}"
                            .format(output_directory, epoch_filename))
                if regression_loss is not None:
                    visualize_offsethsv(
                            outputs.detach(),
                            "{}/offsets/{}"
                            .format(output_directory, epoch_filename))
                    visualize_positionplusoffset(
                            outputs,
                            "{}/offsets/means/{}"
                            .format(output_directory, epoch_filename),
                            groundtruth=offset_gt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="Train a segmentation and offset model.")
    parser.add_argument("--model", type=str, 
                        help="Choose a model.")
    args = parser.parse_args()

    main(args.model)
