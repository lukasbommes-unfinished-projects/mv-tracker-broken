import os
import time
import copy
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from lib.model import PropagationNetwork
from lib.dataset.dataset_precomputed import MotionVectorDatasetPrecomputed


def train(model, criterion, optimizer, scheduler, num_epochs=2):
    tstart = time.time()
    writer = SummaryWriter()

    best_model_wts = copy.deepcopy(model.state_dict())
    pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))
    best_loss = 99999.0
    iterations = 0

    for epoch in range(num_epochs):

        # get current learning rate
        learning_rate = 0
        for param_group in optimizer.param_groups:
            learning_rate = param_group['lr']

        print("Epoch {}/{} - Learning rate: {}".format(epoch, num_epochs-1, learning_rate))

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            pbar = tqdm(total=len(dataloaders[phase]))
            for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask, motion_vector_scale) in enumerate(dataloaders[phase]):

                # remove batch dimension as precomputed data is already batched
                motion_vectors.squeeze_(0)
                boxes_prev.squeeze_(0)
                velocities.squeeze_(0)
                num_boxes_mask.squeeze_(0)
                motion_vector_scale.squeeze_(0)

                # move to GPU
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                velocities = velocities.to(device)
                num_boxes_mask = num_boxes_mask.to(device)
                motion_vector_scale = motion_vector_scale.to(device)

                velocities = velocities[num_boxes_mask]
                velocities = velocities.view(-1, 4)

                # normalize velocities to range [0, 1]
                #vel_min = torch.min(velocities)
                #vel_max = torch.max(velocities)
                #velocities = (velocities - vel_min) / (vel_max - vel_min)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    velocities_pred = model(motion_vectors, boxes_prev, num_boxes_mask, motion_vector_scale)

                    #print(step)

                    #if step % 1000 == 0:
                    #    print(velocities)
                    #    print(velocities_pred)

                    loss = criterion(velocities_pred, velocities)

                    if phase == "train":
                        params_before_update = list(model.parameters())[0].clone()
                        loss.backward()
                        optimizer.step()
                        params_after_update = list(model.parameters())[0].clone()

                        # check if model parameters are still being updated
                        if torch.equal(params_before_update.data, params_after_update.data):
                            raise RuntimeError("The model stopped learning. Parameters are not getting updated anymore.")

                    pbar.update()

                running_loss += loss.item() * motion_vectors.size(0)
                writer.add_scalar('Running Loss/{}'.format(phase), running_loss, iterations)
                iterations += 1

            pbar.close()

            epoch_loss = running_loss / len(datasets[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            writer.add_scalar('Epoch Loss/{}'.format(phase), epoch_loss, epoch)

            if phase == "val":
                model_wts = copy.deepcopy(model.state_dict())
                pickle.dump(best_model_wts, open("models/model_{:04d}.pkl".format(epoch), "wb"))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))

        scheduler.step() # after each train epoch

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    writer.close()
    return model


if __name__ == "__main__":
    root_dir = "data_precomputed"
    modes = ["train", "val"]
    datasets = {x: MotionVectorDatasetPrecomputed(root_dir=os.path.join(root_dir, x)) for x in modes}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=1, shuffle=False, num_workers=8) for x in modes}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PropagationNetwork()
    model = model.to(device)

    #criterion = nn.SmoothL1Loss(reduction='mean')
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0001, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    best_model = train(model, criterion, optimizer, scheduler, num_epochs=300)
