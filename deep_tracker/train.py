import time
import copy
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from lib.model import PropagationNetwork
from lib.datasets import MotionVectorDataset


def train(model, criterion, optimizer, scheduler, num_epochs=2):
    tstart = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))
    best_loss = 99999.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            pbar = tqdm(total=len(dataloaders[phase]))
            for step, (motion_vectors, boxes_prev, velocities, num_boxes_mask) in enumerate(dataloaders[phase]):
                motion_vectors = motion_vectors.to(device)
                boxes_prev = boxes_prev.to(device)
                velocities = velocities.to(device)
                num_boxes_mask = num_boxes_mask.to(device)

                velocities = velocities[num_boxes_mask]
                velocities = velocities.view(-1, 4)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    velocities_pred = model(motion_vectors, boxes_prev, num_boxes_mask)

                    #print(step)

                    if step % 1000 == 0:
                        print(velocities)
                        print(velocities_pred)

                    loss = criterion(velocities_pred, velocities)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    pbar.update()

                running_loss += loss.item() * motion_vectors.size(0)

            pbar.close()

            epoch_loss = running_loss / len(datasets[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == "val":
                model_wts = copy.deepcopy(model.state_dict())
                pickle.dump(best_model_wts, open("models/model_{:04d}.pkl".format(epoch), "wb"))

            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                pickle.dump(best_model_wts, open("models/best_model.pkl", "wb"))

    time_elapsed = time.time() - tstart
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Lowest validation loss: {:4f}'.format(best_loss))

    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    datasets = {x: MotionVectorDataset(root_dir='data', window_length=1, codec="mpeg4", visu=False, mode=x) for x in ["train", "val", "test"]}
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4) for x in ["train", "val", "test"]}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PropagationNetwork()
    model = model.to(device)

    criterion = nn.SmoothL1Loss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    best_model = train(model, criterion, optimizer, scheduler, num_epochs=60)
