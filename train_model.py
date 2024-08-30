import os
import sys
import time
import torch
from tqdm import tqdm
import copy


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10, schedular=None, model_name='',
                two_step_optimizer=False, max_no_progress=1):
    train_dir = os.path.join('results', model_name)
    since = time.time()
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = sys.float_info.max
    count_no_progress = 0
    stop = False
    for epoch in tqdm(range(num_epochs), desc='Training epochs'):
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            pbar = tqdm(dataloaders[phase], total=len(dataloaders[phase]), desc=phase)
            count = 1
            # Iterate over data.
            for inputs, labels in pbar:
                pbar.set_postfix_str(f'{phase}: Loss=>{running_loss / count}')
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if two_step_optimizer:
                            optimizer.first_step(zero_grad=True)
                            criterion(model(inputs), labels).backward()  # make sure to do a full forward pass
                            optimizer.second_step(zero_grad=True)
                        else:
                            optimizer.step()
                        if schedular is not None:
                            schedular.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, os.path.join(train_dir, f'{model_name}.pth'))
                else:
                    count_no_progress += 1
                val_loss_history.append(best_loss)
                # if schedular is not None:
                #     schedular.step()
            if count_no_progress == max_no_progress:
                print(f'stopped due to {count_no_progress} with no progress')
                stop = True
        if stop:
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history