from .planner import Planner, save_model 
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_data
from . import dense_transforms

def train(args):
    from os import path
    model = Planner()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    """
    Your code here, modify your HW4 code
    Hint: Use the log function below to debug and visualize your model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.MSELoss()

    train_dataset = load_data('drive_data', transform=dense_transforms.Compose([dense_transforms.RandomHorizontalFlip(), \
                                dense_transforms.ColorJitter(0.9, 0.9, 0.9, 0.1), \
                                dense_transforms.ToTensor()]), batch_size=32)

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0.0

        for image, aim_pt in train_dataset:
            pred = model(image)
            aim = torch.tensor(aim_pt)
            loss = criterion(pred, aim)
            total_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step()
        
        avg_loss = total_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{args.num_epochs}, train loss: {avg_loss}")
        log(logger=train_logger, img=image, label=aim, pred=pred, global_step=epoch)

    save_model(model)

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='')
    parser.add_argument('--num_epochs', type=int, default=100)
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
