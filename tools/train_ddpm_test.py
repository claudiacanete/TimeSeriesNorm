import torch
import yaml
import argparse
import os
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from dataset.mnist_dataset import MnistDataset
from torch.utils.data import DataLoader
from models.unet_base_test import Unet
from scheduler.linear_noise_scheduler import CosineNoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    dataset_config = config['dataset_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Create the noise scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    
    # Create the dataset
    mnist = MnistDataset('train', im_path=dataset_config['im_path'])
    mnist_loader = DataLoader(mnist, batch_size=train_config['batch_size'], shuffle=True, num_workers=4)
    
    # Instantiate the model
    model = Unet(model_config).to(device)




    
    # Create output directories
    if not os.path.exists(train_config['task_name']):
        os.mkdir(train_config['task_name'])
    
    # Load checkpoint if found
    #if os.path.exists(os.path.join(train_config['task_name'],train_config['ckpt_name'])):
    #    print('Loading checkpoint as found one')
    #    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
    #                                                  train_config['ckpt_name']), map_location=device))
        
    # Specify training parameters
    num_epochs = train_config['num_epochs']
    optimizer = Adam(model.parameters(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss()
    
    # Run training
    
    #for im in tqdm(mnist_loader):
    #    im = torch.reshape(im, (im.shape[0], im.shape[1], im.shape[2]*im.shape[3]))
    

    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(mnist_loader):
            optimizer.zero_grad()
            #print('im size', im.shape)# [16,1,1,256]
            im = im.float().to(device)
            
            # Sample random noise
            noise = torch.randn_like(im).to(device)
            #print('noise', noise.shape)
            
            # Sample timestep
            t = torch.randint(0, diffusion_config['num_timesteps'], (im.shape[0],)).to(device)
            
            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            #print('noisy_im',noisy_im.shape)
            
            #noisy_im = torch.reshape(noisy_im, (noisy_im.shape[0], noisy_im.shape[1], noisy_im.shape[2]*im.shape[3]))
            noisy_im = torch.reshape(noisy_im, (noisy_im.shape[0], noisy_im.shape[1], noisy_im.shape[2]*noisy_im.shape[3])).float()
            noise = torch.reshape(noise, (noise.shape[0], noise.shape[1], noise.shape[2]*noise.shape[3])).float()
            noise_pred = model(noisy_im, t)
            
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        print('Finished epoch:{} | Loss : {:.4f}'.format(
            epoch_idx + 1,
            np.mean(losses),
        ))
        torch.save(model.state_dict(), os.path.join(train_config['task_name'],
                                                    train_config['ckpt_name']))
    
    print('Done Training ...')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm training')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    train(args)
