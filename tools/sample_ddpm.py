import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from models.unet_base_test import Unet
from scheduler.linear_noise_scheduler import CosineNoiseScheduler
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.extract_mnist_images import df_resized as chunks_not_normalized_notone
import numpy as np

print('chunks_not_normalized_notone',chunks_not_normalized_notone)
chunks_not_normalized_notone.drop(chunks_not_normalized_notone.columns[0], axis=1, inplace=True)
print('chunks_not_normalized_notone',chunks_not_normalized_notone)
print('mean',np.mean(chunks_not_normalized_notone, axis=1))

def sample(model, scheduler, train_config, model_config, diffusion_config):
    r"""
    Sample stepwise by going backward one timestep at a time.
    We save the x0 predictions
    """
    df_list = []
    xt = torch.randn((train_config['num_samples'],
                      model_config['im_channels'],
                      model_config['time_horizon']
                    )).to(device)
    for i in tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # Get prediction of noise
        noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        
        # Use scheduler to get x0 and xt-1
        xt, x0_pred = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))
        
        # Save x0
        ims = torch.clamp(xt, -1., 1.).detach().cpu()

        #print('ims size',ims.shape)
        #ims = (ims + 1) / 2
        #grid = make_grid(ims, nrow=train_config['num_grid_rows'])
        #print(grid.shape)
        #img = torchvision.transforms.ToPILImage()(grid)
        

        
        if not os.path.exists(os.path.join(train_config['task_name'], 'samples')):
            os.mkdir(os.path.join(train_config['task_name'], 'samples'))
        #img.save(os.path.join(train_config['task_name'], 'samples', 'x0_{}.png'.format(i)))
        #img.close()
        # Convert tensor to DataFrame and append to list
        df = pd.DataFrame(ims.numpy().reshape(-1, model_config['im_channels'] * model_config['time_horizon'] ))
    
    print('ultimo df', df)#print('ims',ims)
    # Concatenate all DataFrames in the list
    df_all = df
    print('df_all',df_all)
    i=0
    print('df_all', df_all)
    rows, cols = df_all.shape
    df_all=pd.DataFrame(df_all)


    # Calculate mean and variance of df_all
    df_all_mean = np.mean(np.mean(df_all, axis=1),axis=0)
    df_all_variance = np.mean(np.var(df_all, axis=1),axis=0)

    # Calculate mean and variance of chunks_not_normalized
    chunks_not_normalized_mean = np.mean(np.mean(chunks_not_normalized_notone, axis=1),axis=0)
    chunks_not_normalized_variance = np.mean(np.var(chunks_not_normalized_notone, axis=1),axis=0)

    # Calculate autocorrelation of df_all
    df_all_autocorr = np.mean(df_all.apply(lambda x: x.autocorr(), axis=1),axis=0)

    # Calculate autocorrelation of chunks_not_normalized
    chunks_not_normalized_autocorr = np.mean(pd.DataFrame(chunks_not_normalized_notone).apply(lambda x: x.autocorr(), axis=1),axis=0)
    def generative_score(df_all, chunks_not_normalized_notone):
        # Calculate mean and variance of df_all
        df_all_mean = np.mean(df_all, axis=1)
        df_all_variance = np.var(df_all, axis=1)
        
        # Calculate mean and variance of chunks_not_normalized
        chunks_not_normalized_mean = np.mean(chunks_not_normalized_notone, axis=1)
        chunks_not_normalized_variance = np.var(chunks_not_normalized_notone, axis=1)
        
        # Calculate generative score
        generative_score = np.mean(np.abs(df_all_mean - chunks_not_normalized_mean)) + np.mean(np.abs(df_all_variance - chunks_not_normalized_variance))
        
        return generative_score

    generative_score = generative_score(df_all, chunks_not_normalized_notone)
    print("Generative Score:", generative_score)
    print("Autocorrelation of generated:", df_all_autocorr)
    print("Autocorrelation of original:", chunks_not_normalized_autocorr)
    print("Mean of generated:", df_all_mean)
    print("Mean of original:", chunks_not_normalized_mean)
    print("Variance of generated:", df_all_variance)
    print("Variance of original:", chunks_not_normalized_variance)

    df_all.to_excel('data/samples.xlsx', index=False)
    plt.title('Generated Sample 1')
    plt.plot(df_all.iloc[0])
    plt.show()
    plt.title('Generated Sample 2')
    plt.plot(df_all.iloc[1])
    plt.show()
    plt.title('Generated Sample 3')
    plt.plot(df_all.iloc[2])
    plt.show()
    plt.title('Generated Sample 4')
    plt.plot(df_all.iloc[3])
    plt.show()
    plt.title('Generated Sample 5')
    plt.plot(df_all.iloc[4])
    plt.show()
    plt.title('Generated Sample 6')
    plt.plot(df_all.iloc[5])
    plt.show()
    plt.title('Generated Sample 7')
    plt.plot(df_all.iloc[6])
    plt.show()
    plt.title('Generated Sample 8')
    plt.plot(df_all.iloc[7])
    plt.show()
    plt.title('Generated Sample 9')
    plt.plot(df_all.iloc[8])
    plt.show()
    plt.title('Generated Sample 10')
    plt.plot(df_all.iloc[9])
    plt.show()
    plt.title('Generated Sample 11')
    plt.plot(df_all.iloc[10])
    plt.show()
    
    

    pca = PCA(n_components=2)
    pca_original = pca.fit_transform(chunks_not_normalized_notone)
    
    pca_generated=pca.fit_transform(df_all)
    plt.figure(figsize=(6, 5))
    plt.scatter(pca_original[:, 0], pca_original[:, 1], label='Original')
    plt.scatter(pca_generated[:, 0], pca_generated[:, 1], label='Generated')
    plt.title('PCA plot')
    plt.legend()
    plt.show()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(chunks_not_normalized_notone)
    tsne_results_generated = tsne.fit_transform(df_all)
    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label='Original')
    plt.scatter(tsne_results_generated[:, 0], tsne_results_generated[:, 1], label='Generated')
    plt.title('t-SNE plot')
    plt.legend()
    plt.show()

    



def infer(args):
    # Read the config file #
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    #print(config)
    ########################
    
    diffusion_config = config['diffusion_params']
    model_config = config['model_params']
    train_config = config['train_params']
    
    # Load model with checkpoint
    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(os.path.join(train_config['task_name'],
                                                  train_config['ckpt_name']), map_location=device))
    model.eval()
    
    # Create the noise scheduler
    scheduler = CosineNoiseScheduler(num_timesteps=diffusion_config['num_timesteps'],
                                     beta_start=diffusion_config['beta_start'],
                                     beta_end=diffusion_config['beta_end'])
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ddpm image generation')
    parser.add_argument('--config', dest='config_path',
                        default='config/default.yaml', type=str)
    args = parser.parse_args()
    infer(args)