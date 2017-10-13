import matplotlib
from matplotlib import pyplot as plt

def close_all_plots():
    plt.close("all")

def plot_data(data, rates, truth=None, cos_sample=None, cos_mean=None):
    fig, axes = plt.subplots(4, 1) 
    axes = axes.flatten()
    axes[0].imshow(data.T)
    axes[0].set_aspect('auto')
    axes[1].imshow(rates.T)
    axes[1].set_aspect('auto')
    if truth is not None:
        axes[2].imshow(truth.T)
        axes[2].set_aspect('auto')
    if cos_sample is not None:
        axes[3].plot(cos_sample)
        axes[3].set_aspect('auto')
    
    if cos_mean is not None:
        axes[3].plot(cos_mean, 'r-')
        axes[3].set_aspect('auto')
    
    plt.ion()
    plt.show()
    plt.draw()
    return plt
