from matplotlib import pyplot as plt
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
import torch
import torch.nn.functional as F
import numpy as np

class PlotSettings:
    def __init__(self, sym_bounds=10):
        self.sym_bound = sym_bounds
        self.resolution = 100
        self.clean_colors = ['navy', 'darkred', 'darkgreen', 'gold', 'darkviolet', 'c', 'm']
        self.colors = ['b', 'r', 'g', 'orange', 'purple', 'cyan', 'm']
        self.color_maps = []
        self.edge_colors = ['k', 'k', 'k', 'k', 'k', 'k', 'k']
        self.markers = ['v', '^', '<', '>', 'd']
        self.target_markers = ['1', '2', '3', '4', '+']

        def_color_maps = [pl.cm.Blues, pl.cm.Reds, pl.cm.Greens, pl.cm.Oranges, pl.cm.Purples]
        # Get the colormap colors
        for cmap in def_color_maps:
            my_cmap = cmap(np.arange(cmap.N))
            # Set alpha
            my_cmap[:, -1] = np.linspace(0, 1, cmap.N)
            # Create new colormap
            my_cmap = ListedColormap(my_cmap)
            self.color_maps.append(my_cmap)
            

def plot_decision_regions(centroids, points, labels, out_name, plot_conf, model, device):
    num_classes = len(centroids)
    assert num_classes <= len(plot_conf.colors)

    # Make custom colormaps with the transparency we need
    # No, I don't know how I got here

    xx, yy = np.meshgrid(
        np.linspace(-plot_conf.sym_bound, plot_conf.sym_bound, plot_conf.resolution),  # np.geomspace(-10, 10, resolution)
        np.linspace(-plot_conf.sym_bound, plot_conf.sym_bound, plot_conf.resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.from_numpy(grid_points).type(torch.FloatTensor).to(device)

    # All the points in the grid need to have an associated probability for each class
    # to generate the colours
    grid_logits = F.softmax(model.predict(grid_tensor).cpu(), dim=1)

    # normalize within each row
    grid_pred = torch.argmax(grid_logits, dim=-1)
    grid_conf = torch.max(grid_logits, dim=1)[0]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(16, 16)
    plt.ylim(-plot_conf.sym_bound, plot_conf.sym_bound)
    plt.xlim(-plot_conf.sym_bound, plot_conf.sym_bound)
    centroids_np = centroids.cpu().numpy()
    for c in range(num_classes):
        grid_c = (grid_pred == c).type(torch.DoubleTensor)
        height = (grid_c * grid_conf).reshape(plot_conf.resolution, plot_conf.resolution)
        ax.contourf(xx, yy, height, cmap=plot_conf.color_maps[c])

    for c in range(num_classes):
        mask_c = (labels == c)
        plt.scatter(*zip(*(points[mask_c].cpu().numpy())), marker='.', c=plot_conf.colors[c],
                    alpha=0.8, edgecolors=plot_conf.edge_colors[c], s=150)
        
        plt.scatter(centroids_np[c][0], centroids_np[c][1], marker='o', c=plot_conf.colors[c], edgecolors=plot_conf.edge_colors[c], s=250, alpha=0.8)

    plt.savefig(out_name)
    plt.close()
