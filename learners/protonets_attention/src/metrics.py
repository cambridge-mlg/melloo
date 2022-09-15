import utils

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn.metrics
from sklearn.manifold import TSNE
from scipy.stats import kendalltau
import numpy as np
import torch
import os


class Metrics:

    def __init__(self, checkpoint_dir, way, top_k, logger):
        self.checkpoint_dir = checkpoint_dir
        self.way = way
        self.top_k = top_k
        self.logger = logger
        
        
    def save_image_set(self, task_num, images, descrip, labels=None):
        if task_num != -1:
            path = os.path.join(self.checkpoint_dir, str(task_num))
        else:
            path = self.checkpoint_dir
        if not os.path.exists(path):
            os.makedirs(path)
        tmp_images = images.cpu().detach().numpy()
        if not labels is None:
            assert len(labels) == len(images)

        for i in range(len(tmp_images)):
            if labels is None:
                image_path = os.path.join(path, '{}_index_{}.png'.format(descrip, i))
            else:
                image_path = os.path.join(path, '{}_index_{}_label_{}.png'.format(descrip, i, labels[i]))
            utils.save_image(tmp_images[i], image_path)

    def plot_scatter(self, x, y, x_label, y_label, plot_title, output_name, 
        class_labels=None, color=None, split_by_class_label=False, x_min=None, x_max=None, y_min=None, y_max=None):
        class_colors = ["#dc0f87", "#e8b90e", "#29e414", "#f76b1f", "#585d9c"]
        x = utils.convert_to_numpy(x)
        y = utils.convert_to_numpy(y)
        if class_labels != None:
            t_color = [class_colors[lbl] for lbl in class_labels]
            num_classes_local = len(class_labels.unique())
        else:
            assert not split_by_class_label
            assert color != None
            t_color = color
            num_classes_local = self.way

        if split_by_class_label:
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            for tc in range(0, num_classes_local):
                class_mask = (class_labels == tc).cpu().numpy()
                plot_scatter(x[class_mask], y[class_mask], x_label, y_label, plot_title + "_{}".format(tc), output_name + "_{}".format(tc), color = class_colors[tc], 
                    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, split_by_class_label=False)

        handles = [mpatches.Rectangle((0, 0), 1, 1, fc=ccol) for ccol in class_colors ]
        plt.scatter(x, y, c=t_color)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(plot_title)
        plt.grid(True)
        plt.legend(handles, ['Class {}'.format(lbl) for lbl in range(num_classes_local)])
        plt.savefig(os.path.join(self.checkpoint_dir, output_name + ".png"))
        plt.close()
        
    def print_and_log_metric(self, values, item, metric_name="Accuracy"):
        metric = np.array(values).mean() * 100.0
        metric_confidence = (196.0 * np.array(values).std()) / np.sqrt(len(values))

        self.logger.print_and_log('{} {}: {:3.1f}+/-{:2.1f}'.format(metric_name, item, metric, metric_confidence))
    
        
    def compare_rankings(self, series1, series2, descriptor="", weights=False):
        if weights:
            ranking1 = torch.argsort(series1, dim=1, descending=True)
            ranking2 = torch.argsort(series2, dim=1, descending=True)
        else:
            ranking1, ranking2 = series1, series2
            
        ave_corr = 0.0
        ave_intersected = 0.0
        
        for t in range(ranking1.shape[0]):
            corr, p_value = kendalltau(ranking1[t], ranking2[t])
            ave_corr = ave_corr + corr
            
            top_k_1 = set(np.array(ranking1[t][1:self.top_k]))
            top_k_2 = set(np.array(ranking2[t][1:self.top_k]))
            intersected = sorted(top_k_1 & top_k_2)
            ave_intersected =  ave_intersected + len(intersected)
            
        ave_corr = ave_corr/ranking1.shape[0]
        ave_intersected = ave_intersected/ranking1.shape[0]
        
        #self.logger.print_and_log("Ave num intersected {}: {}".format(descriptor, ave_intersected))
        #self.logger.print_and_log("Ave corr {}: {}".format(descriptor, ave_corr))
        
        return ave_corr, ave_intersected
        
    def plot_confusion_matrix(self, true_labels, filename, logits=None, pred_labels=None):
        labels = true_labels.cpu()
        if logits is not None:
            preds = logits.argmax(axis=1).cpu()
        else:
            preds = pred_labels.cpu()
        cm = sklearn.metrics.confusion_matrix(labels, preds)
        cm = cm.transpose()
        plt.matshow(cm.transpose())
        for (x, y), value in np.ndenumerate(cm):
            plt.text(x, y, f"{value:d}", va="center", ha="center")
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()
            
    def plot_and_log(self, vals, descrip, filename):
        self.logger.log(descrip)
        self.logger.log("{}".format(vals))
        plt.plot(vals)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()

    def plot_with_error(self, vals, errs,  descrip, filename):
        self.logger.log(descrip)
        self.logger.log("{}".format(vals))
        plt.errorbar(x=range(len(vals)),y=vals, yerr=errs)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()

        
    def bar_plot_and_log(self, keys, vals, descrip, filename):
        self.logger.log(descrip)
        for key, val in zip(keys, vals):
            self.logger.log("{} : {}".format(key, val))
        plt.bar(keys, vals)
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()
    

    def plot_hist(self, x, bins, filename, task_num=None, title='', x_label='', y_label='', density=False):
        x = utils.convert_to_numpy(x)
        plt.hist(x, bins=bins, density=density)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.grid(True)
        if task_num != None:
            filename = "{}_".format(task_num) + filename
        filename += ".png"
        plt.savefig(os.path.join(self.checkpoint_dir, filename))
        plt.close()


    def plot_tsne(self, points, labels, prototypes, descrip):
        class_colors = ["#dc0f87", "#e8b90e", "#29e414", "#f76b1f", "#585d9c", "#323ca8", "#ff0303", "#03ffc4", "#afff03", "#a566e3"]
        prototypes = utils.convert_to_numpy(prototypes)
        prototype_labels = utils.convert_to_numpy(torch.unique(labels)) # Need the ordering of torch.unique
        points = utils.convert_to_numpy(points)
        labels = utils.convert_to_numpy(labels)
        points = np.concatenate([points, prototypes], axis=0)
        labels = np.concatenate([labels, prototype_labels], axis=0)
        perplexities = [30] #[2, 5, 10, 20, 30, 50, 75, 100]
        learning_rates = [50] #[10.0, 25, 50, 100, 200, 500]
        for r in learning_rates:
            for p in perplexities:
                p_embedded = TSNE(n_components=2, init='random', n_iter=1000, learning_rate=r).fit_transform(points)
                fig = plt.figure()
                ax = fig.add_subplot(111)
                for class_label in range(self.way):
                    class_mask = labels == class_label
                    # Plot regular points
                    ax.scatter(p_embedded[class_mask][:-1, 0],p_embedded[class_mask][:-1, 1], label='Class {}'.format(class_label), c=class_colors[class_label])
                    # Plot prototypes
                    ax.scatter(p_embedded[class_mask][-1:, 0],p_embedded[class_mask][-1:, 1], c=class_colors[class_label], marker='s')

                plt.legend()
                plt.savefig(os.path.join(self.checkpoint_dir, "tsne_" + descrip.replace(":", "-") + "_perp_{}_lr_{}.png".format(p, r)))
                plt.close()
