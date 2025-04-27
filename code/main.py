import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from models import sample, get_param, latent_dim
from scipy.stats import multivariate_normal
import numpy as np
import os
import models

TRAIN_FLAG = True
mutual_iteration = 2
VAE_interations = 50
GMM_interations = 100
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./../model/npy", exist_ok=True)
os.makedirs("./../model/pth", exist_ok=True)
os.makedirs("./../plots", exist_ok=True)
os.makedirs("./../samples", exist_ok=True)


full_train_dataset = datasets.MNIST(
    "./../data", train=True, transform=transforms.ToTensor(), download=True
)

# Subsample 10,000 samples
# subset_indices = torch.randperm(len(full_train_dataset))[:10000]

train_size = int(len(full_train_dataset) * 0.8)
subset_indices = list(range(0, train_size))
train_dataset = torch.utils.data.Subset(full_train_dataset, subset_indices)

train_size = len(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=False)
# Create a DataLoader for the entire dataset
all_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_size, shuffle=False
)


def train_model(train_loader, all_loader):
    for it in range(mutual_iteration):
        print(f"Round {it + 1}/{mutual_iteration} of mutual training")
        if it == 0:
            gmm_mu = None
            gmm_var = None

        # VAE training
        z, label = models.VAE_train(
            iteration=it,
            gmm_mu=gmm_mu,
            gmm_var=gmm_var,
            epochs=VAE_interations,
            train_loader=train_loader,
            all_loader=all_loader,
        )

        # GMM training
        gmm_mu, gmm_var = models.GMM_train(
            iteration=it,
            z=z,
            label=label,
            epochs=GMM_interations,
        )


def plot_latent(iteration, all_loader):
    print("Plot latent space")
    # Load trained VAE model
    model = models.VAE().to(device)
    model.load_state_dict(
        torch.load(f"./../model/pth/vae_{iteration}.pth", weights_only=True)
    )
    model.eval()

    data, labels = next(iter(all_loader))
    data = data.to(device)

    with torch.no_grad():
        _, _, _, z = model(data)

    z = z.cpu().numpy()
    labels = np.array(labels)

    # Reduce dimensionality to 2D using t-SNE
    z_2d = TSNE(n_components=2, random_state=0).fit_transform(z)

    # Assign a color to each unique label
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    color_map = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 10))
    plt.title("Latent space on VAE")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tick_params(labelsize=14)

    # Plot each class with a different color
    for label in unique_labels:
        idx = labels == label
        plt.scatter(
            z_2d[idx, 0], z_2d[idx, 1], c=[color_map[label]], label=str(label), s=100
        )

    plt.legend(title="Labels", fontsize=12, title_fontsize=14)
    plt.savefig(f"./../plots/z_{iteration}.png")
    plt.close()


def visualize_gmm(iteration, decode_k, sample_num):
    mu_gmm_kd, sigma_gmm_kdd, pi_gmm_k = get_param(iteration=iteration)

    manual_sample = sample(
        latent_dim=latent_dim,
        mu_gmm=mu_gmm_kd,
        sample_k=decode_k,
        sample_num=sample_num,
    )

    num_clusters = mu_gmm_kd.shape[0]
    mu_gmm2d_kd = np.zeros((num_clusters, 2))  # mu 2D
    sigma_gmm2d_kdd = np.zeros((num_clusters, 2, 2))  # sigma 2D

    for k in range(num_clusters):
        mu_gmm2d_kd[k] = mu_gmm_kd[k][:2]
        sigma_gmm2d_kdd[k] = sigma_gmm_kdd[k][:2, :2]

    std_x = np.sqrt(sigma_gmm2d_kdd[:, 0, 0])
    std_y = np.sqrt(sigma_gmm2d_kdd[:, 1, 1])

    x_1_line = np.linspace(
        np.min(mu_gmm2d_kd[:, 0] - 3 * std_x),
        np.max(mu_gmm2d_kd[:, 0] + 3 * std_x),
        num=900,
    )
    x_2_line = np.linspace(
        np.min(mu_gmm2d_kd[:, 1] - 3 * std_y),
        np.max(mu_gmm2d_kd[:, 1] + 3 * std_y),
        num=900,
    )

    x_1_grid, x_2_grid = np.meshgrid(x_1_line, x_2_line)
    x_point = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
    x_dim = x_1_grid.shape

    # Compute the density of the selected component (decode_k)
    res_density_k = (
        multivariate_normal.pdf(
            x=x_point, mean=mu_gmm2d_kd[decode_k], cov=sigma_gmm2d_kdd[decode_k]
        )
        * pi_gmm_k[decode_k]
    )

    plt.figure(figsize=(12, 9))
    plt.scatter(x=manual_sample[:, 0], y=manual_sample[:, 1])
    plt.scatter(
        x=mu_gmm2d_kd[:, 0],
        y=mu_gmm2d_kd[:, 1],
        color="red",
        s=100,
        marker="x",
        label="Means",
    )
    contour = plt.contour(
        x_1_grid, x_2_grid, res_density_k.reshape(x_dim), alpha=0.5, linestyles="dashed"
    )
    plt.clabel(contour, inline=1, fontsize=10)
    plt.suptitle("Gaussian Mixture Model", fontsize=20)
    plt.title(f"Number of samples = {len(manual_sample)}, K = {decode_k}")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.colorbar(contour, label="Density")
    plt.legend()

    # Make the chosen component mean the center
    center = mu_gmm2d_kd[decode_k]

    plt.xlim(center[0] - 1, center[0] + 1)
    plt.ylim(center[1] - 1, center[1] + 1)

    # Stack all relevant points
    all_points = np.vstack([mu_gmm2d_kd, manual_sample[:, :2]])

    # Compute max distance in x and y from the center
    x_dists = np.abs(all_points[:, 0] - center[0])
    y_dists = np.abs(all_points[:, 1] - center[1])

    # Set a symmetric view around the selected mean
    x_radius = np.max(x_dists) * 1.3  # 1.3 adds padding
    y_radius = np.max(y_dists) * 1.3

    plt.xlim(center[0] - x_radius, center[0] + x_radius)
    plt.ylim(center[1] - y_radius, center[1] + y_radius)

    plt.savefig(f"./../plots/gaussian_mean_sample.png")
    plt.show()
    plt.close()


def main():
    if TRAIN_FLAG:
        train_model(train_loader=train_loader, all_loader=all_loader)

    plot_latent(iteration=mutual_iteration - 1, all_loader=all_loader)
    print("Reconstruct images")
    for i in range(10):
        models.decode_from_cluster(
            iteration=mutual_iteration - 1,
            cluster_k=i,
            sample_num=16,
        )

    visualize_gmm(iteration=mutual_iteration - 1, decode_k=5, sample_num=16)


if __name__ == "__main__":
    main()
