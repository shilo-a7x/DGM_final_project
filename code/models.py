import os
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 12  # dimension of latent variable


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )
        self.fc1 = nn.Linear(784, 256)

        self.fc3 = nn.Linear(latent_dim, 256)
        self.fc4 = nn.Linear(256, 784)

    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z

    def loss(self, recon, x, mu, logvar, gmm_mu=None, gmm_var=None):
        BCE = F.binary_cross_entropy(recon, x.view(-1, 784), reduction="sum")
        if gmm_mu is None or gmm_var is None:
            # Standard VAE loss
            KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            # Non-standard normal prior
            # gmm_mu and gmm_var are the means and variances of the GMM
            prior_mu = gmm_mu.expand_as(mu).to(device)
            prior_var = gmm_var.expand_as(logvar).to(device)
            prior_logvar = prior_var.log()
            KL = -0.5 * torch.sum(
                (1 + logvar - prior_logvar - (mu - prior_mu).pow(2) / prior_var)
                - logvar.exp() / prior_var
            )
        return BCE + KL


def VAE_train(iteration, gmm_mu, gmm_var, epochs, train_loader, all_loader):
    model = VAE().to(device)
    print("VAE Training Start")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    elbo_list = np.zeros(epochs)

    # Training loop
    for i in range(epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar, z = model(data)
            if iteration == 0:  # Prior is N(0,I) in 1st round of mutual training
                loss = model.loss(recon_batch, data, mu, logvar)
            else:  # Prior is N(gmm_mu,gmm_var) in subsequent rounds of mutual training
                loss = model.loss(
                    recon_batch,
                    data,
                    mu,
                    logvar,
                    gmm_mu[batch_idx],
                    gmm_var[batch_idx],
                )
            loss = loss.mean()
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        avg_loss = train_loss / len(train_loader.dataset)
        elbo_list[i] = -(train_loss / len(train_loader.dataset))
        print(f"Epoch: {i+1} Average loss: {avg_loss:.4f}")
        elbo_list[i] = -avg_loss

        elbo_list[i] = -(train_loss / len(train_loader.dataset))

    # Save model & ELBO
    np.save(f"./../model/npy/elbo_{iteration}.npy", elbo_list)
    torch.save(model.state_dict(), f"./../model/pth/vae_{iteration}.pth")
    # Plot ELBO
    plt.figure()
    colors = ["blue", "green", "red", "purple", "orange", "cyan"]
    for past_it in range(iteration + 1):
        elbo_path = f"./../model/npy/elbo_{past_it}.npy"
        if os.path.exists(elbo_path):
            past_elbo = np.load(elbo_path)
            plt.plot(
                range(len(past_elbo)),
                past_elbo,
                label=f"Iteration {past_it}",
                color=colors[past_it % len(colors)],
            )
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend(loc="lower right")
    plt.savefig(f"./../plots/vae_elbo_{iteration}.png")
    plt.close()

    model.eval()
    with torch.no_grad():
        data, labels = next(iter(all_loader))
        data = data.to(device)
        _, _, _, z = model(data)
        z = z.cpu().numpy()
        labels = labels.numpy()
    return z, labels


def decode_from_cluster(iteration, cluster_k, sample_num):
    model = VAE().to(device)
    model.load_state_dict(
        torch.load(f"./../model/pth/vae_{iteration}.pth", weights_only=True)
    )
    model.eval()
    mu_gmm_kd, _, _ = get_param(iteration)
    sample_d = sample(
        latent_dim=latent_dim,
        mu_gmm=mu_gmm_kd,
        sample_num=sample_num,
        sample_k=cluster_k,
    )
    sample_d = torch.from_numpy(sample_d.astype(np.float32)).clone()
    with torch.no_grad():
        sample_d = model.decode(sample_d.to(device)).cpu()
        save_image(
            sample_d.view(sample_num, 1, 28, 28),
            f"./../samples/cluster_{cluster_k}.png",
        )


def GMM_train(iteration, z, label, epochs=100):
    D = len(z)
    dim = len(z[0])

    gmm = GaussianMixture(n_components=10, max_iter=epochs)
    # Fit GMM to latent space
    print("GMM Training Start")
    gmm.fit(z)

    # Get the cluster means and covariances (parameters for each cluster)
    mu_kd = gmm.means_
    sigma_kdd = gmm.covariances_  # Shape: (D, dim, dim)
    pi_k = gmm.weights_

    # Get the predicted cluster labels
    cluster_labels = gmm.predict(z)
    accuracy = calc_acc(cluster_labels, label)
    print(f"GMM Training - Mutual Iteration {iteration + 1} - Accuracy: {accuracy}")

    # Initialize variables to store the per-data-point means and variances
    mu_d = np.zeros((D, dim))
    var_d = np.zeros((D, dim))

    # For each data point, assign the mean and variance based on its cluster assignment
    for d in range(D):
        cluster_idx = cluster_labels[d]
        mu_d[d] = mu_kd[cluster_idx]
        var_d[d] = np.diag(sigma_kdd[cluster_idx])

    # Save the GMM parameters for later use
    np.save(f"./../model/npy/mu_{iteration}.npy", mu_kd)
    np.save(f"./../model/npy/sigma_{iteration}.npy", sigma_kdd)
    np.save(f"./../model/npy/pi_{iteration}.npy", pi_k)

    return torch.from_numpy(mu_d), torch.from_numpy(var_d)


def calc_acc(results, correct):
    K = np.max(results) + 1  # Number of category
    D = len(results)  # Number of data points
    max_acc = 0
    changed = True
    while changed:
        changed = False
        for i in range(K):
            for j in range(K):
                tmp_result = np.zeros(D)

                for n in range(D):
                    if results[n] == i:
                        tmp_result[n] = j
                    elif results[n] == j:
                        tmp_result[n] = i
                    else:
                        tmp_result[n] = results[n]
                acc = (tmp_result == correct).sum() / float(D)
                if acc > max_acc:
                    max_acc = acc
                    results = tmp_result
                    changed = True
    return max_acc


def sample(latent_dim, mu_gmm, sample_num, sample_k):
    # Sample close to the mean of the k-th cluster (small variance)
    sigma = 0.1 * np.identity(latent_dim, dtype=float)
    sample = np.random.multivariate_normal(
        mean=mu_gmm[sample_k], cov=sigma, size=sample_num
    )

    return sample


def get_param(iteration):
    mu_gmm_kd = np.load(f"./../model/npy/mu_{iteration}.npy")
    sigma_gmm_kdd = np.load(f"./../model/npy/sigma_{iteration}.npy")
    pi_gmm_k = np.load(f"./../model/npy/pi_{iteration}.npy")

    return mu_gmm_kd, sigma_gmm_kdd, pi_gmm_k
