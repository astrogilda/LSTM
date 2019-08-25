# In [0]:
# import packages
import numpy as np

import torch
from torch import nn
from torch import distributions
from torch.nn.parameter import Parameter
from torch.autograd import Variable

from astropy.io import fits


#========================================================================================================
# In [1]:
# import training set
# restore data
hdulist = fits.open('../Catalog_Apogee_Payne.fits.gz')
Teff = hdulist[1].data["Teff"]
Logg = hdulist[1].data["Logg"]
FeH = hdulist[1].data["FeH"]

y_tr = np.vstack([Teff,Logg,FeH]).T
print(y_tr.shape)

# convert into torch
y_tr = torch.from_numpy(y_tr).type(torch.cuda.FloatTensor)

# input dimension
dim_in = y_tr.shape[-1]

# standardize
mu_y = y_tr.mean(dim=0)
std_y = y_tr.std(dim=0)
y_tr = (y_tr - mu_y) / std_y


#=======================================================================================================
# In [2]:
# define normalizing flow
class RealNVP(nn.Module):
    def __init__(self, nets, nett, mask, prior):
        super(RealNVP, self).__init__()

        self.prior = prior
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([nett() for _ in range(len(masks))])
        self.s = torch.nn.ModuleList([nets() for _ in range(len(masks))])

    def g(self, z):
        x = z
        for i in range(len(self.t)):
            x_ = x*self.mask[i]
            s = self.s[i](x_)*(1 - self.mask[i])
            t = self.t[i](x_)*(1 - self.mask[i])
            x = x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)
        return x

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in reversed(range(len(self.t))):
            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1-self.mask[i])
            t = self.t[i](z_) * (1-self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        return z, log_det_J

    def log_prob(self,x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        x = self.g(z)
        return x


#=======================================================================================================
# In [3]:
# the latent dimension
# the latent dimension
device = torch.device("cuda")
num_neurons = 10

nets = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()
nett = lambda: nn.Sequential(nn.Linear(dim_in, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, num_neurons), nn.LeakyReLU(),\
                             nn.Linear(num_neurons, dim_in)).cuda()

num_layers = 3
masks = []
for i in range(num_layers):
    mask_layer = np.random.randint(2,size=(dim_in))
    masks.append(mask_layer)
    masks.append(1-mask_layer)
masks = torch.from_numpy(np.array(masks).astype(np.float32))
masks.to(device)
prior = distributions.MultivariateNormal(torch.zeros(dim_in, device='cuda'),\
                                         torch.eye(dim_in, device='cuda'))

flow = RealNVP(nets, nett, masks, prior)
low.cuda()



#=======================================================================================================
# In [4]
# optimizing flow models
optimizer = torch.optim.Adam([p for p in flow.parameters() if p.requires_grad==True], lr=1e-4)
num_epoch = 1001

for t in range(num_epoch):
    loss = -flow.log_prob(y_tr).mean()

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if t % 50 == 0:
        print('iter %s:' % t, 'loss = %.3f' % loss)


#========================================================================================================
# sample results
z1 = flow.f(y_tr)[0].detach().cpu().numpy()
x1 = y_tr.cpu().numpy()
z2 = np.random.multivariate_normal(np.zeros(dim_in), np.eye(dim_in), x1.shape[0])
x2 = flow.sample(x1.shape[0]).detach().cpu().numpy()

# rescale the results
x1 = x1*std_y + mu_y
x2 = x2*std_y + mu_y

# save results
np.savez("real_nvp_results.npz",\
         z1 = z1,\
         z2 = z2,\
         x1 = x1,\
         x2 = x2)
