import torch, torchvision, os, PIL, pdb
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d


def show(tensor, num=25, wandb=0, name=''):
  data = tensor.detach().cpu()
  grid = make_grid(data[:num], nrow=5).permute(1,2,0)

  plt.imshow(grid.clip(0,1))
  plt.show()

# hyperparam and general param
n_epochs = 10000
batch_size = 128
lr = 1e-4
z_dim = 200
device = 'cuda' 

cur_step = 0
crit_cycles = 5
gen_losses = []
crit_losses = []
show_step = 35
save_step = 35


# LOAD CELBA DATA SET
dir_path='data/celba'

if not os.path.exists(dir_path):
  os.makedirs(dir_path)
 
with zipfile.ZipFile(path, 'r') as ziphandler:
  ziphandler.extractall(dir_path)

class Dataset(Dataset):
  def __init__(self, path, size=128, lim=10000):
    self.sizes=[size, size]
    items, labels = [], []

    for data in os.listdir(path)[:lim]:
      # path ./data/celeba/img_align_celeba/img_align_celeba
      # data: 39292.jpg

      item = os.path.join(path, data)
      items.append(item)
      labels.append(data)

    self.items = items
    self.labels = labels

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    data = PIL.Image.open(self.items[idx]).convert('RGB') # will return a data struc w/ (178x218), will be size of image. want to resize to 128x128
    data = np.asarray(torchvision.transforms.Resize(self.sizes)(data)) # return 128 x 128 w/ 3 channels
    data = np.transpose(data, (2,0,1)).astype(np.float32, copy=False) # 3 channcels X 128 X 128, go from 0-255, typical rgb 
    data = torch.from_numpy(data).div(255) # divide by 255, to put everything into 0-1 for the network

    return data, self.labels[idx]


#GENERATOR
class Generator(nn.Module):
  def __init__(self, z_dim=64, d_dim=16):
    super(Generator, self).__init__()
    self.z_dim=z_dim

    self.gen = nn.Sequential(
        # nn.ConvTranspose2d: in_channels, out_channels, kernel_size, stride=1, padding=0
        # new widht + height: (n-1)*stride - 2*padding + ks
        # n = widht or height
        # ks = kernel size
        #begin with 1x1 image with z_dim number of channels (200)
        nn.ConvTranspose2d(z_dim, d_dim*32, 4, 1, 0), #size 4x4, channhes 200,512
        nn.BatchNorm2d(d_dim*32),
        nn.ReLU(True),

        nn.ConvTranspose2d(d_dim*32, d_dim*16, 4, 2, 1), # 8x8, 512, 256
        nn.BatchNorm2d(d_dim*16),
        nn.ReLU(True),

        nn.ConvTranspose2d(d_dim*16, d_dim*8, 4, 2, 1), # 16x16, 256->128
        nn.BatchNorm2d(d_dim*8),
        nn.ReLU(True),

        nn.ConvTranspose2d(d_dim*8, d_dim*4, 4, 2, 1), # 32x32, 128->64
        nn.BatchNorm2d(d_dim*4),
        nn.ReLU(True),

        nn.ConvTranspose2d(d_dim*4, d_dim*2, 4, 2, 1), # 64x64, 64->32
        nn.BatchNorm2d(d_dim*2),
        nn.ReLU(True),

        nn.ConvTranspose2d(d_dim*2, 3, 4, 2, 1), # 128x128, 32->3
        nn.Tanh() # will produce result in the range from -1 to 1
    )

  def forward(self, noise):
    x = noise.view(len(noise), self.z_dim, 1, 1) # size of batch 128, size of channels 200, input width, height
    return self.gen(x)

def gen_noise(num, z_dim, device='cuda'):
  return torch.rand(num, z_dim, device=device)

# CRITIC
class Critic(nn.Module):
  def __init__(self, d_dim=16):
    super(Critic, self).__init__()

    self.crit = nn.Sequential(
        nn.Conv2d(3, d_dim, 4, 2, 1), #initial size 128x128 -> 64x64, 3, 16
        nn.InstanceNorm2d(d_dim),
        nn.LeakyReLU(0.2),
        
        nn.Conv2d(d_dim, d_dim*2, 4, 2, 1), # 32x32 (16->32)
        nn.InstanceNorm2d(d_dim*2),
        nn.LeakyReLU(0.2),

        nn.Conv2d(d_dim*2, d_dim*4, 4, 2, 1), #16x16 (32->64)
        nn.InstanceNorm2d(d_dim*4),
        nn.LeakyReLU(0.2),

        nn.Conv2d(d_dim*4, d_dim*8, 4, 2, 1), #8x8 (64->128)
        nn.InstanceNorm2d(d_dim*8),
        nn.LeakyReLU(0.2),

        nn.Conv2d(d_dim*8, d_dim*16, 4, 2, 1), #4x4 (128->256)
        nn.InstanceNorm2d(d_dim*16),
        nn.LeakyReLU(0.2),

        nn.Conv2d(d_dim*16, 1, 4, 1, 0) #1x1 (256->1)
    )

  def forward(self, image):
    crit_pred = self.crit(image)
    return crit_pred.view(len(crit_pred), -1)

def init_weights(m):
  if isintance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
    torch.nn.init.normal_(m.weight, 0.0, 0.02)
    torch.nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.BatchNorm2d):
      torch.nn.init.normal_(m.weight, 0.0, 0.2)
      torch.nn.init.constant_(m.bias, 0)

root_path = './data/'

def save_checkpoint(name):
  torch.save({
      'epoch': epoch,
      'model_state_dict': gen.state_dict(),
      'optimizer_state_dict': gen_opt.state_dict()
  }, f"{root_path}G-{name}.pkl")

  torch.save({
      'epoch': epoch,
      'model_state_dict': crit.state_dict(),
      'optimizer_state_dict': crit_opt.state_dict()
  }, f"{root_path}C-{name}.pkl")

  print("Checkpoint saved")


def load_checkpoint(name):
  checkpoint = torch.load(f"{root_path}G-{name}.pkl")
  gen.load_state_dict(checkpoint['model_state_dict'])
  gen_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  checkpoint = torch.load(f"{root_path}C-{name}.pkl")
  crit.load_state_dict(checkpoint['model_state_dict'])
  crit_opt.load_state_dict(checkpoint['optimizer_state_dict'])

  print('Checkpoints loaded')


# GRADIENT PENALTY
def get_gp(real, fake, crit, alpha, gamma=10):
  mix_images = real * alpha + fake * (1-alpha) # 128 x 3 chan x wid 128 x ht 128
  mix_scores = crit(mix_images) # 128 x 1
  
  gradient = torch.autograd.grad(
      inputs = mix_images,
      outputs = mix_scores,
      grad_outputs=torch.ones_like(mix_scores),
      retain_graph=True,
      create_graph=True
  )[0]

  gradient = gradient.view(len(gradient), -1) 
  gradient_norm = gradient.norm(2, dim=1)
  gp = gamma * ((gradient_norm - 1)**2).mean()

  return gp

data_path = './data/celba/'
ds = Dataset(data_path, size=128, lim=10000)
dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

# MODLES
gen = Generator(z_dim).to(device)
crit = Critic().to(device)

# OPTIMIZERS
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5,0.9))
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(0.5,0.9))

x,y = next(iter(dataloader))

# TRAINING
for epoch in range(n_epochs):
  for real, _ in tqdm(dataloader):
    cur_bs = len(real) #128
    real = real.to(device)

    # CRITIC
    mean_crit_loss = 0
    for _ in range(crit_cycles):
      crit_opt.zero_grad()

      noise = gen_noise(cur_bs, z_dim)
      fake = gen(noise)
      crit_fake_pred = crit(fake.detach())
      crit_real_pred = crit(real)

      alpha = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True) #128 x 1 x 1 x 1
      gp = get_gp(real, fake.detach(), crit, alpha)

      crit_loss = crit_fake_pred.mean() - crit_real_pred.mean() + gp 
      mean_crit_loss += crit_loss.item() / crit_cycles 

      crit_loss.backward(retain_graph=True)
      crit_opt.step()

    crit_losses += [mean_crit_loss]


    # GENERATOR
    gen_opt.zero_grad()
    noise = gen_noise(cur_bs, z_dim)
    fake = gen(noise)
    crit_fake_pred = crit(fake)

    gen_loss = -crit_fake_pred.mean()
    gen_loss.backward()
    gen_opt.step()

    gen_losses += [gen_loss.item()]

    if (cur_step % show_step == 0 and cur_step > 0):
      print("saving checkpoint: ", cur_step, save_step)
      # save_checkpoint("latest")

      show(fake, wandb=1, name='fake')
      show(real, wandb=1, name='real')

      gen_mean = sum(gen_losses[-show_step:]) / show_step
      crit_mean = sum(crit_losses[-show_step:]) / show_step

      print(f"Epoch: {epoch}, Step {cur_step}, Gen loss: {gen_mean}, crit_loss: {crit_mean}") 

      plt.plot(
          range(len(gen_losses)),
                torch.tensor(gen_losses),
                label=('Gen Loss')
      )

      plt.plot(
          range(len(gen_losses)),
                torch.tensor(crit_losses),
                label=('Crit Loss')
      )

      plt.ylim(-1000,1000)
      plt.legend()
      plt.show()
    
    cur_step += 1


# Generate New Faces
noise = gen_noise(batch_size, z_dim)
fake = gen(noise)
show(fake)



# from mpl_toolkits.axes_grid1 import ImageGrid
# # MORPHING, interpolation bewteen points in latent space
# gen_set = []
# z_shape = [1,200,1,1]
# rows = 4
# steps = 17

# for i in range(rows):
#   z1, z2 = torch.randn(z_shape), torch.randn(z_shape)
#   for alpha in np.linspace(0,1,steps):
#     z = alpha * z1 + (1 - alpha) * z2
#     res = gen(z.cuda())[0]
#     gen_set.append(res)

# fig = plt.figure(figsize=(25,11))
# grid = ImageGrid(fig, 111, nrows_ncols=(rows, steps), axis_pad=0.1)

# for ax, img in zip (grid, gen_set):
#   ax.axis('off')

#   res = img.cpu().detach().permute(1,2,0)
#   res = res - res.min()
#   res = res/(res.max()-res.min())
#   ax.imshow(res)