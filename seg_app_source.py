import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os, cv2, time, torch as trc
from imutils import paths
from torchvision import transforms
from torchvision.utils import save_image
import sys, shutil

ROOT = os.path.dirname(os.path.realpath(__file__))
PROCESS = os.path.join(ROOT, '_ToProcess')
OUTPUT = os.path.join(ROOT, '_Masks')
NET = os.path.join(ROOT, 'MODEL')
SIZE, SHARPNESS = 256, 10

class ConvBlock(trc.nn.Module):
  def __init__(self, in_ch, out_ch):
    super().__init__()
    self.conv3_1 = trc.nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.conv3_2 = trc.nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.act = trc.nn.ReLU()
  
  def forward(self, x):
    x = self.act(self.conv3_1(x))
    return self.act(self.conv3_2(x))

class Encoder(trc.nn.Module):
  def __init__(self, ch=(3, 16, 32, 64)):
    super().__init__()
    blocks = [ConvBlock(ch[i], ch[i+1]) for i in range(len(ch)-1)]
    self.encBlocks = trc.nn.ModuleList(blocks)
    self.pool = trc.nn.MaxPool2d(2)
    self.act = trc.nn.ReLU()
  
  def forward(self, x):
    blockOutputs = []
    for block in self.encBlocks:
      x = block(x)
      blockOutputs.append(x)
      x = self.pool(x)
    return blockOutputs

class Decoder(trc.nn.Module):
  def __init__(self, ch=(64, 32, 16)):
    super().__init__()
    upsamplers, decBlocks = [], []
    for i in range(len(ch)-1):
      upsamplers.append(trc.nn.ConvTranspose2d(ch[i], ch[i+1], 2, 2))
      decBlocks.append(ConvBlock(ch[i], ch[i+1]))
    
    self.ch = ch
    self.upsamplers = trc.nn.ModuleList(upsamplers)
    self.decBlocks = trc.nn.ModuleList(decBlocks)
	
  def forward(self, x, encFeatures):
    for i in range(len(self.ch) - 1):
      x = self.upsamplers[i](x)

      # crop the current features from the encoder blocks (if needed),
      # concatenate them with the current upsampled features,
      # and pass the concatenated output through the current decoder block
      # encFeatures_for_cat = self.crop(encFeatures[i], x)
      x = trc.cat([x, encFeatures[i]], dim=1)
      x = self.decBlocks[i](x)
    return x
	
  def crop(self, encFeatures, x):
    _, _, h, w = x.shape
    res = transforms.CenterCrop([h, w])(encFeatures)
    return res

class SegNet(trc.nn.Module):
  def __init__(self, encCH=(1, 16, 32, 64), decCH=(64, 32, 16),
               nClasses=1, retainDim=True, outSize=(256, 256)):
    super().__init__()
    self.retainDim, self.outSize = retainDim, outSize
    self.encoder, self.decoder = Encoder(encCH), Decoder(decCH)
    self.head = trc.nn.Conv2d(decCH[-1], nClasses, 1)
    self.act = trc.nn.Sigmoid()
  
  def forward(self, x):
    encFeatures = self.encoder(x)
    dec_res = self.decoder(encFeatures[-1], encFeatures[::-1][1:])
    mask = self.head(dec_res)
    # resize to match initial size if needed
    # trc.nn.functional.interpolate(mask, self.outSize)
    return self.act(mask)

class processDataset(trc.utils.data.Dataset):
  def __init__(self, imgPaths, transforms=None):
    self.imgPaths, self.transforms = imgPaths, transforms

  def __len__(self):
    return len(self.imgPaths)
	
  def __getitem__(self, idx):
    imageOriginal = cv2.imread(self.imgPaths[idx])
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if self.transforms is not None:
      image = self.transforms(imageOriginal)

    imgName = self.imgPaths[idx].split('\\')[-1]
    imgShape = (imageOriginal.shape[0], imageOriginal.shape[1])
    return image, (imgName, imgShape)

read_img = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize((SIZE, SIZE)),
  transforms.ToTensor(),

  transforms.Grayscale(),
  transforms.RandomAdjustSharpness(SHARPNESS, 1),
])

imgPaths = list(paths.list_images(PROCESS))

if len(imgPaths) > 0:
  dt_process = processDataset(imgPaths, read_img)
  batch_size = min(500, len(dt_process))
  process_dl = trc.utils.data.DataLoader(dt_process, batch_size)
  print('Processing '+str(len(dt_process))+' images...')
  start_time = time.time()
  
  net = trc.load(NET)
  preds, Data  = trc.tensor([]), []

  for X, data in process_dl:
    with trc.set_grad_enabled(False):
      res = net.forward(X).data
    for i in range(len(data[0])):
      Data.append((data[0][i], (data[1][0][i], data[1][1][i])))
    preds = trc.cat([preds, res])
  print('Finished! Total time: '+str(time.time()-start_time)+'\n')

  print('Saving masks...')
  shutil.rmtree(OUTPUT)
  os.mkdir(OUTPUT)

  if len(sys.argv) == 1:
    for i, data in enumerate(Data):
      resize = transforms.Resize(data[1])
      image = preds[i].clone()
      image = resize(image)
      sv_path = os.path.join(OUTPUT, data[0])
      save_image(image, sv_path)
  else:
    for i, data in enumerate(Data):
      resize = transforms.Resize(data[1])
      image = preds[i].clone()
      image = (image >= float(sys.argv[1])).int()*255.0
      image = resize(image)
      sv_path = os.path.join(OUTPUT, data[0])
      save_image(image, sv_path)
  print('Done!')
else:
  print('Images not found...')
