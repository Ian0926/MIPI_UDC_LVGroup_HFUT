import time
import torch

import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

from utils import *
from options import TestOptions
from models import UNet, UNet64, UNetLong
from datasets import SingleImgDataset

print('---------------------------------------- step 1/4 : parameters preparing... ----------------------------------------')
opt = TestOptions().parse()

image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer/img'
npy_dir = opt.outputs_dir + '/' + opt.experiment + '/infer/npy'
clean_dir(image_dir, delete=opt.save_image)
clean_dir(npy_dir, delete=opt.save_image)

print('---------------------------------------- step 2/4 : data loading... ------------------------------------------------')
print('inferring data loading...')
infer_dataset = SingleImgDataset(data_source=opt.data_source)
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))

print('---------------------------------------- step 3/4 : model defining... ----------------------------------------------')
model = UNet().cuda()
# model = UNet64().cuda()
# model = UNetLong().cuda()
# model = nn.DataParallel(model)
print_para_num(model)

model.load_state_dict(torch.load(opt.model_path))
print('successfully loading pretrained model.')

print('---------------------------------------- step 4/4 : testing... ----------------------------------------------------')   
def main():
    model.eval()
    
    psnr_meter = AverageMeter()
    time_meter = AverageMeter()
    
    for i, (img, path) in enumerate(infer_dataloader):
        img = img.cuda()

        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            times = time.time() - start_time
        
        pred_clip = torch.clamp(pred, 0, 1)
        
        npy_clip = torch.clamp(pred.squeeze(0), 0, 500/500.25).cpu().numpy()
        npy_clip = np.transpose(npy_clip, (1, 2, 0))
        npy_clip = inverse_tonemap(npy_clip)
        
        time_meter.update(times, 1)

        print('Iteration: ' + str(i+1) + '/' + str(len(infer_dataset)) + '  Processing image... ' + str(path) + '  Time ' + str(times))
            
        if opt.save_image:
            npy_path = os.path.basename(path[0])
            raw_name = os.path.splitext(npy_path)[0]
            save_image(pred_clip, image_dir + '/' + raw_name + '_restored.png')
            save_image(img, image_dir + '/' + raw_name + '_img.png')
            np.save(npy_dir + '/' + npy_path, npy_clip)
            
    print('Avg time: ' + str(time_meter.average()))

def inverse_tonemap(y):
    return 0.25/(1-y) - 0.25

if __name__ == '__main__':
    main()
    