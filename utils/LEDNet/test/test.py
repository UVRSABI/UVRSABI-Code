import torch
import os
from PIL import Image
from argparse import ArgumentParser
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize
from torchvision.transforms import ToTensor, ToPILImage
from transform import Relabel, ToLabel, Colorize
from torchvision.utils import save_image
from dataset import VOCSegmentation
from lednet import Net
import warnings
import sys
warnings.filterwarnings("ignore")

NUM_CHANNELS = 3
NUM_CLASSES = 2

class MyVOCTransform(object):
    def __init__(self):
        pass
    def __call__(self, input, target):
        input =  Resize((512, 1024),Image.Resampling.BILINEAR)(input)
        input = ToTensor()(input)
        if target is not None:
            target = Resize((512, 1024),Image.NEAREST)(target)
            target =ToLabel()(target)
            target = Relabel(255, 0)(target)
        return input, target

def print_progress_bar(index, total, label):
    n_bar = 50  
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

def main(args):

    modelpath = args.loadModel
    weightspath = os.path.join(args.loadDir,args.loadWeights)

    print ("Loading model: " + modelpath)
    print ("Loading weights: " + weightspath)

    model = Net(NUM_CLASSES)
  
    model = torch.nn.DataParallel(model)
    if (not args.cpu):
        model = model.cuda()


    def load_my_state_dict(model, state_dict): 
        own_state = model.state_dict()
        for name, param in state_dict['state_dict'].items():
            if name not in own_state:
                 continue
            own_state[name].copy_(param)
           
        return model

    model = load_my_state_dict(model, torch.load(weightspath))
    print ("Model and weights loaded successfully.")

    model.eval()
    print('Model Eval completed successfully.')

    if(not os.path.exists(args.datadir)):
        print ("Error: datadir could not be loaded")

    voc_transform = MyVOCTransform()
    dataset = VOCSegmentation(args.datadir, transforms=voc_transform)

    loader = DataLoader(dataset,num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)

    for step, (images, filename) in enumerate(loader):
        print_progress_bar(step, len(loader)-1, 'Image: '+ str(filename[0])+'.png')
        filtered_Filename = dataset.FileName(step)
        # filtered_Filename = filtered_Filename[2:len(filtered_Filename)-3]
        if (not args.cpu):
            images = images.cuda()

        inputs = Variable(images)
        with torch.no_grad():
            outputs = model(inputs)


        label = outputs[0].max(0)[1].byte().cpu().data
        # filtered_Filename += ".png"
        os.makedirs(args.resultdir, exist_ok=True)
        filenameSave = os.path.join(args.resultdir,filtered_Filename)

        os.makedirs(os.path.dirname(filenameSave), exist_ok=True)

        label = label*255.0
 
        label_save = ToPILImage()(label.type(torch.uint8))  
        label_save = label_save.resize(dataset.ImageSize(step))

        label_save.save(filenameSave)
    print('\nModel inference on all images saved successfully.')
    

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--state')


    parser.add_argument('--loadDir',default="../save/logs/")
    parser.add_argument('--loadWeights', default="model_best.pth.tar")
    parser.add_argument('--loadModel', default="lednet.py")
    parser.add_argument('--resultdir', default="./results")
    parser.add_argument('--datadir', required='True')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')


    main(parser.parse_args())