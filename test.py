import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from darknet import Darknet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from utils import meshgrid


# Load model
net = Darknet()
net.load_state_dict(torch.load('model/net.pth'))
net.eval()

# Load test image
# img = Image.open('/mnt/hgfs/D/download/PASCAL VOC/voc_all_images/2007_000001.jpg')
img = Image.open('./imgs/000001.jpg')
w,h = img.size

# Forward
img1 = img.resize((416,416))
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
img1 = transform(img1)

# Forward
y = net(Variable(img1[None,:,:,:], volatile=True))  # [1,5,25,13,13]
y = y.data.view(5,25,13,13)

# Decode
encoder = DataEncoder()
boxes = encoder.decode(y, 416)

# Show
draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
img.show()
