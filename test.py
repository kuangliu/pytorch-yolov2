import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from torch.autograd import Variable

from darknet import Darknet
from encoder import DataEncoder
from PIL import Image, ImageDraw


# Load model
net = Darknet()
net.load_state_dict(torch.load('model/net.pth'))
net.eval()

# Load test image
img = Image.open('/mnt/hgfs/D/air.jpg')
w,h = img.size

img1 = img.resize((416,416))
transform = transforms.Compose([transforms.ToTensor()])
img1 = transform(img1)

# Forward
y = net(Variable(img1[None,:,:,:], volatile=True))  # [1,5,25,13,13]

# Decode
encoder = DataEncoder()
boxes = encoder.decode(y.data, 416)

draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
img.show()
