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
net.load_state_dict(torch.load('net.pth'))
net.eval()

# Load test image
# img = Image.open('/mnt/hgfs/D/download/PASCAL VOC/voc_all_images/2007_000001.jpg')
img = Image.open('/search/data/user/liukuang/data/VOC2012_trainval_test_images/2007_000061.jpg')
#boxes = torch.Tensor([48, 240, 195, 371, 8, 12, 352, 498]).view(2,4)
#labels = torch.LongTensor([11,14])
#w,h = img.size
#boxes /= torch.Tensor([w,h,w,h]).expand_as(boxes)

# Forward
img1 = img.resize((416,416))
transform = transforms.Compose([transforms.ToTensor()])
img1 = transform(img1)

# Forward
y = net(Variable(img1[None,:,:,:], volatile=True))  # [1,5,25,13,13]
y = y.data.view(5,25,13,13)
#
# # Decode
encoder = DataEncoder()
boxes = encoder.decode(y, 416)
#
# # Show
draw = ImageDraw.Draw(img)
for box in boxes:
    box[::2] *= img.width
    box[1::2] *= img.height
    draw.rectangle(list(box), outline='red')
# img.show()
img.save('ret.jpg')

#loc, conf, prob = encoder.encode(boxes, labels, 416)
#loc
