import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import PythianEngine
from utils import *
from collections import deque

class Tester:
    def __init__(self, size=(256,256),
                       nheads=6,
                       expansion=4,
                       nlayers=6,
                       maxlen=16,
                       kernel_size=(7, 7),
                       insize=8*8*3,
                    ):
        self.size = size
        self.nheads = nheads
        self.nlayers = nlayers
        self.expansion = expansion
        self.maxlen = maxlen

        self.engine = PythianEngine(nheads, expansion,nlayers,3,3,kernel_size,maxlen)
        self.camera, self.display, self.surface = init_camera_and_window()
        self.memory = deque(maxlen=self.maxlen)

    def load(self, path='../saves/checkpoint.pt'):
        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']

    def evaluate(self, start=None):
        with torch.no_grad(): 
            if start is None:
                self.memory.append(torch.zeros((1,3,self.size[0],self.size[1])))
            else:
                self.memory.extend(start)

            self.engine.eval()
            acc = 0
            while True:
                if space_is_pressed():
                    outp_engine = self.engine(torch.stack(list(self.memory), -1))[0]
                    print(outp_engine.shape)
                    self.memory.append(outp_engine.split(1, -1)[-1].squeeze(-1))
                    show_tensor(self.memory[-1], self.display, self.surface)
                else:
                    self.memory.append((get_shot(self.camera, size=self.size)))
                    show_tensor((self.memory[-1]), self.display, self.surface)
if __name__ == '__main__':
    trainer = Tester()
    trainer.load()
    trainer.evaluate()