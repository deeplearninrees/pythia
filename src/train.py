from multiprocessing import reduction
import torch
import torch.nn.functional as F
import torch.utils.data.dataloader as D
from net import PythianEngine
from dataset import Dataset
from optim import Lamb
from utils import *
from sklearn.model_selection import train_test_split
import atexit
import time
import sys

def train_test_data(dataset, test_split):
    train_idx, test_idx = train_test_split(range(len(dataset)), test_size=test_split)
    return torch.utils.data.Subset(dataset, train_idx), torch.utils.data.Subset(dataset, test_idx)

class Trainer:
    def __init__(self, nheads=6,
                       nlayers=6,
                       expansion=4,
                       kernel_size=(7,7),
                       nmbatches=64,
                       epochs=100,
                       max_ctx_length=8,
                       batch_size=1,
                       lr=3e-3
                    ):
        self.nlayers = nlayers
        self.expansion = expansion
        self.kernel_size = kernel_size
        self.nmbatches = nmbatches
        self.epochs = epochs
        self.max_ctx_length = max_ctx_length
        self.batch_size = batch_size
        self.lr = lr
        
        print("Loading Net...")
        self.engine = PythianEngine(nheads, expansion, nlayers, 3, 3, kernel_size=self.kernel_size, max_length=self.max_ctx_length)
        print("Net has", get_nparams(self.engine), "parameters.")
        
        self.dataset = Dataset(max_ctx_length=max_ctx_length)
        self.train_dataset, self.test_dataset = train_test_data(self.dataset, .2)
        self.train_dataloader = D.DataLoader(self.train_dataset, shuffle=True, batch_size=batch_size)
        self.test_dataloader = D.DataLoader(self.test_dataset, shuffle=True, batch_size=batch_size)

        self.optim = Lamb(self.engine.parameters(), lr=self.lr)
        _, self.display, self.surface = init_camera_and_window()

    def train(self):
        """Train the model."""

        print("Beginning Training...")
        running_engine_loss = 0
        self.engine.train()
        for epoch in range(self.epochs):
            print("Starting Epoch:", epoch)
            for i, (x, y) in enumerate(self.train_dataloader):
                if i < self.nmbatches:
                    
                    engine_loss = self.training_step(x, y, i)
                    running_engine_loss += engine_loss

                    print("Engine Loss:", "{:3f}".format(engine_loss), "~", 
                          "Avg Engine Loss:", "{:3f}".format(running_engine_loss/(i+1)), "~",
                            "Iterations:", i+1)
                else:
                   break
            print("End Training Epoch:", epoch)
            
            running_engine_loss = 0
            
            if epoch % 2 == 0 and epoch != 0:
                print("Starting Validation Epoch.")
                self.engine.eval()
                for i, (x, y) in enumerate(self.test_dataloader):
                    if i < self.nmbatches:

                        engine_loss = self.validation_step(x, y, i)
                        running_engine_loss += engine_loss
                        print("Engine Loss:", "{:3f}".format(engine_loss), "~",
                            "Avg Engine Loss:", "{:3f}".format(running_engine_loss/(i+1)), "~",
                            "Iterations:", i+1)

                    else:
                        break
            running_engine_loss = 0

            self.save()

    def training_step(self, x, y, step):
        """
        One optimization step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        #Generator training step.
        self.optim.zero_grad()
        y_false, kl = self.engine(x)

        bce_loss = 20 * torch.nn.functional.binary_cross_entropy(y_false, y, reduction='sum')
        loss = kl + bce_loss
        
        loss.backward()
        self.optim.step()
        gen_loss = loss.item()


        #Show results. :)
        y_seq = torch.cat([y, y_false], 2)
        for i in y_seq.split(1, -1):
            show_tensor(i.squeeze(-1), self.display, self.surface)
            time.sleep(0.1)

        return gen_loss
    def validation_step(self, x, y, step):
        """
        One validation step
        :param x: Input data
        :param y: Target data
        :param step: Current training step
        :return: loss
        """
        y_false, kl = self.engine(x)
        loss = (20 * F.binary_cross_entropy(y_false, y, reduction='sum') + kl)

        return loss.item()

    def save(self, path='../saves/checkpoint.pt'):
        """Save the model to disk."""
        torch.save({
            'optim':self.optim.state_dict(),
            'engine':self.engine.state_dict(),
            }, path)

    def load(self, path='../saves/checkpoint.pt'):
        """Load the model from disk."""

        checkpoint = torch.load(path, map_location='cpu')
        self.engine.load_state_dict(checkpoint['engine'])
        del checkpoint['engine']
        self.optim.load_state_dict(checkpoint['optim'])
        del checkpoint['optim']
        
if __name__ == '__main__':
    trainer = Trainer()
    atexit.register(lambda:trainer.save())
    trainer.train()