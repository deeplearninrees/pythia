import glob
import torch
import torch.nn.functional as F
import torchvision.io as Tvio
import multiprocessing as mp

class Dataset(torch.utils.data.Dataset):
    def __init__(self, directory='../data/*', get_subdirs=True, max_ctx_length=32):
        print("Loading dataset...")
        self.data = glob.glob(directory)
        if get_subdirs:
            self.data_temp = []
            with mp.Pool(mp.cpu_count()) as data_procs:
                for i in data_procs.imap_unordered(self.extract_from_data, list(range(len(self.data)))):
                    self.data_temp.extend(i)
            data_procs.join()
            self.data = self.data_temp
            self.max_ctx_length = max_ctx_length
    def extract_from_data(self, i):
        i_data = self.data[i]
        file_data = glob.glob(i_data+"/*")
        file_data.sort(key=lambda r: int(''.join(x for x in r if (x.isdigit()))))
        return file_data

    def __len__(self):
        return len(self.data) - self.max_ctx_length - 1

    def __getitem__(self, key):
        frame_start = key
        frames = []
        i_frame = frame_start
           
        while len(frames) <= self.max_ctx_length:
            frame = ((Tvio.read_image(self.data[i_frame], mode=Tvio.ImageReadMode.RGB).float() / 255))
            frames.append(frame)
            i_frame += 1 #OOPS, I think I wasnt moving the frames along, loll... Whoopsa...
        data_x = frames[0:-1]
        data_y =frames[1:]

        return torch.stack(data_x, dim=-1), torch.stack(data_y, dim=-1)

if __name__ == "__main__":
    dataset = Dataset()
    print(len(dataset))
    print(dataset.__getitem__(8)[0].shape)