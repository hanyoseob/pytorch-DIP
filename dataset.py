import numpy as np
import torch
import skimage
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, transform=None, sgm=25.0, wgt=0.5, size_target=(256, 256, 3), size_input=(256, 256, 3)):
        self.data_dir = data_dir
        self.transform = transform
        self.sgm = sgm
        self.wgt = wgt

        self.size_target = size_target
        self.size_input = size_input

        lst_data = os.listdir(data_dir)

        lst_data.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
        # lst_data.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        self.lst_data = lst_data
        # self.noise = self.sgm / 255.0 * np.random.randn(len(self.lst_data), self.size_target[0], self.size_target[1], self.size_target[2])
        #
        # self.input = np.random.rand(len(self.lst_data), self.size_input[0], self.size_input[1], self.size_input[2])

        mask = np.zeros(np.prod(self.size_input[:2]))

        id_mask = np.arange(np.prod(self.size_input[:2]))
        np.random.shuffle(id_mask)
        id_smp = id_mask[0:int(wgt*np.prod(self.size_input[:2]))]
        mask[id_smp] = 1
        self.mask = np.repeat(mask.reshape(self.size_input[:2])[:, :, np.newaxis], self.size_input[2], axis=2)

    def __getitem__(self, index):

        data = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))

        if data.dtype == np.uint8:
            data = data / 255.0

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)

        if data.shape[0] > data.shape[1]:
            data = data.transpose((1, 0, 2))

        # label = data
        # input = self.input[index] + self.wgt * np.random.randn(self.size_input[0], self.size_input[1], self.size_input[2])
        # target = data + self.noise[index]
        #
        # data = {'label': label, 'input': input, 'target': target}

        label = data
        mask = self.mask
        input = data * mask
        target = data * mask

        data = {'label': label, 'input': input, 'target': target, 'mask': mask}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.lst_data)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        input, label, target, mask = data['input'], data['label'], data['target'], data['mask']

        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        target = target.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label), 'target': torch.from_numpy(target), 'mask': torch.from_numpy(mask)}


class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input, label, target = data['input'], data['label'], data['target']

        input = (input - self.mean) / self.std
        label = (label - self.mean) / self.std
        target = (target - self.mean) / self.std

        data = {'input': input, 'label': label, 'target': target}
        return data


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label, target = data['input'], data['label'], data['target']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)
            target = np.fliplr(target)

        if np.random.rand() > 0.5:
            input = np.flipud(input)
            label = np.flipud(label)
            target = np.flipud(target)

        return {'input': input, 'label': label, 'target': target}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, target = data['input'], data['label'], data['target']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))
    target = transform.resize(target, (new_h, new_w))

    return {'input': input, 'label': label, 'target': target}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    input, label, target = data['input'], data['label'], data['target']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h) if (h - new_h) > 0 else 0
    left = np.random.randint(0, w - new_w) if (w - new_w) > 0 else 0

    id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
    id_x = np.arange(left, left + new_w, 1).astype(np.int32)

    # input = input[top: top + new_h, left: left + new_w]
    # label = label[top: top + new_h, left: left + new_w]

    input = input[id_y, id_x]
    label = label[id_y, id_x]
    target = target[id_y, id_x]

    return {'input': input, 'label': label, 'target': target}


class UnifromSample(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, stride):
    assert isinstance(stride, (int, tuple))
    if isinstance(stride, int):
      self.stride = (stride, stride)
    else:
      assert len(stride) == 2
      self.stride = stride

  def __call__(self, data):
    input, label, target = data['input'], data['label'], data['target]']

    h, w = input.shape[:2]
    stride_h, stride_w = self.stride
    new_h = h//stride_h
    new_w = w//stride_w

    top = np.random.randint(0, stride_h + (h - new_h * stride_h))
    left = np.random.randint(0, stride_w + (w - new_w * stride_w))

    id_h = np.arange(top, h, stride_h)[:, np.newaxis]
    id_w = np.arange(left, w, stride_w)

    input = input[id_h, id_w]
    label = label[id_h, id_w]
    target = target[id_h, id_w]

    return {'input': input, 'label': label, 'target': target}


class ZeroPad(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    input, label, target = data['input'], data['label'], data['target']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    l = (new_w - w)//2
    r = (new_w - w) - l

    u = (new_h - h)//2
    b = (new_h - h) - u

    input = np.pad(input, pad_width=((u, b), (l, r), (0, 0)))
    label = np.pad(label, pad_width=((u, b), (l, r), (0, 0)))
    target = np.pad(target, pad_width=((u, b), (l, r), (0, 0)))

    return {'input': input, 'label': label, 'target': target}

class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}

class Denormalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data = self.std * data + self.mean
        return data
