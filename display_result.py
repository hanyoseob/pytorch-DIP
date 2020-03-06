import os
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt

dir_result = './results/inpainting_hourglass_lr1e-1/restoration1/train/images'
lst_result = os.listdir(dir_result)

lst_input = [f for f in lst_result if f.endswith('input.png')]
lst_label = [f for f in lst_result if f.endswith('label.png')]
lst_output = [f for f in lst_result if f.endswith('output.png')]

lst_input.sort()
lst_label.sort()
lst_output.sort()

# lst_input.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
# lst_label.sort(key=lambda f: (''.join(filter(str.isdigit, f))))
# lst_output.sort(key=lambda f: (''.join(filter(str.isdigit, f))))

nx = 512
ny = 512
nch = 1

n = 3
m = 6

inputs = torch.zeros((m, ny, nx, nch))
labels = torch.zeros((m, ny, nx, nch))
outputs = torch.zeros((m, ny, nx, nch))

##
id = [1, 500, 1000, 3000, 10000, 20000]
for i in range(m):
    inputs[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, '%04d-input.png' % id[i]))[:, :, :nch])
    labels[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, '%04d-label.png' % id[i]))[:, :, :nch])
    outputs[i, :, :, :] = torch.from_numpy(plt.imread(os.path.join(dir_result, '%04d-output.png' % id[i]))[:, :, :nch])

inputs = inputs.permute((0, 3, 1, 2))
labels = labels.permute((0, 3, 1, 2))
outputs = outputs.permute((0, 3, 1, 2))

images = torch.cat([inputs, labels, outputs], axis=2)

plt.figure(figsize=(n, m))
plt.axis("off")
# plt.title("Generated Images")
plt.imshow(np.transpose(vutils.make_grid(images, padding=2, normalize=False), (1, 2, 0)))

plt.show()

