from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms as T
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
import math
from torchvision.transforms.transforms import _setup_size, _log_api_usage_once
from collections.abc import Sequence
from typing import List, Optional, Tuple, Union
from torch import Tensor
import random



def hide_integers(image, int1, int2):
    assert image.dtype == torch.uint8
    assert int1 < 2**16
    assert int2 < 2**16

    # Ensure the image is a 224x224x3 tensor
    assert list(image.size()) == [3, 224, 224]

    # Convert integers to binary strings, zero-padded to 16 bits
    bin1 = format(int1, '016b')
    bin2 = format(int2, '016b')

    # Convert binary strings to tensors of bits
    bits1 = torch.tensor([int(bit) for bit in bin1], dtype=torch.uint8)
    bits2 = torch.tensor([int(bit) for bit in bin2], dtype=torch.uint8)

    # We'll store int1 in the top-left corner of the first channel, and int2 in the top-left corner of the second channel
    image[0, 0:4, 0:4] = (image[0, 0:4, 0:4] & 0b11111110) | bits1.view(4, 4)
    image[1, 0:4, 0:4] = (image[1, 0:4, 0:4] & 0b11111110) | bits2.view(4, 4)
    return image

def extract_integers(image):
    bits1 = (image[0, 0:4, 0:4] & 0b00000001).view(-1)
    bits2 = (image[1, 0:4, 0:4] & 0b00000001).view(-1)

    # avoid unnecessary conversion to numpy
    int1 = int(''.join(map(str, bits1.tolist())), 2)
    int2 = int(''.join(map(str, bits2.tolist())), 2)

    return int1, int2

class ToUint8Tensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
        """
        Args:
            image (PIL Image): Image to be converted to uint8 tensor.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f'pic should be PIL Image. Got {type(image)}')

        # Convert PIL image to numpy array
        np_img = np.array(image)

        # Convert numpy array to torch tensor
        tensor = torch.from_numpy(np_img)

        # Rearrange dimensions to: C x H x W
        return tensor.permute(2, 0, 1).contiguous()


class ToFloatTensor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image):
       return image.float().div(255)


class ResizeAndEncodeOriginalSize(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int)
        assert isinstance(output_size[1], int)
        self.output_size = output_size

    def forward(self, image):
        assert image.dtype == torch.uint8
        assert image.dim() == 3
        assert image.size(0) == 3

        # get original size
        orig_size = image.size()
        orig_height = orig_size[1]
        orig_width = orig_size[2]

        # resize image
        output_size = self.output_size
        image = F.resize(image, output_size, antialias=True)

        # encode original size
        image = hide_integers(image, orig_height, orig_width)

        return image

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + f"(output_size={self.output_size})"
        return format_string


iwm_raw_transform = T.Compose([
    ToUint8Tensor(),
    ResizeAndEncodeOriginalSize((224, 224)),
])


class RandomResizedCrop(torch.nn.Module):
    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__()
        _log_api_usage_once(self)
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            # warnings.warn("Scale and ratio should be of kind (min, max)")
            raise ValueError("Scale and ratio should be of kind (min, max)")

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (Tensor): Input image.
            scale (list): range of scale of the origin size cropped
            ratio (list): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        # _, height, width = F.get_dimensions(img)
        assert isinstance(img, Tensor)
        assert img.dim() == 3
        assert img.dtype == torch.uint8
        _, resized_height, resized_width = img.size()
        height, width = extract_integers(img)

        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                # i = torch.randint(0, height - h + 1, size=(1,)).item()
                # j = torch.randint(0, width - w + 1, size=(1,)).item()

                # [author1]:
                # correct with respect to the resized size
                w = int(round(w * resized_width / width))
                h = int(round(h * resized_height / height))
                i = torch.randint(0, resized_height - h + 1, size=(1,)).item()
                j = torch.randint(0, resized_width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2

        # [author1]:
        # correct with respect to the resized size
        w = int(round(w * resized_width / width))
        h = int(round(h * resized_height / height))
        i = int(round(i * resized_height / height))
        j = int(round(j * resized_width / width))
        return i, j, h, w

    def forward(self, img: Tensor) -> Tensor:
        # check if img is a batch
        if img.dim() == 4:
            return self.batch_forward(img)
        else:
            return self.forward_single(img)

    def forward_single(self, img):
        """
        Args:
            img (Tensor): Image to be cropped and resized.

        Returns:
            Tensor: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

    def batch_forward(self, img):
        batch_size = img.size(0)
        output = torch.empty(
            [batch_size, 3, self.size[0], self.size[1]],
            dtype=torch.uint8,
            device=img.device
        )
        for batch_idx in range(batch_size):
            output[batch_idx] = self.forward_single(img[batch_idx])

        return output

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str}"
        format_string += f", antialias={self.antialias})"
        return format_string
