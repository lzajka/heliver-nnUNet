#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import os
import numpy as np
import dask.array as da
import zarr
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from skimage import io
import PIL.Image
from dask import delayed
from numcodecs import Blosc
# It is necessary to increase max pixels to load large images. In the future I could use more reasonable approach and split image based on background.
PIL.Image.MAX_IMAGE_PIXELS = None

class NaturalImage2DIO(BaseReaderWriter):
    """
    ONLY SUPPORTS 2D IMAGES!!!
    """

    # there are surely more we could add here. Everything that can be read by skimage.io should be supported
    supported_file_endings = [
        '.png',
        # '.jpg',
        # '.jpeg', # jpg not supported because we cannot allow lossy compression! segmentation maps!
        '.bmp',
        '.tif'
    ]

    @staticmethod
    def _parse_png_chunk_shape(height: int, width: int, channels: int = None):
        default_h = int(os.environ.get('NNUNET_PNG_ZARR_CHUNK_H', 1024))
        default_w = int(os.environ.get('NNUNET_PNG_ZARR_CHUNK_W', 1024))
        if channels is None:
            return max(1, min(default_h, height)), max(1, min(default_w, width))
        return max(1, min(default_h, height)), max(1, min(default_w, width)), channels

    @staticmethod
    def _png_compressor():
        clevel = int(os.environ.get('NNUNET_PNG_ZARR_CLEVEL', os.environ.get('NNUNET_ZARR_CLEVEL', 5)))
        cname = os.environ.get('NNUNET_PNG_ZARR_CNAME', os.environ.get('NNUNET_ZARR_CNAME', 'zstd'))
        return Blosc(cname=cname, clevel=clevel, shuffle=Blosc.BITSHUFFLE)

    @staticmethod
    def _read_via_dask_and_zarr(fname: str):
        with PIL.Image.open(fname) as img:
            width, height = img.size
            mode = img.mode
        channels = len(mode) if mode in ('RGB', 'RGBA') else None
        dtype = np.uint16 if mode.startswith('I;16') else np.uint8
        delayed_img = delayed(io.imread)(fname)
        delayed_shape = (height, width, channels) if channels is not None else (height, width)
        raw = da.from_delayed(delayed_img, shape=delayed_shape, dtype=dtype)
        if channels is not None:
            chunks = NaturalImage2DIO._parse_png_chunk_shape(height, width, channels)
        else:
            chunks = NaturalImage2DIO._parse_png_chunk_shape(height, width, None)

        raw = raw.rechunk(chunks)

        cache_root = os.environ.get('NNUNET_PNG_ZARR_CACHE_DIR', None)
        if cache_root:
            os.makedirs(cache_root, exist_ok=True)
            zarr_cache = os.path.join(cache_root, os.path.basename(fname) + '.zarr')
            arr = zarr.open(
                zarr_cache,
                mode='w',
                shape=tuple(int(i) for i in raw.shape),
                chunks=chunks,
                dtype=raw.dtype,
                compressor=NaturalImage2DIO._png_compressor()
            )
            da.store(raw, arr, lock=False)
            return np.asarray(arr)
        return raw.compute()

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]]) -> Tuple[np.ndarray, dict]:
        images = []
        use_dask_zarr = os.environ.get('NNUNET_PNG_USE_DASK_ZARR', '0').lower() in ('1', 'true', 'yes', 'on')
        for f in image_fnames:
            npy_img = self._read_via_dask_and_zarr(f) if use_dask_zarr else io.imread(f)
            if npy_img.ndim == 3:
                # rgb image, last dimension should be the color channel and the size of that channel should be 3
                # (or 4 if we have alpha)
                assert npy_img.shape[-1] == 3 or npy_img.shape[-1] == 4, "If image has three dimensions then the last " \
                                                                         "dimension must have shape 3 or 4 " \
                                                                         f"(RGB or RGBA). Image shape here is {npy_img.shape}"
                # move RGB(A) to front, add additional dim so that we have shape (c, 1, X, Y), where c is either 3 or 4
                images.append(npy_img.transpose((2, 0, 1))[:, None])
            elif npy_img.ndim == 2:
                # grayscale image
                images.append(npy_img[None, None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images, dtype=np.float32, casting='unsafe'), {'spacing': (999, 1, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ))

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        io.imsave(output_fname, seg[0].astype(np.uint8 if np.max(seg) < 255 else np.uint16, copy=False), check_contrast=False)


if __name__ == '__main__':
    images = ('/media/fabian/data/nnUNet_raw/Dataset120_RoadSegmentation/imagesTr/img-11_0000.png',)
    segmentation = '/media/fabian/data/nnUNet_raw/Dataset120_RoadSegmentation/labelsTr/img-11.png'
    imgio = NaturalImage2DIO()
    img, props = imgio.read_images(images)
    seg, segprops = imgio.read_seg(segmentation)
