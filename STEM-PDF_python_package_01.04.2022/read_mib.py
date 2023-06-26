import time
import io
import os
import numpy as np

class MIBFile(object):
    def __init__(self, path, fields=None):
        self.path = path
        if fields is None:
            self._fields = {}
        else:
            self._fields = fields

    def _get_np_dtype(self, dtype):
        dtype = dtype.lower()
        assert dtype[0] == "u"
        num_bytes = int(dtype[1:]) // 8
        return ">u%d" % num_bytes

    def read_header(self):
        with io.open(file=self.path, mode="r", encoding="ascii") as f:
            header = f.read(100)
            filesize = os.fstat(f.fileno()).st_size
        parts = header.split(",")
        image_size = (int(parts[5]), int(parts[4]))
        header_size_bytes = int(parts[2])
        bytes_per_pixel = int(parts[6][1:]) // 8
        num_images = filesize // (
            image_size[0] * image_size[1] * bytes_per_pixel + header_size_bytes
        )
        self._fields = {
            'header_size_bytes': header_size_bytes,
            'dtype': self._get_np_dtype(parts[6]),
            'bytes_per_pixel': bytes_per_pixel,
            'image_size': image_size,
            'sequence_first_image': int(parts[1]),
            'filesize': filesize,
            'num_images': num_images,
        }
        return self._fields

    @property
    def fields(self):
        if not self._fields:
            self.read_header()
        return self._fields

    def _frames(self, num, offset):
        """
        read frames as views into the memmapped file
        Parameters
        ----------
        num : int
            number of frames to read
        offset : int
            index of first frame to read (number of frames to skip)
        """
        bpp = self.fields['bytes_per_pixel']
        hsize = self.fields['header_size_bytes']
        assert hsize % bpp == 0
        size_px = self.fields['image_size'][0] * self.fields['image_size'][1]
        size = size_px * bpp  # bytes
        imagesize_incl_header = size + hsize  # bytes
        mapped = np.memmap(self.path, dtype=self.fields['dtype'], mode='r',
                           offset=offset * imagesize_incl_header)

        # limit to number of frames to read
        a = np.int64(num) * np.int64((size_px + hsize // bpp))
        #mapped = mapped[:num * (size_px + hsize // bpp)]
        mapped = mapped[:a]
        # reshape (num_frames, pixels) incl. header
        mapped = mapped.reshape((num, size_px + hsize // bpp))
        # cut off headers
        mapped = mapped[:, (hsize // bpp):]
        # reshape to (num_frames, pixels_y, pixels_x)
        return mapped.reshape((num, self.fields['image_size'][0], self.fields['image_size'][1]))

    def read_frames(self, num, offset, out, crop_to):
        """
        Read a number of frames into an existing buffer, skipping the headers
        Parameters
        ----------
        num : int
            number of frames to read
        offset : int
            index of first frame to read
        out : buffer
            output buffer that should fit `num` frames
        crop_to : Slice
            crop to the signal part of this Slice
        """
        frames = self._frames(num=num, offset=offset)
        if crop_to is not None:
            frames = frames[(...,) + crop_to.get(sig_only=True)]
        out[:] = frames
        return out


if __name__ == "__main__":
    pass
