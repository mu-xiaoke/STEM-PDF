from libertem.udf import UDF
import numpy as np
import numba

def mean_center(fig, range_x, range_y):
    """calculates COM of a 2d image and returns center coordinates"""
    fig_x = np.sum(fig, axis=1)
    fig_y = np.sum(fig, axis=0)
    x = np.dot(fig_x, range_x)
    weight = np.sum(fig_x)
    x = x/weight
    y = np.dot(fig_y, range_y)
    y = y/weight
    
    return x, y

@numba.njit(inline="always")
def round_for_hist(r):
    # TODO
    return int(np.round(r))


@numba.njit
def radial_bins(arr, cy, cx, detector_shape):
    # how big is it? the smallest inner circle on the detector
    # depending on the shift of the center, we then need to cut
    # the result afterwards
    max_r = int(min(detector_shape)/2)
    hist = np.zeros((round_for_hist(max_r)), dtype=np.float32)
    count = np.zeros((round_for_hist(max_r)), dtype=np.float32)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r = ((cy-y)**2+(cx-x)**2)**0.5
            if r > max_r-1:
                continue
            h_r = round_for_hist(r)
            hist[h_r] += arr[y, x]
            count[h_r] += 1
    return np.divide(hist, count, out=np.zeros_like(hist), where=count!=0)

class RadialIntegration(UDF):
    def get_task_data(self):
        shape = self.meta.dataset_shape.sig
        return{
            "range_y" : np.arange(0, shape[0]),
            "range_x" : np.arange(0, shape[1])
        }
        
    def get_result_buffers(self):
        max_r=int(min(self.meta.dataset_shape.sig)/2)
        return {
            "radial" : self.buffer(kind="nav", dtype="float32",extra_shape=(max_r,))
        }
    
    def process_frame(self, frame):
        x, y = mean_center(frame,range_x=self.task_data.range_x, range_y=self.task_data.range_y)
        self.results.radial[:] = radial_bins(frame, x, y, frame.shape)

class CoMUDF(UDF):
    def get_task_data(self):
        shape = self.meta.dataset_shape.sig
        return{
            "range_y" : np.arange(0, shape[0]),
            "range_x" : np.arange(0, shape[1])
        }
        
    def get_result_buffers(self):
        return {
            "center" : self.buffer(kind="nav", dtype="float32",extra_shape=(2,))
        }
    
    def process_frame(self, frame):
        self.results.center[:] = mean_center(frame,range_x=self.task_data.range_x, range_y=self.task_data.range_y)

class NaNcount(UDF):
        
    def get_result_buffers(self):
        return {
            "nans" : self.buffer(kind="single", dtype="float32")
        }
    
    def process_frame(self, frame):
        self.results.nans[:] = np.count_nonzero(frame==0)#np.sum(np.isnan(frame))
    
    def merge(self, dest, src):
        
        dest.nans[:] += src.nans