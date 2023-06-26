import os
import copy
import time
import matplotlib
import numpy as np
from PIL import Image
from pandas import cut
import scipy.signal as signal
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.transform import hough_circle, hough_circle_peaks, hough_ellipse
from skimage.filters import sobel

def fill_mask(mask, initial_point):
    fill_in = np.zeros(np.shape(mask))
    edge_list_old = [initial_point]

    for i in np.arange(1, 500):
        edge_list = []
        for item in edge_list_old:
            edge_list, fill_in = neighbor_fill(item, mask, edge_list, fill_in)
        if np.array_equiv(edge_list, edge_list_old):
            print('search complete\n', 'iterations:', i)
            return fill_in
        else:
            edge_list_old = edge_list
    return fill_in


def neighbor_fill(initial_point, mask, edge_list_old, fill_in):
    mask_size = np.shape(mask)
    neighbor = np.array([[-5, 0],
                         [0, -5],
                         [5, 0],
                         [0, 5]])
    neighbor = neighbor + initial_point
    try:
        x = initial_point[0]
        y = initial_point[1]
    except IndexError:
        return edge_list_old, fill_in
    else:

        for i in np.arange(np.shape(neighbor)[0]):
            x = neighbor[i, 0]
            y = neighbor[i, 1]
            if x < 0 or x >= mask_size[0] or y < 0 or y >= mask_size[1]:
                continue
            if fill_in[x, y] == 1:
                continue
            if mask[x, y] == 0:
                fill_in[x, y] = 1
                if not len(edge_list_old):
                    edge_list_old = [x, y]
                else:
                    edge_list_old = np.row_stack((edge_list_old, [x, y]))

        return edge_list_old, fill_in


def distance_matrix(fig, center):
    """
    the center is n-th pixel, not the position read from matplot(because they start from 0
    """
    fig_size = np.shape(fig)
    row = np.array([np.arange(1, fig_size[0] + 1)])
    col = np.array([np.arange(1, fig_size[1] + 1)])
    row_mat = np.tile(row.T, [1, fig_size[1]])
    col_mat = np.tile(col, [fig_size[0], 1])
    distance_mat = np.power(row_mat - center[0], 2) + np.power(col_mat - center[1], 2)
    distance_mat = np.sqrt(distance_mat)
    distance_mat = np.array(distance_mat + 0.5, dtype=np.int)
    '''
    a = Figure()
    a.add_subplot(111).imshow(distance_mat)
    RDF_TK_GUI.ImageWindow(a)
    '''
    return distance_mat


def delete_zero(list):
    distance = list[0, :]
    value = list[1, :]
    index = np.argwhere(value == 0)
    value_new = np.delete(value, index)
    distance_new = np.delete(distance, index)
    return distance_new, value_new


def annular_average(distance, value, radius_length):
    distance = np.array(distance).astype(np.float)
    value = np.array(value).astype(np.float)
    index = np.isfinite(value)
    distance = distance[index]
    value = value[index]

    acc_weight = np.histogram(distance, bins=np.array(np.arange(int(radius_length))), weights=value)[0]
    acc_number = np.histogram(distance, bins=np.array(np.arange(int(radius_length))))[0]
    acc_number[np.where(acc_number == 0)] = 1e5
    radial_average = np.divide(acc_weight, acc_number)
    #radial_average[np.where(radial_average == 0)] = None
    '''
    fig1 = Figure()
    fig1.suptitle('acc_weight')
    fig1.add_subplot(111).plot(acc_weight)
    RDF_TK_GUI.ImageWindow(fig1)

    fig2 = Figure()
    fig2.suptitle('acc_number')
    fig2.add_subplot(111).plot(acc_number)
    RDF_TK_GUI.ImageWindow(fig2)

    fig3 = Figure()
    fig3.suptitle('divide')
    fig3.add_subplot(111).plot(np.divide(acc_weight, acc_number))
    RDF_TK_GUI.ImageWindow(fig3)

    fig4 = Figure()
    fig4.suptitle('gragient')
    m = np.divide(acc_weight, acc_number)
    fig4.add_subplot(111).plot(m[0:-1]-m[1:])
    RDF_TK_GUI.ImageWindow(fig4)
    '''
    '''
    plt.subplot(311)
    plt.plot(acc_weight)
    plt.subplot(312)
    plt.plot(acc_number)
    plt.subplot(313)
    plt.plot(np.divide(acc_weight, acc_number))
    plt.show()
    '''
    return radial_average


def auto_beam_stopper(image, initial_point, alpha_low=0.05, alpha_high=0.9):
    if image is None:
        return None, None
    timer = time.time()
    fig = np.array(image, dtype=np.float)
    low_thres = fig.max() * float(alpha_low)
    high_thres = fig.max() * float(alpha_high)
    edges = canny(fig, sigma=1, low_threshold=low_thres, high_threshold=high_thres)

    edges = edges * 1
    edges_f = signal.convolve2d(edges, np.ones((5, 5)))[2:-2, 2:-2]
    edges_f = np.minimum(edges_f, 1)
    fill_in = fill_mask(edges_f, initial_point)
    fill_in = signal.convolve2d(fill_in, np.ones((21, 21)))[10:-10, 10:-10]
    fill_in = np.minimum(fill_in, 1)
    fill_in = 1 - fill_in
    fill_in[np.where(fill_in == 0)] = None
    filted_fig = np.multiply(fill_in, fig)
    print('auto_beam_stopper:', time.time()-timer)
    return filted_fig, fill_in


def intensity_average(filted_fig, initial_point):
    #timer = time.time()
    distance_mat = distance_matrix(filted_fig, initial_point)

    distance_vec = distance_mat.reshape((np.size(distance_mat), 1))[:, 0]
    filted_fig_vec = filted_fig.reshape((np.size(filted_fig), 1))[:, 0]
    max_r = distance_vec.max() + 1

    radius_average = annular_average(distance_vec, filted_fig_vec, int(np.floor(max_r)))
    #print('intensity_average:', time.time()-timer)
    return radius_average


def test_hough(fig):
    # doesn´t work
    a = time.time()
    small_fig_size = 300  # set image dimensions to max 600 x 600 px
    fig = np.array(fig, dtype=np.float)
    mass_x, mass_y = mean_center(fig)
    # Cuts image to specified size (600x600px) around the COM
    small_fig_x_low = np.maximum(0, mass_x-small_fig_size)
    small_fig_x_up = np.minimum(np.shape(fig)[0], mass_x+small_fig_size)
    small_fig_y_low = np.maximum(0, mass_y-small_fig_size)
    small_fig_y_up = np.minimum(np.shape(fig)[1], mass_y+small_fig_size)
    small_fig = fig[small_fig_x_low:small_fig_x_up, small_fig_y_low:small_fig_y_up]
    # mask cut figure to the 80th 
    num75 = np.percentile(small_fig.reshape((np.size(small_fig), 1)), [25, 50, 80])
    mask = small_fig * 0
    mask[np.where(small_fig > num75[2])] = 1

    #small_fig = np.maximum(small_fig-num75[2], 0)
    #small_fig = np.minimum(small_fig, 1)

    low_thres = 0.1 #small_fig.max() * 0.04
    high_thres = 0.9 #small_fig.max() * 0.05
    edges = canny(mask, sigma=1, low_threshold=low_thres, high_threshold=high_thres)

    index = np.where(edges)
    plt.imsave("edges.jpg",edges)
    plt.imsave("mask.jpg",mask)
    radii_ex = index[0].max() - index[0].min()
    radii_ex2 = index[1].max() - index[1].min()
    # Detect two radii
    hough_radii = np.arange(np.minimum(radii_ex, radii_ex2)/2, np.maximum(radii_ex, radii_ex2)/2, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    #hough_res = hough_circle(edges, radii)
    #accums, cx, cy, radii = hough_circle_peaks(hough_res, radii,
    #                                           total_num_peaks=1)
    print(cx, cy, radii)
    #plt.imshow(np.multiply(np.multiply(mask, small_fig),edges))
    #plt.show()
    x = small_fig_y_low + cx + 1
    y = small_fig_x_low + cy + 1
    return x[0], y[0], radii[0]

def test_hough_2(fig):
    """sobel filtering with hough transform"""
    
    #a = time.time()
    fig = np.array(fig, dtype=np.float)
    
    edges = sobel(fig)
    #plt.imsave("edges.png",edges[230:260,260:285])
    mask = edges * 0
    mask[np.where(edges > np.max(edges)*0.75)] = 1
    index = np.where(mask)
    rad_1 = index[0].max() - index[0].min()
    rad_2 = index[1].max() - index[1].min()
    rad=int(np.mean([rad_1,rad_2])/2)
    start = np.max([2,rad-5])
    stop=rad+5
    hough_radii = np.arange(start,stop)
    hough_res = hough_circle(mask, hough_radii)
    #for i in np.arange(start,stop):
    #    plt.imsave(str(i)+"hough.png",hough_res[i-start])
    # Select the most prominent circle
    circle=np.unravel_index(np.argmax(hough_res, axis=None), hough_res.shape)
    radius, x, y = circle[0]+start, circle[1],circle[2]
    print(x, y, radius)
    #print("edge detection in: ")
    #print((time.time()-a)*1000)
    return x, y, radius


def calc_center_map(data_4d):
    """
    Takes 4d data and calculates the center of the bf disk at each scanning position. 
    Returns arrays of x and y center positions.
    """
    
    x, y = int(data_4d.shape[0]/2),int(data_4d.shape[1]/2)
            
    x1, y1, r1 = test_hough_2(data_4d[0,0])
    x2, y2, r2 = test_hough_2(data_4d[0,-1])
    x3, y3, r3 = test_hough_2(data_4d[-1,0])
    x4, y4, r4 = test_hough_2(data_4d[-1,-1])
    x5, y5, r5 = test_hough_2(data_4d[x,y])
    
    center=[int(np.mean([x1,x2,x3,x4,x5])),int(np.mean([y1,y2,y3,y4,y5]))]
    r = round(np.mean([r1,r2,r3,r4,r5]))

    cut = 50
    s = cut + r
    data_4d_cut = data_4d[...,center[0]-s:center[0]+s,center[1]-s:center[1]+s]
    
    cx = np.zeros(data_4d_cut.shape[0:2])
    cy = np.zeros(data_4d_cut.shape[0:2])
    for i in np.arange(data_4d_cut.shape[0]):
        for j in np.arange(data_4d_cut.shape[1]):
            edges= sobel(data_4d_cut[i,j])
            mask=np.where(edges > np.max(edges)*0.75,1,0)
            hough_res = hough_circle(mask, r)
            circle=np.unravel_index(np.argmax(hough_res, axis=None), hough_res.shape)
            cx[i,j], cy[i,j] = circle[1],circle[2]
    cx += center[0]-s
    cy += center[1]-s
    #np.savetxt("cx.txt",cx,fmt='%1.0f')
    #np.savetxt("cy.txt",cy,fmt='%1.0f')
    return np.array([cx,cy]), r

def mean_center(fig):
    """calculates COM of a 2d image and returns center coordinates"""
    fig = np.array(fig, dtype=np.float)
    fig_x = np.sum(fig, axis=1)
    fig_y = np.sum(fig, axis=0)
    x = np.dot(fig_x, np.arange(1, np.size(fig_x)+1))
    weight = np.sum(fig_x)
    x = x/weight
    y = np.dot(fig_y, np.arange(1, np.size(fig_y)+1))
    y = y/weight
    
    return int(x), int(y)


def auto_find_center_hough(fig):
    "this function has been replaced by function test_hough"
    cx, cy, radii = test_hough(fig)
    return cx, cy, radii

    a = time.time()
    fig = np.array(fig, dtype=np.float)
    low_thres = fig.max()*0.05
    high_thres = fig.max()*0.1
    edges = canny(fig, sigma=4, low_threshold=low_thres, high_threshold=high_thres)

    # Detect two radii
    hough_radii = np.arange(150, 250, 2)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent circle
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                               total_num_peaks=1)
    print(time.time()-a)
    return cx, cy, radii


if __name__ == '__main__':
    path = 'ZrO2_Zr-Zr.txt'
    data = []
    with open(path, 'r') as a:
        while True:
            try:
                data.append(float(a.readline()))
            except:
                break
    file_name = 'test1'

    threshold = 990 #  512*1024*1024/10
    num_file = len(data) // threshold + 1

    with open('test2', 'w+') as f:
        np.savetxt(f, [0.00006], delimiter=',', fmt='%.4f')
    with open('test3', 'w+') as f:
        np.savetxt(f, [0.00006], delimiter=',', fmt='%.4e')
