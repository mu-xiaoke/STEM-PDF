import copy
import pickle
import tkinter as tk
import matplotlib
import numpy as np
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # do not move this
import RDF_Package
import RDF_preparation
import RDF_TK_GUI

class AnalysisWindow(tk.Toplevel):
    def __init__(self, saved_data, parent=None):
        super().__init__()
        self.title('Analysis window')
        self.data = saved_data
        self.plot_frame = None
        self.left_type = 'profile'
        self.plot_data = copy.deepcopy(self.data.aver_inten)
        self.left_x = 0
        self.left_y = 0

        self.int_amin = None
        self.int_amax = None

        self.parent = parent

        self.rdf_title = 'rdf'
        self.change_range_en = 0
        self.profile_title = 'profile'
        self.image_title = 'diffraction pattern'
        self.left_title = self.profile_title
        self.setupGUI()

    def setupGUI(self):
        button_width = 5
        button_height = 2
        row0 = tk.Frame(self)
        self.y_position = tk.Label(row0, text='0')
        self.y_position.pack(side=tk.RIGHT)
        tk.Label(row0, text='Y:').pack(side=tk.RIGHT)
        self.x_position = tk.Label(row0, text='0')
        self.x_position.pack(side=tk.RIGHT)
        row0.pack(side=tk.TOP, expand=1, fill='x')
        tk.Label(row0, text='X:').pack(side=tk.RIGHT)

        tk.Label(row0, text='scanning size:').pack(side=tk.LEFT)
        self.image_size_x_sv = tk.StringVar(self, value='10')
        self.image_size_x_entry = tk.Entry(row0, textvariable=self.image_size_x_sv, width=5)
        self.image_size_x_entry.pack(side=tk.LEFT)

        self.image_size_y_sv = tk.StringVar(self, value='10')
        self.image_size_y_entry = tk.Entry(row0, textvariable=self.image_size_y_sv, width=5)
        self.image_size_y_entry.pack(side=tk.LEFT)

        row01 = tk.Frame(self)
        row01.pack(side=tk.TOP, expand=1)
        row01_1 = tk.Frame(row01)
        row01_1.pack(side=tk.LEFT)
        tk.Button(row01_1, text='save', command=self.save, width=button_width,
                  height=button_height).pack(side=tk.LEFT)
        tk.Button(row01_1, text='load', command=self.load, width=button_width,
                  height=button_height).pack(side=tk.LEFT)

        row11 = tk.Frame(self)
        row11.pack(side=tk.TOP, expand=1)
        row11_1 = tk.Frame(row01)
        row11_1.pack(side=tk.RIGHT)
        self.button_profile = tk.Button(row11_1, text='profile', command=self.choose_profile,
                                        width=button_width, height=button_height)
        self.button_profile.pack(side=tk.LEFT)
        self.button_rdf = tk.Button(row11_1, text='rdf', command=self.choose_rdf,
                                    width=button_width, height=button_height)
        self.button_rdf.pack(side=tk.LEFT)
        self.button_image = tk.Button(row11_1, text='image', command=self.choose_image,
                                      width=button_width, height=button_height)
        self.button_image.pack(side=tk.LEFT)

        tk.Button(row11, text='recalculating rdf', command=self.recal_rdf).pack(side=tk.RIGHT)
        tk.Button(row11, text='update whole data', command=self.update_data).pack(side=tk.RIGHT)

        row12 = tk.Frame(self)
        row12.pack(side=tk.TOP, fill='x', expand=1)
        self.choose_range_int = tk.IntVar(value=0)
        self.choose_range_check = tk.Checkbutton(row12, text='define integration range',
                                                 variable=self.choose_range_int)
        self.choose_range_check.pack(side=tk.LEFT)
        tk.Button(row12, text='reset integration range', command=self.left_reset).pack(side=tk.LEFT)
        self.choose_square_int = tk.IntVar(value=0)
        self.choose_square_check = tk.Checkbutton(row12, text='choose integration square',
                                                  variable=self.choose_square_int)
        self.choose_square_check.pack(side=tk.RIGHT)

        row13 = tk.Frame(self)
        row13.pack(side=tk.TOP, fill='x', expand=1)

        tk.Button(row13, text='set boundary', command=self.set_boundary).pack(side=tk.RIGHT)

        self.upper_boundary_string = tk.StringVar()
        self.ubs = tk.Entry(row13, textvariable=self.upper_boundary_string, width=5)
        self.ubs.pack(side=tk.RIGHT)
        tk.Label(row13, text='upper boundary').pack(side=tk.RIGHT)

        self.lower_boundary_string = tk.StringVar()
        self.lbs = tk.Entry(row13, textvariable=self.lower_boundary_string, width=5)
        self.lbs.pack(side=tk.RIGHT)
        tk.Label(row13, text='lower boundary').pack(side=tk.RIGHT)

        tk.Label(row13, text='upper threshold').pack(side=tk.LEFT)
        self.lower_threshold = tk.StringVar()
        self.image_lbs = tk.Entry(row13, textvariable=self.lower_threshold, width=5)
        self.image_lbs.pack(side=tk.LEFT)
        tk.Label(row13, text='lower threshold').pack(side=tk.LEFT)
        self.upper_threshold = tk.StringVar()
        self.image_ubs = tk.Entry(row13, textvariable=self.upper_threshold, width=5)
        self.image_ubs.pack(side=tk.LEFT)

        tk.Button(row13, text='image threshold', command=self.image_threshold).pack(side=tk.LEFT)

        row2 = tk.Frame(self)
        row2.pack(fill=tk.BOTH, expand=1)
        r2left = tk.Frame(row2)
        r2left.pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
        r2right = tk.Frame(row2)
        r2right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=0)
        self.left_fig = Figure()
        self.left_fig.suptitle('profile')
        self.left_frame = self.left_fig.add_subplot(111)
        self.left_frame.plot(self.data.aver_inten[:, 0, 0])
        self.right_fig = Figure()
        self.right_frame = self.right_fig.add_subplot(111)
        ax = self.right_frame.imshow(sum(self.plot_data, 0))
        self.right_fig.colorbar(ax)

        self.left_canvas = self.build_canvas(master_frame=r2left, plot_frame=self.left_fig)
        self.left_canvas.mpl_connect('button_press_event', handlerAdaptor(self.user_choose_inter_range))
        self.right_canvas = self.build_canvas(master_frame=r2right, plot_frame=self.right_fig)
        self.right_canvas.mpl_connect('button_press_event', handlerAdaptor(self.user_change_left_data))

    def build_canvas(self, master_frame, plot_frame=None):
        if plot_frame is None:
            plot_frame = self.update_plot_frame(y=self.plot_data[:, 0, 0])
        canvas = FigureCanvasTkAgg(plot_frame, master=master_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, master_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def user_change_left_data(self, event):
        if self.choose_square_int.get():
            if not hasattr(self, 'square_x') or self.square_x is None:
                try:
                    self.square_x = round(event.xdata)
                    self.square_y = round(event.ydata)
                except:
                    pass
                return
            else:
                try:
                    square_range = [np.minimum(self.square_x, event.xdata),
                                    np.minimum(self.square_y, event.ydata),
                                    np.maximum(self.square_x, event.xdata),
                                    np.maximum(self.square_y, event.ydata)]
                except:
                    pass
            if self.left_title == self.image_title:
                self.update_left_image(square_range)
            else:
                self.update_left_plot(square_range)
            self.square_x = None
            self.square_y = None
            self.choose_square_int.set(0)
            return

        try:
            self.left_x = round(event.xdata)
            self.left_y = round(event.ydata)
            self.x_position.config(text=self.left_x)
            self.y_position.config(text=self.left_y)
            if self.left_title == self.image_title:
                self.update_left_image()
            else:
                self.update_left_plot()
        except TypeError:
            pass

    def user_choose_inter_range(self, event):
        if self.choose_range_int.get():
            if self.left_title == self.image_title:
                if not hasattr(self, 'image_range_x_1') or self.image_range_x_1 is None:
                    self.image_range_x_1 = round(event.xdata)
                    self.image_range_y_1 = round(event.ydata)
                else:
                    self.image_inte_range = [np.minimum(self.image_range_x_1, round(event.xdata)),
                                             np.minimum(self.image_range_y_1, round(event.ydata)),
                                             np.maximum(self.image_range_x_1, round(event.xdata)),
                                             np.maximum(self.image_range_y_1, round(event.ydata))]

                    self.update_right_image()
                    self.image_range_x_1 = None
                    self.image_range_y_1 = None
                    self.choose_range_int.set(0)
                return

        if self.choose_range_int.get():
            self.inter_range_conut()

            if self.count == 1:
                self.update_left_plot()
            self.left_ymax = np.amax(self.plot_data[:, self.left_y, self.left_x])
            self.left_ymin = np.amin(self.plot_data[:, self.left_y, self.left_x])

            xposition = event.xdata
            if self.data.angstrom is not None and self.left_title == self.rdf_title:
                xposition = np.maximum(self.data.angstrom[0], xposition)
                xposition = np.minimum(self.data.angstrom[-1], xposition)
            else:
                xposition = np.maximum(0, xposition)
                xposition = np.minimum(np.size(self.plot_data[:, 0, 0]), xposition)

            if self.count == 0:
                self.change_range_en = 1
                self.int_amin = np.minimum(self.previous_x, xposition)
                self.int_amax = np.maximum(self.previous_x, xposition)
                self.choose_range_int.set(0)
                self.plot_left_frame_base()
                self.update_left_plot()
                self.update_right_plot()

            else:
                self.left_fig.clf()
                self.left_frame = self.left_fig.add_subplot(111)
                self.plot_left_frame_base()
                self.previous_x = xposition
                self.left_frame.plot([xposition, xposition], [self.left_ymin, self.left_ymax], 'r')
                self.left_canvas.draw()

    def left_reset(self, all_frame=1):
        self.count = 0
        self.change_range_en = 0
        self.int_amin = None
        self.int_amax = None
        self.left_ymin = None
        self.left_ymax = None
        if all_frame:
            self.update_left_plot()
            self.update_right_plot()

    def range_mask(self):
        self.left_ymax = np.amax(self.plot_data[:, self.left_y, self.left_x])
        self.left_ymin = np.amin(self.plot_data[:, self.left_y, self.left_x])
        step = (self.int_amax - self.int_amin)/100
        a = np.arange(self.int_amin, self.int_amax, step)
        self.left_frame.fill_between(a, self.left_ymin, self.left_ymax, alpha=0.2, color='r')
        self.left_frame.plot([self.int_amin, self.int_amin], [self.left_ymin, self.left_ymax], 'r')
        self.left_frame.plot([self.int_amax, self.int_amax], [self.left_ymin, self.left_ymax], 'r')

    def inter_range_conut(self):
        if not hasattr(self, 'count'):
            self.count = 0
        self.count += 1
        if self.count > 1:
            self.count = 0

    def update_plot_frame(self, x=None, y=None, image=None):
        plot_frame = plt.figure()
        if y is not None:
            if x is not None:
                plt.plot(x, y)
            else:
                plt.plot(y)
        if image is not None:
            plt.imshow(image)
        return plot_frame

    def choose_profile(self):
        self.left_reset(all_frame=0)
        self.left_title = self.profile_title
        self.plot_data = copy.deepcopy(self.data.aver_inten)
        self.update_left_plot()
        #self.right_frame.imshow(sum(self.plot_data, 0))
        self.update_right_plot()

    def choose_G_corrected(self):
        pass

    def choose_rdf(self):
        self.left_reset(all_frame=0)
        self.left_title = self.rdf_title
        self.plot_data = copy.deepcopy(self.data.rdf_data)
        self.update_left_plot()
        #self.right_frame.imshow(sum(self.plot_data, 0))
        self.update_right_plot()

    def choose_image(self):
        self.left_reset(all_frame=0)
        self.left_title = self.image_title
        self.update_left_image()
        self.update_right_image()
        self.image_map_exist = 1

    def plot_left_frame_base(self, plot_data=None):
        self.left_fig.clf()
        self.left_fig.suptitle(self.left_title)
        self.left_frame = self.left_fig.add_subplot(111)
        if plot_data is None:
            plot_data = self.plot_data[:, self.left_y, self.left_x]
        if self.data.angstrom is not None and self.left_title == self.rdf_title:
            self.left_frame.plot(self.data.angstrom, plot_data)
        else:
            self.left_frame.plot(plot_data)

    def update_left_plot(self, square_range=None):
        if square_range is None:
            self.left_plot_data = self.plot_data[:, self.left_y, self.left_x]
        else:
            image = self.plot_data[:, self.left_y, self.left_x] * 0
            for x in np.arange(square_range[0], square_range[2]+1):
                for y in np.arange(square_range[1], square_range[3]+1):
                    image += self.plot_data[:, int(y), int(x)]
            self.left_plot_data = image
        self.plot_left_frame_base(self.left_plot_data)
        if self.change_range_en:
            self.range_mask()
        self.left_canvas.draw()

    def update_right_plot(self):
        self.right_fig.clf()
        begin = self.int_amin
        end = self.int_amax
        if begin is not None and end is not None:
            if self.left_title == self.rdf_title and self.data.angstrom is not None:
                begin = round(begin/self.data.angstrom[1])
                end = round(end/self.data.angstrom[1])
            else:
                begin = round(begin)
                end = round(end)

        if begin is not None and end is not None:
            self.right_image = sum(self.plot_data[begin:end+1, :, :], 0)
        else:
            self.right_image = sum(self.plot_data, 0)
        self.set_boundary()

    def update_left_image(self, square_range=None):
        num = self.left_y*self.data.shape[1] + self.left_x
        if square_range is None:
            image = self.parent.mib_data._frames(1, num)[0, :, :]
        else:
            image = np.array(self.parent.mib_data._frames(1, 0)[0, :, :], dtype='float64')

            for x in np.arange(square_range[0], square_range[2]+1):
                for y in np.arange(square_range[1], square_range[3]+1):
                    num = int(y * self.data.shape[1] + x)
                    new_image = np.array(self.parent.mib_data._frames(1, num)[0, :, :])
                    image += new_image

        self.left_fig.clf()
        self.left_fig.suptitle(self.left_title)

        self.left_frame = self.left_fig.add_subplot(111)
        self.left_image = np.array(image)
        ax = self.left_frame.imshow(image)
        self.left_fig.colorbar(ax)
        self.left_canvas.draw()

    def update_right_image(self):
        if hasattr(self, 'image_inte_range'):
            x1 = self.image_inte_range[0]
            y1 = self.image_inte_range[1]
            x2 = self.image_inte_range[2]
            y2 = self.image_inte_range[3]
        else:
            x1 = 0
            y1 = 0
            x2 = int(self.image_size_x_entry.get())-1
            y2 = int(self.image_size_y_entry.get())-1
        x_size = int(self.image_size_x_entry.get())
        y_size = int(self.image_size_y_entry.get())
        data_cube = self.parent.mib_data._frames(x_size*y_size, 0)[:, y1:y2+1, x1:x2+1]
        image = np.sum(data_cube, 1)
        image = np.sum(image, 1)
        self.right_image = image.reshape((x_size, y_size))
        self.right_fig.clf()
        self.right_frame = self.right_fig.add_subplot(111)
        ax = self.right_frame.imshow(self.right_image)
        self.right_fig.colorbar(ax)
        self.right_canvas.draw()

    def recal_rdf(self):
        if self.left_title == self.profile_title or self.left_title == self.rdf_title:
            plot_dict = RDF_Package.rdf_cal(self.parent.RDF_parameter, data_trans=1)
            plot_dict['aver_inten'] = self.left_plot_data
            RDF_TK_GUI.RDFWindow(plot_dict=plot_dict, parent=self,
                                 rdfpara_dict=self.parent.rdf_window_return_info)

    def update_data(self):
        if hasattr(self, 'rdf_window_return_info'):
            parameter_dict = copy.deepcopy(self.parent.RDF_parameter)
            for keys in self.rdf_window_return_info.keys():
                parameter_dict.__setattr__(keys, self.rdf_window_return_info[keys])
            step = parameter_dict.L/parameter_dict.rn
            angstrom = np.arange(0, parameter_dict.L, step)
            self.data = batch_rdf_whole(self.plot_data, self.data.shape,
                                        inten_length=np.shape(self.data.aver_inten)[0],
                                        rdf_length=np.shape(self.data.rdf_data)[0],
                                        angstrom=angstrom,
                                        para=parameter_dict, recal=1)
            self.choose_profile()

    def save(self):
        path = asksaveasfilename(filetypes=(("csv File", "*.csv"), ("All Files", "*.*")))
        file = open(path, 'wb+')
        pickle.dump(self.data, file)

    def load(self):
        path = askopenfilename(filetypes=(("csv File", "*.csv"), ("All Files", "*.*")))
        file = open(path, 'rb+')
        self.data = pickle.load(file)
        self.choose_profile()

    def set_boundary(self):
        self.right_fig.clf()
        self.right_frame = self.right_fig.add_subplot(111)
        try:
            upper = float(self.ubs.get())
            lower = float(self.lbs.get())
            if upper >=lower:
                right_image = np.maximum(self.right_image, lower)
                right_image = np.minimum(right_image, upper)
            else:
                right_image = self.right_image
            ax = self.right_frame.imshow(right_image)
        except:
            ax = self.right_frame.imshow(self.right_image)
        self.right_fig.colorbar(ax)
        self.right_canvas.draw()

    def image_threshold(self):
        if self.left_title == self.image_title:
            self.left_fig.clf()
            self.left_frame = self.left_fig.add_subplot(111)
            try:
                upper = float(self.image_ubs.get())
                lower = float(self.image_lbs.get())
                left_image = np.maximum(self.left_image, lower)
                left_image = np.minimum(left_image, upper)
                ax = self.left_frame.imshow(left_image)
            except:
                ax = self.left_frame.imshow(self.left_image)
                self.left_fig.suptitle(self.left_title)
            self.left_fig.colorbar(ax)
            self.left_canvas.draw()


def round(num):
    return int(num+0.5)


def batch_rdf_single(frame, center, beam_stopper=None, para=None, recal=0):
    if recal:
        para.aver_inten = frame
        rdf_data = RDF_Package.rdf_cal(para)
        return rdf_data

    if beam_stopper is not None:
        frame = np.multiply(frame, beam_stopper)
    aver_inten = RDF_preparation.intensity_average(frame, center)
    if para is not None:
        para.aver_inten = aver_inten
        rdf_data = RDF_Package.rdf_cal(para)
    else:
        rdf_data = None
    return aver_inten, rdf_data


def batch_rdf_whole(dataset, shape, inten_length=None, rdf_length=None, angstrom=None,
                    auto_center=0, center=None, beam_stopper=None, para=None, recal=0):
    if recal:
        num = int(np.size(dataset)/np.shape(dataset)[0])
        new_data = DataBank(shape, inten_length, rdf_length, angstrom)
        dataset_vec = dataset.reshape((np.shape(dataset)[0], num))

        for i in np.arange(num):
            frame = dataset_vec[:, i]
            rdf_data = batch_rdf_single(frame, None, None, para, recal)
            new_data.set_data(i, center=center, aver_inten=frame, rdf_data=rdf_data)
        return new_data

    if inten_length is None or rdf_length is None:
        if para is not None:
            center = RDF_preparation.test_hough(dataset[0, :, :])
            aver_inten, rdf_data = batch_rdf_single(dataset[0, :, :], center, beam_stopper, para)
            inten_length = np.size(aver_inten)
            rdf_length = np.size(rdf_data['G_corrected'])
        else:
            aver_inten, rdf_data = batch_rdf_single(dataset[0, :, :], center, beam_stopper, para)
            inten_length = np.size(aver_inten)

    saved_data = DataBank(shape, inten_length, rdf_length, angstrom)
    for i in np.arange(np.shape(dataset)[0]):
        frame = dataset[i, :, :]
        if auto_center:
            center = RDF_preparation.test_hough(frame)[:, 0]
        aver_inten, rdf_data = batch_rdf_single(frame, center, beam_stopper, para)
        saved_data.set_data(i, center=center, aver_inten=aver_inten, rdf_data=rdf_data)
    return saved_data


class DataBank:
    def __init__(self, shape, inten_length, rdf_length, angstrom=None):
        self.shape = shape
        self.center = np.zeros((2, shape[0], shape[1]))
        self.aver_inten = np.zeros((inten_length, shape[0], shape[1]))
        if rdf_length is not None:
            self.rdf_data = np.zeros((rdf_length, shape[0], shape[1]))
        else:
            self.rdf_data = None
        self.angstrom = angstrom

    def set_data(self, i, center=None, aver_inten=None, rdf_data=None,  *args, **kwargs):
        x = int(np.floor(i / self.shape[1]))
        y = i - x * self.shape[1]
        self.aver_inten[:, x, y] = aver_inten
        if center is not None:
            self.center[:, x, y] = [center[0], center[1]]
        if rdf_data is not None:
            self.rdf_data[:, x, y] = rdf_data['G_corrected']


def handlerAdaptor(fun, **kwargs):
    return lambda event, fun=fun, kwds=kwargs: fun(event, **kwds)


if __name__ == '__main__':
    pass