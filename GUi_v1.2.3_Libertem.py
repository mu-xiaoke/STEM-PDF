import copy
import time
import pickle
import matplotlib
import pandas
import tkinter as tk
import numpy as np
import scipy.io as sio
from tkinter.filedialog import askopenfilename, asksaveasfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from sympy import Ray3D, false
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # do not move this
from PIL import Image, ImageDraw
import RDF_preparation
import RDF_Package
import read_mib
import simulation
import mrcfile
#from read_empad import read_empad
try:
    import hyperspy.api as hs
except:
  print("HyperSpy is missing")
  
try:
    import libertem.api as lt
except:
  print("Libertem is missing")
  
global data_4d, global_radius, ctx

class MainGui(tk.Tk):
    def __init__(self):
        super().__init__()

        self.replot_profile = 0

        self.title('RDF Calculator')
        "initialization of variable"
        self.command_button = None
        self.pre_frame = None
        self.front_page = 'read_part'
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()

        self.geometry("%dx%d+%d+%d" % (0.8*width, 0.8*height, 0.1*width, 0.1*height))
        self.command_frame = tk.Frame(self, width=200, height=400, background='gray')
        self.command_frame.pack(side=tk.LEFT)
        self.base_frame = tk.Frame(self)
        self.base_frame.pack(side=tk.RIGHT, fill='both', expand=1)

        self.disp_frame = tk.Frame(self.base_frame, width=100, height=400, background='black')
        self.disp_frame.pack(side=tk.RIGHT, fill='both', expand=1)
        self.disp_frame.rowconfigure(0, weight=100)
        self.disp_frame.columnconfigure(0, weight=100)
        self.frame = {}

        for func in (RdfFrame,
                     PreProcessingFrame,
                     RdfMapFrame,
                     ProfileMapFrame,
                     SimulationFrame):
            frame = func(self.disp_frame, self)
            self.frame[func] = frame
        self.setup_command_frame(self.command_frame)

        self.pre_process()

    def setup_command_frame(self, command_frame):
        self.command_button = CommandButton(self, command_frame)
        #self.command_button.add_new_button('read_part', text='read data')
        self.command_button.add_new_button('pre_process', text='pre processing')
        self.command_button.add_new_button('profile_map', text='profile map')
        self.command_button.add_new_button('rdf_part', text='RDF')
        self.command_button.add_new_button('map_part', text='map viewer')
        self.command_button.add_new_button('simulation_part', text='simulation')
        #self.command_button.add_new_button('compare_part', text='comparing')

    def change_disp_frame(self, frame_name):
        self.forget_frame()
        self.front_page = frame_name
        print(frame_name)

    def rdf_part(self):
        self.front_page = 'rdf_part'
        self.forget_frame()
        self.frame[RdfFrame].pack(fill='both', expand=1)
        print('rdf_part')

    def pre_process(self):
        self.front_page = 'pre_process'
        self.forget_frame()
        self.frame[PreProcessingFrame].pack(fill='both', expand=1)
        print('pre_process')
        if self.replot_profile:
            self.frame[PreProcessingFrame].plot_map()

    def profile_map(self):
        self.front_page = 'profile_map'
        self.forget_frame()
        self.frame[ProfileMapFrame].pack(fill='both', expand=1)
        if self.replot_profile:
            self.frame[ProfileMapFrame].plot_map(self)
        print('profile map')

    def map_part(self):
        self.front_page = 'map_part'
        self.forget_frame()
        self.frame[RdfMapFrame].pack(fill='both', expand=1)
        if self.replot_profile:
            self.frame[RdfMapFrame].plot_map()
        print('map_part')

    def simulation_part(self):
        print('simulation_part')
        self.forget_frame()
        self.frame[SimulationFrame].pack(fill='both', expand=1)
        print('map_part')
        pass

    def compare_part(self):
        print('compare_part')
        pass

    def forget_frame(self):
        for frame in self.frame.values():
            frame.pack_forget()

    def pre_return_info(self, pre_dict_info=None):
        try:
            self.frame[ProfileMapFrame].path = pre_dict_info['path']
            self.frame[ProfileMapFrame].mode = pre_dict_info['path'][-3:]
            self.frame[ProfileMapFrame].center = pre_dict_info['center']
            self.frame[ProfileMapFrame].sv_center_x.set('{0:.2f}'.format(pre_dict_info['center'][0]))
            self.frame[ProfileMapFrame].sv_center_y.set('{0:.2f}'.format(pre_dict_info['center'][1]))
            self.frame[ProfileMapFrame].sv_radius.set('{0:.2f}'.format(pre_dict_info['radius']))
            self.frame[ProfileMapFrame].sv_shape_x.set(pre_dict_info['scanning_shape'][0])
            self.frame[ProfileMapFrame].sv_shape_y.set(pre_dict_info['scanning_shape'][1])
            self.frame[ProfileMapFrame].process_size = pre_dict_info['process_size']
            self.frame[ProfileMapFrame].beam_stopper = pre_dict_info['beam_stopper']
            self.frame[ProfileMapFrame].shape_beam_stopper = np.shape(pre_dict_info['beam_stopper'])
            self.frame[ProfileMapFrame].delete_mask = pre_dict_info['delete_mask']
            self.frame[ProfileMapFrame].center_map = pre_dict_info['center_map']
            self.frame[ProfileMapFrame].calculate_profile_map(pre_dict_info['process_area'])
        except TypeError:
            pass

    def profile_map_return_info(self, mode=1, dict_info=None):
        self.frame[RdfFrame].read_profile(path=0, profile=dict_info['profile'])
        self.frame[RdfMapFrame].get_dict_info(dict_info)

    def rdf_return_info(self, para_dict):
        self.frame[RdfMapFrame].rdfpara_dict = para_dict
        #try:
        self.frame[RdfMapFrame].calculate_rdf_map()
        self.map_part()
    #    except:
    #        pass

    def rdf_return_position(self, position1, position2):
        if self.frame[PreProcessingFrame].to_be_process_area is None:
            self.frame[PreProcessingFrame].click_position1 = position1
            self.frame[PreProcessingFrame].click_position2 = position2
        else:
            _po1, _po2 = swap_position(self.frame[PreProcessingFrame].area_1,
                                       self.frame[PreProcessingFrame].area_2)
            self.frame[PreProcessingFrame].click_position1 = [position1[0]+_po1[0], position1[1]+_po1[1]]
            self.frame[PreProcessingFrame].click_position2 = [position2[0]+_po1[0], position2[1]+_po1[1]]
        print("prepart", self.frame[PreProcessingFrame].click_position1,
              self.frame[PreProcessingFrame].click_position2)
        self.frame[ProfileMapFrame].click_position1 = position1
        self.frame[ProfileMapFrame].click_position2 = position2
        self.frame[RdfMapFrame].click_position1 = position1
        self.frame[RdfMapFrame].click_position2 = position2
        self.replot_profile = 1


class PreProcessingFrame(tk.Frame):
    def __init__(self, frame, parent):
        super().__init__(frame)
        self.parent = parent
        self.plot_frame = frame

        self.path = None
        self.data_type = None

        self.center = [0, 0]
        self.radius = 25
        
        global global_radius
        global_radius = self.radius
                
        self.center_fig = None
        self.beam_stopper = None
        self.no_bs_image = None
        self.map_axes = None
        self.fig_map = None
        self.test_center = [0, 0]
        self.profile = None
        self.center_map = None
        self.click_count = 0
        self.click_position1 = [0, 0]
        self.click_position2 = [0, 0]

        self.delete_mask = None
        self.to_be_process_area = None
        self.area_count = 0
        self.area_1 = None
        self.area_2 = None

        self.click_diffraction_1 = [0, 0]
        self.click_diffraction_2 = [0, 0]
        self.diffraction_count = 0
        self.replot_map = 0
        self.button_frame = tk.Frame(self)
        self.button_frame.pack(side=tk.LEFT)
        self.frame_right = tk.Frame(self)
        self.frame_right.pack(side=tk.RIGHT, fill='both', expan=1)

        self.disp_frame = tk.Frame(self.frame_right)
        self.disp_frame.pack(side=tk.BOTTOM, fill='both', expand=1)
        self.setup_button(self.button_frame)
        self.setup_disp()

        self.frame_tool = tk.Frame(self.frame_right)
        self.frame_tool.pack(side=tk.TOP, fill='x', expand=1)
        self.frame_tool_left = tk.Frame(self.frame_tool)
        self.frame_tool_left.pack(side=tk.LEFT)

        self.setup_tool(self.frame_tool_left)
        self.frame_tool_right = tk.Frame(self.frame_tool)
        self.frame_tool_right.pack(side=tk.RIGHT)

        self.frame_delete_pixel = tk.Frame(self.frame_tool_right)
        self.frame_delete_pixel.grid(row=0, column=0)
        self.set_delete_pixel_frame(self.frame_delete_pixel)

        self.buttonset_frame = tk.Frame(self.frame_tool_right)
        self.buttonset_frame.grid(row=0, column=3)
        self.set_button_frame(self.buttonset_frame)

        self.buttonset_frame2 = tk.Frame(self.frame_tool_right)
        self.buttonset_frame2.grid(row=0, column=2)
        self.set_button2_frame(self.buttonset_frame2)

    def set_delete_pixel_frame(self, frame):
        tk.Label(frame, text='delete pixels').grid(row=0, column=0)
        self.delete_pixel_num_sv = tk.StringVar(value=0)
        tk.Entry(frame, textvariable=self.delete_pixel_num_sv, width=3).grid(row=0, column=1)

    def set_button_frame(self, frame):
        self.mode3_int = tk.IntVar(value=0)
        tk.Checkbutton(frame,
                       variable=self.mode3_int).grid(row=0, column=0)
        tk.Label(frame, text='set calculation area\nin right map').grid(row=0, column=1)

        self.mode2_int = tk.IntVar(value=0)
        tk.Checkbutton(frame, variable=self.mode2_int).grid(row=1, column=0)
        tk.Label(frame, text='set integration area\nin right map').grid(row=1,  column=1)

        self.mask2_int = tk.IntVar(value=0)
        tk.Checkbutton(frame, variable=self.mask2_int, command=self.plot_map).grid(row=2, column=0)

        tk.Label(frame, text='set integration area\n in diffraction pattern').grid(row=2, column=1)

    def set_button2_frame(self,frame):
        tk.Button(frame, text='histogram of diffraction',
                  command=lambda fun=self.hist_fun, mode=1: fun(mode)).pack(side=tk.TOP)
        tk.Button(frame, text='histogram of map',
                  command=lambda fun=self.hist_fun, mode=2: fun(mode)).pack(side=tk.TOP)

    def setup_button(self, frame):

        tk.Label(frame, text='mapping size:').pack()
        row0 = tk.Frame(frame)
        row0.pack(fill='x', expand=1)
        self.sv_map_y = tk.StringVar(value='0')
        self.entry_map_y = tk.Entry(row0, textvariable=self.sv_map_y, width=4)
        self.entry_map_y.grid(row=0, column=0)

        tk.Label(row0, text=',').grid(row=0, column=1)
        self.sv_map_x = tk.StringVar(value='0')
        self.entry_map_x = tk.Entry(row0, textvariable=self.sv_map_x, width=4)
        self.entry_map_x.grid(row=0, column=2)

        self.button_import = tk.Button(frame, text='import\n 4d raw data', command=self.import_data)
        self.button_import.pack()

        self.button_auto_center = tk.Button(frame, text='auto \nfind center',
                                            command=lambda fun=self.find_center, mode='auto': fun(mode))
        self.button_auto_center.pack()

        # center mapping button
        #lambda?
        self.button_center_map = tk.Button(frame, text='map \ncenter position',
                                            command=lambda fun=self.calc_center_map: fun())
        self.button_center_map.pack()
        
        self.button_save_center_map = tk.Button(frame, text='save \ncenter map', command=self.save_center_map)
        self.button_save_center_map.pack()
        self.button_load_center_map = tk.Button(frame, text='load \ncenter map', command=self.load_center_map)
        self.button_load_center_map.pack()

        
        row_set_center = tk.Frame(frame)
        row_set_center.pack(fill='x', expand=1)

        row_set_left = tk.Frame(row_set_center)
        row_set_left.pack(fill='x', expand=1, side=tk.LEFT)

        row10_R = tk.Frame(row_set_center)
        row10_R.pack(expand=1, fill='both')

        row1 = tk.Frame(row_set_left)
        row1.pack()

        tk.Label(row1, text='X: ').pack(side=tk.LEFT)
        self.sv_center_x = tk.StringVar(value='0.00')
        self.label_center_x = tk.Entry(row1, textvariable=self.sv_center_x, width=5)
        self.label_center_x.pack(side=tk.RIGHT, fill='x', expand=1)

        row2 = tk.Frame(row_set_left)
        row2.pack(fill='x', expand=1)
        tk.Label(row2, text='Y: ').pack(side=tk.LEFT)
        self.sv_center_y = tk.StringVar(value='0.00')
        self.label_center_y = tk.Entry(row2, textvariable=self.sv_center_y, width=5)
        self.label_center_y.pack(side=tk.RIGHT, fill='x', expand=1)

        row21 = tk.Frame(frame)
        row21.pack(fill='x', expand=1)
        tk.Label(row21, text='R: ').pack(side=tk.LEFT)
        self.sv_radius = tk.StringVar(value='0.00')
        self.label_radius = tk.Entry(row21, textvariable=self.sv_radius, width=5)
        self.label_radius.pack(side=tk.RIGHT, fill='x', expand=1)

        self.button_manual_give_center = tk.Button(row10_R, text='set',
                                                 command=lambda fun=self.find_center, mode='given': fun(mode))
        self.button_manual_give_center.pack(fill='both', expand=1)
        self.button_auto_beam_stopper = tk.Button(frame, text='beam stopper',
                                                  command=self.find_beam_stopper)
        self.button_auto_beam_stopper.pack()
        row3 = tk.Frame(frame)
        row3.pack(fill='x', expand=1)
        self.entry_beam_stopper_lower = tk.Entry(row3, textvariable=tk.StringVar(value=0.004), width=5)
        self.entry_beam_stopper_lower.pack(side=tk.LEFT)
        self.entry_beam_stopper_upper = tk.Entry(row3, textvariable=tk.StringVar(value=0.050), width=5)
        self.entry_beam_stopper_upper.pack(side=tk.LEFT)
        #tk.Button(frame, text='export\n beam stoppter', command=self.save_beam_stopper).pack()

        tk.Button(frame, text='calculate profile',
                  command=self.calculate_profile).pack()
        tk.Button(frame, text='save\ncurrent profile', command=self.save_current_profile).pack()
        tk.Button(frame, text='calculate\nprofile cube',
                  command=self.calculate_cube).pack()

        #tk.Button(frame, text='transfer\ndata', command=self.transfer_data).pack()

    def setup_disp(self):
        self.disp_figure = Figure()
        self.disp_figure.set_size_inches(5, 9)
        a = self.disp_figure.add_subplot(111)
        canvas_disp = FigureCanvasTkAgg(self.disp_figure, master=self.disp_frame)
        canvas_disp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_disp, self.disp_frame)
        toolbar.update()
        canvas_disp._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_disp.draw()
        self.canvas_disp = canvas_disp
        self.canvas_disp.mpl_connect('button_press_event', handlerAdaptor(self.update_figure))
        self.canvas_disp.mpl_connect('key_press_event', handlerAdaptor(self.press_key))

    def setup_tool(self, frame):
        row1 = tk.Frame(frame)
        row1.pack(fill='x', expand=1)
        self.scale_value_min = tk.StringVar(value='0.00')
        self.scale_right = tk.Scale(row1, orient=tk.HORIZONTAL,
                                    resolution=0.01,
                                    length=200, sliderlength=20, 
                                    variable=self.scale_value_min,
                                    showvalue= 0)
        self.scale_right.bind("<ButtonRelease-1>", self.plot_map)
        self.scale_right.pack(side=tk.RIGHT)
        tk.Label(row1, text='Min Intensity:').pack(side=tk.LEFT)
        
        self.scale_value_entry_min = tk.Entry(row1, textvariable=self.scale_value_min, width=5)
        self.scale_value_entry_min.pack(side=tk.LEFT)
        tk.Label(row1, text='%').pack(side=tk.LEFT)


        row2 = tk.Frame(frame)
        row2.pack(fill='x', expand=1)
        self.scale_value_max = tk.StringVar(value='0.00')
        self.scale_left = tk.Scale(row2, orient=tk.HORIZONTAL,
                                   resolution=0.01,
                                   length=200, sliderlength=20, 
                                   variable=self.scale_value_max,
                                   showvalue=0)
        self.scale_left.bind("<ButtonRelease-1>", self.plot_map)
        self.scale_left.pack(side=tk.RIGHT)
        tk.Label(row2, text='Max Intensity:').pack(side=tk.LEFT)

        self.scale_value_entry_max = tk.Entry(row2, textvariable=self.scale_value_max, width=5)
        self.scale_value_entry_max.pack(side=tk.LEFT)
        tk.Label(row2, text='%').pack(side=tk.LEFT)
        row3 = tk.Frame(frame)
        row3.pack(fill='x', expand=1)

        tk.Button(row3, text='plot', command=self.plot_map).pack()

    def update_image(self, fig, fig_map=None, center=None, radius=None, title='diffraction pattern', mode=1,
                     fig2=None, title2=None):
        self.center_fig = fig
        if mode == 0:
            if fig is not None:
                self.disp_figure.clf()
                if fig_map is None:
                    self.diff_axes = self.disp_figure.add_subplot(111, title=title)
                else:
                    self.diff_axes = self.disp_figure.add_subplot(121, title=title)
                    self.map_axes = self.disp_figure.add_subplot(122, title='diffraction mapping')
                    self.map_axes.imshow(fig_map)
                self.diff_axes.imshow(fig)
                self.canvas_disp.draw()

        elif mode == 1:
            if fig is not None:
                mask = np.zeros(np.shape(fig))
                x = int(center[0]+0.5)
                y = int(center[1]+0.5)
                mask[y-1:y+2, :] = 1
                mask[:, x-1:x+2] = 1
                mask = Image.fromarray(np.uint8(mask))
                draw = ImageDraw.Draw(mask)
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=1)
                draw.ellipse([x - (radius-1), y - (radius-1), x + (radius-1), y + (radius-1)], outline=1)
                mask = np.array(mask)
                mask2 = np.full(np.shape(mask), np.nan)
                mask2[np.where(mask == 1)] = 1
                cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'red'], 256)

                self.disp_figure.clf()
                if fig_map is None:
                    self.diff_axes = self.disp_figure.add_subplot(111, title=title)
                else:
                    self.diff_axes = self.disp_figure.add_subplot(121, title=title)
                    self.map_axes = self.disp_figure.add_subplot(122, title='diffraction mapping')
                    self.map_axes.imshow(fig_map)
                self.diff_axes.imshow(fig)
                self.diff_axes.imshow(mask2, cmap=cmap1)
                self.diff_axes.imshow(fig, alpha=0)
                self.canvas_disp.draw()

        elif mode == 2:
            if fig is not None and fig2 is not None:
                self.disp_figure.clf()
                self.disp_figure.add_subplot(121, title=title).imshow(fig)
                self.disp_figure.add_subplot(122, title=title2).imshow(fig2)
                self.canvas_disp.draw()

    def update_plot(self, array, title=None):
        if array is not None:
            self.disp_figure.clf()
            self.disp_figure.add_subplot(111, title=title).plot(array)
            self.canvas_disp.draw()

    def plot_map(self, event=None, max=0, min=0, mode=0):
        
        a = float(self.scale_value_entry_max.get())
        self.scale_left.set(a)
        b = float(self.scale_value_entry_min.get())
        self.scale_right.set(b)
        
        # Clear the figure
        self.disp_figure.clf()
        
        # Add subplots with adjusted parameters
        self.diff_axes = self.disp_figure.add_subplot(121, title='diffraction pattern')
        self.map_axes = self.disp_figure.add_subplot(122, title='mapping')
        # Remove unnecessary white space
        self.diff_axes.autoscale()
        self.diff_axes.margins(0)

        self.map_axes.autoscale()
        self.map_axes.margins(0)
        # Adjust subplot parameters for tight layout
        self.disp_figure.tight_layout()

        position1 = self.click_position1
        position2 = self.click_position2
        _position1, _position2 = swap_position(position1, position2)
        global data_4d
        # sum over real space to display averaged diffraction pattern
        temp = np.sum(np.sum(data_4d[_position1[1]:_position2[1] + 1,
                              _position1[0]:_position2[0] + 1,
                              :, :], 0), 0)
        max = self.scale_left.get() / 100
        min = self.scale_right.get() / 100
        if max > min:
            max = np.amax(temp) * max
            min = np.amax(temp) * min
            temp = np.minimum(temp, max)
            temp = np.maximum(temp, min)

        if mode == 0:
            self.diff_axes.imshow(temp)

        if mode == 1 or mode == 0:
            # set center to display
            if self.center_map is not None:
                if self.center_map.shape[1:3] == data_4d.shape[0:2]:
                    #average center over masked area in map 
                    cx=np.mean(self.center_map[1][_position1[1]:_position2[1] + 1,
                                _position1[0]:_position2[0] + 1])
                    cy=np.mean(self.center_map[0][_position1[1]:_position2[1] + 1,
                                _position1[0]:_position2[0] + 1])
                    center = [cx,cy]
                else:
                    print("center map shape "+ str(self.center_map.shape[1:3])
                        +" does not fit scan shape "+ str(data_4d.shape[0:2]))
                    center = self.center    
            else:
                center = self.center
            
            radius = self.radius
            self.update_center(center,radius)
            mask = np.zeros(np.shape(temp))
            x = int(center[0] + 0.5)    
            y = int(center[1] + 0.5)
            line_thickness = np.shape(temp)[0] // 256
            line_thickness = np.maximum(line_thickness, 1) - 1
            mask[y - line_thickness: y + line_thickness + 1, :] = 1
            mask[:, x - line_thickness:x + line_thickness + 1] = 1
            mask = Image.fromarray(np.uint8(mask))
            draw = ImageDraw.Draw(mask)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius], outline=1)
            draw.ellipse([x - (radius - 1), y - (radius - 1), x + (radius - 1), y + (radius - 1)], outline=1)
            mask = np.array(mask)
            mask2 = np.full(np.shape(mask), np.nan)
            mask2[np.where(mask == 1)] = 1
            cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'red'], 256)
            self.diff_axes.imshow(temp)
            self.diff_axes.imshow(mask2, cmap=cmap1)
            if self.mask2_int.get():
                mask2, cmap2 = self.get_diffraction_integration_mask()
                print('mask2')
                self.diff_axes.imshow(mask2, cmap=cmap2)
            self.diff_axes.imshow(temp, alpha=0)

        if self.replot_map:
            click_diffraction_1, click_diffraction_2 = swap_position(self.click_diffraction_1,
                                                                     self.click_diffraction_2)
            map = data_4d[:,
                                  :,
                                  click_diffraction_1[1]:click_diffraction_2[1]+1,
                                  click_diffraction_1[0]:click_diffraction_2[0]+1]
            self.fig_map = np.sum(np.sum(map, 2), 2)
        if _position1 is not None and _position2 is not None:
            mask_figure = np.zeros(np.shape(self.fig_map))
            mask_figure[np.where(mask_figure == 0)] = None
            cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'red'], 256)

            mask_figure[_position1[1]:_position2[1]+1, _position1[0]:_position2[0]+1] = 1
            mask_figure[_position1[1]+1:_position2[1], _position1[0]+1:_position2[0]] = None
            self.map_axes.imshow(self.fig_map)
            self.map_axes.imshow(mask_figure, cmap=cmap1)
            try:
                if self.mode3_int.get():
                    mask3, cmap3 = self.get_profile_cube_mask()
                    self.map_axes.imshow(mask3, cmap3)
            except:
                pass
            self.map_axes.imshow(self.fig_map, alpha=0)

        #self.disp_figure.colorbar(self.colorbar_area)
        self.canvas_disp.draw()

    def update_center(self, center, radius):
        self.center = center
        if radius < 1:
            self.radius = 1
        else:
            self.radius = radius
        self.sv_center_x.set('{0:.2f}'.format(center[0]))
        self.sv_center_y.set('{0:.2f}'.format(center[1]))
        self.sv_radius.set('{0:.2f}'.format(radius))

    def save_current_profile(self):
        # save current profile as txt file
        if self.profile is not None:
            path = asksaveasfilename(filetypes=(('text File', '*.txt'),
                                               ("All Files", "*.*")),
                                    title="Choose a file.")
            with open(path, 'wb+') as file:
                np.savetxt(file, self.profile)

    def save_center_map(self):
        # save center map as npy file
        if self.center_map is not None:
            path = asksaveasfilename(filetypes=(("text File", "*.mrc"),
                                                ('Numpy File', '*.npy'),
                                               ("All Files", "*.*")),
                                    title="Choose a file.")
            if path[-3:] == 'npy':
                with open(path, 'wb+') as file:
                    np.save(file, self.center_map)
            elif path[-3:] == 'mrc':
                 #center_map_temp = np.transpose(self.center_map,axes=(1,0))
                 with mrcfile.new(path, overwrite=True) as mrc:
                      mrc.set_data(self.center_map.astype(np.float32))
    
    def load_center_map(self):
        path = askopenfilename(filetypes=(('Numpy File', '*.npy'),
                                            ("All Files", "*.*")),
                                    title="Choose a file.")
        if path is not '':
            center_map = np.load(path, 'rb+')
            self.center_map=center_map
            self.plot_map()

    def import_data(self):
        if not (int(self.entry_map_y.get()) > 0 and int(self.entry_map_x.get()) > 0):
            data_shape = self.ask_userinfo()
            print(data_shape)
            if data_shape is not None:
                self.sv_map_y.set(str(data_shape[1]))
                self.sv_map_x.set(str(data_shape[0]))

        temp_path = askopenfilename(filetypes=(('Numpy File', '*.npy'),
                                               ('Merlin Image Binary File', '*.mib'),
                                               ('Tagged Image File Format File', '*.tif'),
                                               ('HyperSpy File', '*.hspy'),
                                               ("All Files", "*.*")),
                                    title="Choose a file.")

        if temp_path is None:
            return
        else:
            self.path = temp_path
        data_type = self.path[-3:]
        self.data_type = data_type
        try:
            x = int(self.entry_map_y.get())
            y = int(self.entry_map_x.get())
            x = np.maximum(x, 1)
            y = np.maximum(y, 1)
        except ValueError:
            x = 1
            y = 1
        self.data_size = [x, y]
        global data_4d
        if data_type == 'mib':
            self.raw_data_address = read_mib.MIBFile(self.path)
            a = time.time()
            data_4d = np.array(self.raw_data_address._frames(x * y, 0)) #read the 3d array from data file
        elif data_type == "npy":
            data_4d = np.load(self.path, mmap_mode='r')
        elif data_type == "raw":
            from read_empad import read_empad
            data_4d = read_empad(self.path)
        elif data_type == "tif":
            data_4d = np.array(Image.open(self.path))
        elif data_type == ".h5":
            import h5py
            data_4d = h5py.File(self.path)['entry']['data']['data']
        else:
            data_4d  = hs.load(self.path).data
        
        if len(data_4d.shape) < 4:
                 pic_shape = np.shape(data_4d[0]) #size of diffraction pattern
                 data_4d = data_4d[0:x*y].reshape((x, y, pic_shape[0], pic_shape[1])) # 3d array to 4d
        
        self.center_fig = data_4d[data_4d.shape[0]//2,data_4d.shape[1]//2] # assign first frame for center finding
        pic_shape = np.shape(data_4d[0,0]) #size of diffraction pattern
                
        try:
            x, y, radius = RDF_preparation.test_hough_2(self.center_fig)
            self.fig_map = np.sum(data_4d[:, :, x, :], 2) + \
                           np.sum(data_4d[:, :, y, :], 2)
            self.test_center = [x, y]
        except:
            self.fig_map = np.sum(data_4d[:, :, pic_shape[0] // 2, :], 2) + \
                           np.sum(data_4d[:, :, pic_shape[1] // 2, :], 2)
        print('read '+data_type+' file')
        self.plot_map()
        #display_histogram(self.center_fig)

    def save_beam_stopper(self):
        if self.beam_stopper is not None:
            im = Image.fromarray(np.int32(self.beam_stopper))
            path = asksaveasfilename(filetypes=(('Tagged Image File Format File', '*.tif'),
                                                ("All Files", "*.*")),
                                     title="Choose a file.")
            im.save(path)

    def auto_find_center(self, fig):
        # calls the center finding algorithm. At the moment just COM. coordinates are switched here 
        if fig is not None:
            #x, y, radius = RDF_preparation.test_hough_2(fig)
            #if self.path[-3:] == 'mib':
            x, y = RDF_preparation.mean_center(fig)#, self.radius)
            #radius = fig.shape[0]/10
            radius = self.radius
            return y, x, radius

    def manual_give_center(self):
        try:
            x = float(self.sv_center_x.get())
            y = float(self.sv_center_y.get())
            r = float(self.sv_radius.get())
            if r < 0:
                r = 1.00
        except ValueError:
            print('invalid coordination of center.')
            x = self.center[0]
            y = self.center[1]
        return x, y, r

    def find_center(self, mode='manuel'):
        if mode == 'auto':
            x, y, radius = self.auto_find_center(self.center_fig)
            self.radius = radius

        elif mode == 'given':
            x, y, self.radius = self.manual_give_center()
            ##########global_radius = self.radius
        else:
            return
        self.center = [x, y]
        self.update_center(self.center, self.radius)
        self.plot_map(mode=1)
    
    def calc_center_map(self):
        if data_4d is not None:
            self.center_map = RDF_preparation.CoM_Map(data_4d)
            #self.center_map, self.radius = RDF_preparation.calc_center_map(data_4d)
            # c = self.center_map[:,0,0]
            # self.update_center(center= c,radius=r)
            self.plot_map()
    
    def find_beam_stopper(self):
        center = [int(self.center[1]), int(self.center[0])]
        try:
            lower = self.entry_beam_stopper_lower.get()
            upper = self.entry_beam_stopper_upper.get()
            print(lower, upper)
            no_bs_image, beam_stopper = RDF_preparation.auto_beam_stopper(self.center_fig, center, lower, upper)
        except:
            no_bs_image, beam_stopper = RDF_preparation.auto_beam_stopper(self.center_fig, center)
        self.no_bs_image = no_bs_image
        self.beam_stopper = beam_stopper
        self.update_image(no_bs_image, fig2=beam_stopper, mode=2, title='pattern without beam stopper',
                          title2='beam stopper')

    def calculate_profile(self):
        center = [self.center[1] + 1, self.center[0] + 1]
        self.delete_mask = self.delete_pixel()
        if self.no_bs_image is None:
            no_bs_image = np.int32(self.center_fig)
            if self.delete_mask is None:
                profile = RDF_preparation.intensity_average(no_bs_image, center)
            else:
                profile = RDF_preparation.intensity_average(np.multiply(no_bs_image,
                                                                        self.delete_mask),
                                                            center)
        else:
            if self.delete_mask is None:
                profile = RDF_preparation.intensity_average(self.no_bs_image, center)
            else:
                profile = RDF_preparation.intensity_average(np.multiply(self.no_bs_image,
                                                                        self.delete_mask),
                                                            center)
        self.profile = profile
        self.profile[0:int(self.radius)]=0
        self.update_plot(self.profile, title='profile')

    def delete_pixel(self):
        try:
            nums = int(self.delete_pixel_num_sv.get())
            if nums < 0:
                nums = 0
                self.delete_pixel_num_sv.set(0)
        except ValueError:
            nums = 0
            self.delete_pixel_num_sv.set(nums)
        if nums > 0:
            temp_filter = np.zeros(np.shape(self.center_fig))
            temp_filter[nums:-nums, nums:-nums] = 1
            temp_filter[np.where(temp_filter == 0)] = None
            return temp_filter
        else:
            return None

    def calculate_cube(self):
        if not self.mode3_int.get():
            self.to_be_process_area = None
            self.area_count = 0
            self.area_1 = None
            self.area_2 = None
        self.delete_mask = self.delete_pixel()
        self.parent.profile_map()
        self.transfer_data()

    def get_diffraction_integration_mask(self):
        _position1, _position2 = swap_position(self.click_diffraction_1, self.click_diffraction_2)
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['yellow', 'yellow'], 256)
        mask_figure = np.zeros(np.shape(self.center_fig))
        mask_figure[np.where(mask_figure == 0)] = None
        mask_figure[_position1[1]:_position2[1] + 1, _position1[0]:_position2[0] + 1] = 1
        mask_figure[_position1[1] + 1:_position2[1], _position1[0] + 1:_position2[0]] = None
        return mask_figure, cmap2

    def get_profile_cube_mask(self):
        _position1, _position2 = swap_position(self.area_1, self.area_2)
        cmap2 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['yellow', 'yellow'], 256)
        mask_figure = np.zeros(np.shape(self.center_fig))
        mask_figure[np.where(mask_figure == 0)] = None
        mask_figure[_position1[1]:_position2[1] + 1, _position1[0]:_position2[0] + 1] = 1
        mask_figure[_position1[1] + 1:_position2[1], _position1[0] + 1:_position2[0]] = None
        self.to_be_process_area = [_position1, _position2]
        return mask_figure, cmap2

    def update_figure(self, event):
        self.canvas_disp.get_tk_widget().focus_force()
        #if not self.data_type == 'mib':
          #  return
        if event.inaxes == self.map_axes:
            if self.mode3_int.get():
                if self.area_count == 0:
                    self.area_1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.area_count += 1
                else:
                    self.area_2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.area_count = 2
                    self.plot_map()
                    self.area_count = 0
                return

            if not self.mode2_int.get():
                try:
                    self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.click_position2 = self.click_position1
                except ValueError:
                    pass
                self.plot_map()
            else:
                if self.click_count == 0:
                    self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.click_count += 1
                else:
                    self.click_position2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.plot_map()
                    self.click_count = 0

        elif event.inaxes == self.diff_axes:
            if self.mask2_int.get():
                if self.diffraction_count == 0:
                    self.click_diffraction_1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.diffraction_count += 1
                    self.replot_map = 0
                else:
                    self.click_diffraction_2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.replot_map = 1
                    self.plot_map()
                    self.replot_map = 0
                    self.diffraction_count = 0

    def press_key(self, event):
        center = self.center
        radius =self.radius
        key = event.key
        if key == 'left':
            x = self.sv_center_x.get()
            x = float(x) - 1
            center = [x, center[1]]
        elif key == 'right':
            x = self.sv_center_x.get()
            x = float(x) + 1
            center = [x, center[1]]
        elif key == 'up':
            y = self.sv_center_y.get()
            y = float(y) - 1
            center = [center[0], y]
        elif key == 'down':
            y = self.sv_center_y.get()
            y = float(y) + 1
            center = [center[0], y]
        elif key == '+' or key == '=':
            r = self.sv_radius.get()
            radius = float(r) + 1
        elif key == '-':
            r = self.sv_radius.get()
            radius = float(r) - 1
        else:
            return
        self.update_center(center=center, radius=radius)
        self.find_center(mode='given')

    def transfer_data(self):
        beam_stopper = self.beam_stopper
        center = self.center
        center_map =self.center_map
        path = self.path
        delete_mask = self.delete_mask
        process_size = None
        radius = self.radius
        if self.to_be_process_area is not None:
            process_size = [self.to_be_process_area[1][1] - self.to_be_process_area[0][1],
                            self.to_be_process_area[1][0] - self.to_be_process_area[0][0]]
        scanning_shape = [int(self.entry_map_y.get()), int(self.entry_map_x.get())]
        self.parent.pre_return_info({"path": path,
                                     "beam_stopper": beam_stopper,
                                     "center": center,
                                     "radius": radius,
                                     "center_map": center_map,
                                     "delete_mask": delete_mask,
                                     "scanning_shape": scanning_shape,
                                     "process_size": process_size,
                                     "process_area": self.to_be_process_area})

    def hist_fun(self, mode=1):
        if mode == 1:
            if self.center_fig is not None:
                display_histogram(self.center_fig)
        elif mode == 2:
            if self.fig_map is not None:
                display_histogram(self.fig_map)

    def ask_userinfo(self):
        inputDialog = GetDataShape()
        self.wait_window(inputDialog)
        return inputDialog.userinfo


class RdfFrame(tk.Frame):
    def __init__(self, frame=None, parent=None, rdfpara=None, profile=None):
        super().__init__(frame)
        self.parent = parent
        self.rdfpara_dict = rdfpara
        self.profile = profile
        self.frame_left = tk.Frame(self)
        self.frame_left.pack(side=tk.LEFT, fill='both', expand=1)
        self.frame_right = tk.Frame(self, width=10)
        self.frame_right.pack(side=tk.RIGHT, fill='y')

        self.setup_left_frame(self.frame_left)
        self.setup_right_frame(self.frame_right)

    def setup_left_frame(self, frame):
        self.plot_figure = Figure()
        self.plot_figure.add_subplot(221, title='profile')
        self.plot_figure.add_subplot(222, title='')
        self.plot_figure.add_subplot(223, title='')
        self.plot_figure.add_subplot(224, title='rdf')
        self.plot_figure.subplots_adjust(left=0.10, bottom=0.04,
                                         right=0.95, top=0.95,
                                         wspace=0.2, hspace=0.2)
        self.canvas = FigureCanvasTkAgg(self.plot_figure, frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(self.canvas, frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def setup_right_frame(self, frame):
        if self.rdfpara_dict is None:
            self.rdfpara_dict = RdfParaDict.dict
        self.setupGUI(frame)

    def setupGUI(self, frame):
        label_width = 18
        entry_width = 10
        # element
        row1 = tk.Frame(frame)
        row1.pack(fill='x')
        self.element_sv = tk.StringVar(frame,
                                       value=' '.join(str(e) for e in self.rdfpara_dict['element']))
        tk.Label(row1, text='element', justify='left', width=label_width).pack(side=tk.LEFT)
        self.element_entry = tk.Entry(row1, textvariable=self.element_sv, width=entry_width)
        self.element_entry.pack(side=tk.RIGHT)

        # composition
        row2 = tk.Frame(frame)
        row2.pack(fill='x')
        self.composition_sv = tk.StringVar(frame,
                                           value=' '.join(str(e) for e in self.rdfpara_dict['composition']))
        tk.Label(row2, text='composition', width=label_width).pack(side=tk.LEFT)
        self.composition_entry = tk.Entry(row2, textvariable=self.composition_sv, width=entry_width)
        self.composition_entry.pack(side=tk.RIGHT)

        row2_1 = tk.Frame(frame)
        row2_1.pack(fill='x')
        self.autodark_en_int = tk.IntVar(frame, value=self.rdfpara_dict['autodark_en'])
        tk.Label(row2_1, text='autodark enable', width=label_width).pack(side=tk.LEFT)
        self.autodark_en_check = tk.Checkbutton(row2_1, variable=self.autodark_en_int,
                                                command=self.autodark_en_function, width=entry_width)
        self.autodark_en_check.pack(side=tk.RIGHT)

        # phi begin
        row3 = tk.Frame(frame)
        row3.pack(fill='x')
        self.phi_fit_begin_sv = tk.StringVar(frame, value=self.rdfpara_dict['phi_fit_begin'])
        tk.Label(row3, text='beginning of phi', width=label_width).pack(side=tk.LEFT)
        self.phi_fit_begin_entry = tk.Entry(row3, textvariable=self.phi_fit_begin_sv, width=entry_width)
        self.phi_fit_begin_entry.pack(side=tk.RIGHT)

        # phi end
        row4 = tk.Frame(frame)
        row4.pack(fill='x')
        self.phi_fit_end_sv = tk.StringVar(frame, value=self.rdfpara_dict['phi_fit_end'])
        tk.Label(row4, text='end of phi', width=label_width).pack(side=tk.LEFT)
        self.phi_fit_end_entry = tk.Entry(row4, textvariable=self.phi_fit_end_sv, width=entry_width)
        self.phi_fit_end_entry.pack(side=tk.RIGHT)

        row5 = tk.Frame(frame)
        row5.pack(fill='x')
        self.pixel_begin_sv = tk.StringVar(frame, value=self.rdfpara_dict['pixel_begin'])
        tk.Label(row5, text='beginning of pixel', width=label_width).pack(side=tk.LEFT)
        self.pixel_begin_entry = tk.Entry(row5, textvariable=self.pixel_begin_sv, width=entry_width)
        self.pixel_begin_entry.pack(side=tk.RIGHT)

        row6 = tk.Frame(frame)
        row6.pack(fill='x')
        self.pixel_end_sv = tk.StringVar(frame, value=self.rdfpara_dict['pixel_end'])
        tk.Label(row6, text='end of pixel', width=label_width).pack(side=tk.LEFT)
        self.pixel_end_entry = tk.Entry(row6, textvariable=self.pixel_end_sv, width=entry_width)
        self.pixel_end_entry.pack(side=tk.RIGHT)

        row7 = tk.Frame(frame)
        row7.pack(fill='x')
        self.pixel_adjust_sv = tk.StringVar(frame, value=self.rdfpara_dict['pixel_adjust'])
        tk.Label(row7, text='adjust of the pixel', width=label_width).pack(side=tk.LEFT)
        self.pixel_adjust_entry = tk.Entry(row7, textvariable=self.pixel_adjust_sv, width=entry_width)
        self.pixel_adjust_entry.pack(side=tk.RIGHT)

        row8 = tk.Frame(frame)
        row8.pack(fill='x')
        self.smooth_en_int = tk.IntVar(frame, value=self.rdfpara_dict['smooth_en'])
        tk.Label(row8, text='smoothing enable', width=label_width).pack(side=tk.LEFT)
        self.smooth_en_check = tk.Checkbutton(row8, variable=self.smooth_en_int,
                                              command=self.smooth_en_function, width=entry_width)
        self.smooth_en_check.pack(side=tk.RIGHT)

        row9 = tk.Frame(frame)
        row9.pack(fill='x')
        self.smooth_range_sv = tk.StringVar(frame, value=self.rdfpara_dict['smooth_range'])
        tk.Label(row9, text='smoothing range', width=label_width).pack(side=tk.LEFT)
        self.smooth_range_entry = tk.Entry(row9, textvariable=self.smooth_range_sv, width=entry_width)
        self.smooth_range_entry.pack(side=tk.RIGHT)

        row10 = tk.Frame(frame)
        row10.pack(fill='x')
        self.smooth_strength_sv = tk.StringVar(frame, value=self.rdfpara_dict['smooth_strength'])
        tk.Label(row10, text='smoothing strength', width=label_width).pack(side=tk.LEFT)
        self.smooth_strength_entry = tk.Entry(row10, textvariable=self.smooth_strength_sv, width=entry_width)
        self.smooth_strength_entry.pack(side=tk.RIGHT)

        row11 = tk.Frame(frame)
        row11.pack(fill='x')
        self.polyfitN_sv = tk.StringVar(frame, value=self.rdfpara_dict['polyfitN'])
        tk.Label(row11, text='polynomial fitting order', width=label_width).pack(side=tk.LEFT)
        self.polyfitN_entry = tk.Entry(row11, textvariable=self.polyfitN_sv, width=entry_width)
        self.polyfitN_entry.pack(side=tk.RIGHT)

        row12 = tk.Frame(frame)
        row12.pack(fill='x')
        self.califactor_sv = tk.StringVar(frame, value=self.rdfpara_dict['califactor'])
        tk.Label(row12, text='calibration factor', width=label_width).pack(side=tk.LEFT)
        self.califactor_entry = tk.Entry(row12, textvariable=self.califactor_sv, width=entry_width)
        self.califactor_entry.pack(side=tk.RIGHT)

        row13 = tk.Frame(frame)
        row13.pack(fill='x')
        self.damping_en_int = tk.IntVar(frame, value=self.rdfpara_dict['damping_en'])
        tk.Label(row13, text='damping enable', width=label_width).pack(side=tk.LEFT)
        self.damping_en_check = tk.Checkbutton(row13, variable=self.damping_en_int,
                                               command=self.damping_en_function, width=entry_width)
        self.damping_en_check.pack(side=tk.RIGHT)

        row14 = tk.Frame(frame)
        row14.pack(fill='x')
        self.damping_start_point_sv = tk.StringVar(frame, value=self.rdfpara_dict['damping_start_point'])
        tk.Label(row14, text='start point of damping', width=label_width).pack(side=tk.LEFT)
        self.damping_start_point_entry = tk.Entry(row14, textvariable=self.damping_start_point_sv,
                                                  width=entry_width)
        self.damping_start_point_entry.pack(side=tk.RIGHT)

        row15 = tk.Frame(frame)
        row15.pack(fill='x')
        self.damping_strength_sv = tk.StringVar(frame, value=self.rdfpara_dict['damping_strength'])
        tk.Label(row15, text='damping_strength', width=label_width).pack(side=tk.LEFT)
        self.damping_strength_entry = tk.Entry(row15, textvariable=self.damping_strength_sv,
                                               width=entry_width)
        self.damping_strength_entry.pack(side=tk.RIGHT)

        row16 = tk.Frame(frame)
        row16.pack(fill='x')
        self.L_sv = tk.StringVar(frame, value=self.rdfpara_dict['L'])
        tk.Label(row16, text='L', width=label_width).pack(side=tk.LEFT)
        self.L_entry = tk.Entry(row16, textvariable=self.L_sv, width=entry_width)
        self.L_entry.pack(side=tk.RIGHT)

        row17 = tk.Frame(frame)
        row17.pack(fill='x')
        self.rn_sv = tk.StringVar(frame, value=self.rdfpara_dict['rn'])
        tk.Label(row17, text='rn', width=label_width).pack(side=tk.LEFT)
        self.rn_entry = tk.Entry(row17, textvariable=self.rn_sv, width=entry_width)
        self.rn_entry.pack(side=tk.RIGHT)

        self.autodark_en_function()
        self.smooth_en_function()
        self.damping_en_function()

        "system button"
        # return parameters
        row_end2 = tk.Frame(frame)
        row_end2.pack(fill='x', side=tk.BOTTOM)

        tk.Button(row_end2, text='calculate\nrdf cube', width=8, height=4, command=self.transfer_para).pack(side=tk.RIGHT)
        tk.Button(row_end2, text='load\npara', width=8, height=4, command=self.load_para).pack(side=tk.RIGHT)
        tk.Button(row_end2, text='save\npara', width=8, height=4, command=self.save_para).pack(side=tk.RIGHT)
        tk.Button(row_end2, text='update', width=8, height=4, command=self.update_rdf).pack(side=tk.RIGHT)

        row_output = tk.Frame(frame)
        row_output.pack(fill='x', side=tk.BOTTOM)
        tk.Button(row_output, text='save\n rdf', width=8, height=4,
                  command=self.save_rdf_data).pack(side=tk.RIGHT)
        tk.Button(row_output, text='save 1d\n structure\n factor', width=8, height=4,
                  command=self.save_structure_factor).pack(side=tk.RIGHT)
        button_read_profile = tk.Button(row_output, text='read\n1d profile', width=8, height=4,
                                        command=self.read_profile)
        button_read_profile.pack(side=tk.RIGHT)

        row_check = tk.Frame(frame)
        row_check.pack(side=tk.BOTTOM, fill='x')
        self.with_x_int = tk.IntVar()
        tk.Checkbutton(row_check, text='with x-axis data', variable=self.with_x_int).pack(side=tk.RIGHT)

    def read_profile(self, path=None, profile=None):
        if profile is None:
            if path is None:
                path = askopenfilename(filetypes=(("DM files", 'dm4'),
                                                  ("DM files", 'dm3'),
                                                  ("text files", 'txt'), 
                                                  ("All Files", "*.*")),
                                       title="Choose profile file.")
                if path[-3:] == 'dm4' or path[-3:] == 'dm3':
                   profile_array = hs.load(path).data
                elif path[-3:] == 'txt':
                   profile_array = np.loadtxt(path)
                   
                # with open(path, 'r') as a:
                #     while True:
                #         try:
                #             profile_array.append(float(a.readline()))
                #         except:
                #             break
            else:
                return
        else:
            profile_array = profile
            
        profile_array = np.array(profile_array)
        profile_array[np.where(profile_array == 0)] = None
        self.profile = profile_array
        self.update_rdf()
        #self.update_plot(self.profile, title='profile')

    def update_rdf(self):
        new_para_dict = self.read_input()
        if self.check_para(new_para_dict):
            new_para_dict['phi_fit_end'] = np.minimum(new_para_dict['phi_fit_end'],
                                                          len(self.profile))
            self.phi_fit_end_sv.set(new_para_dict['phi_fit_end'])

            if new_para_dict['phi_fit_end'] <= new_para_dict['phi_fit_begin']:
                new_para_dict['phi_fit_begin'] = new_para_dict['phi_fit_end']-1
                self.phi_fit_begin_sv.set(new_para_dict['phi_fit_begin'])
            new_para_dict['pixel_end'] = np.minimum(new_para_dict['pixel_end'],
                                                        len(self.profile))
            self.pixel_end_sv.set(new_para_dict['pixel_end'])
            if new_para_dict['pixel_end'] <= new_para_dict['pixel_begin']:
                new_para_dict['pixel_begin'] = new_para_dict['phi_fit_end']-1
                self.pixel_begin_sv.set(new_para_dict['pixel_begin'])
            self.rdfpara_dict = new_para_dict
            temp = dict2struct(self.rdfpara_dict, RdfParaData(True))
            temp.aver_inten = self.profile
            self.plot_dict = RDF_Package.rdf_cal(temp)
            self.build_plot()
            self.canvas.draw()
        else:
            ErrorWindow('008')
        return self.plot_dict

    def check_para(self, new_para):
        try:
            for item in new_para.values():
                if item is '' or item is None:
                    return 0
            if len(str2list(new_para['element'])) is not len(str2list(new_para['composition'])):
                print('not equal')
                return 0
            if new_para['phi_fit_begin'] >= new_para['phi_fit_end']:
                return 0
            if new_para['pixel_begin'] >= new_para['pixel_end']:
                return 0
            return 1
        except:
            return 0

    def build_plot(self):
        background_phi = self.plot_dict['background_phi']
        yfit = self.plot_dict['yfit']
        phi = self.plot_dict['phi']
        parameter_s = self.plot_dict['parameter_s']
        pixel_end = self.plot_dict['pixel_end']
        RIF_pristine = self.plot_dict['RIF_pristine']
        r = self.plot_dict['r']
        G_corrected = self.plot_dict['G_corrected']
        RIF_nosmo = self.plot_dict['RIF_nosmo']
        G_corrected_nod = self.plot_dict['G_corrected_nod']
        RIF_damped = self.plot_dict['RIF_damped']

        self.plot_figure.clf()
        self.plot_figure.subplots_adjust(left=0.12, bottom=0.10,
                            right=0.95, top=0.95,
                            wspace=0.4, hspace=0.4)
        a1 = self.plot_figure.add_subplot(221, title='background fitting',
                                          xlim=(parameter_s[0], parameter_s[pixel_end - 1]),
                                          xlabel='Å^-1',
                                          ylabel='Intensity')
        a1.plot(parameter_s[0:pixel_end], self.profile[0:pixel_end],
                           label='profile')
        a1.plot(parameter_s[0:pixel_end], background_phi[0:pixel_end],
                           label='background')
        a1.legend(loc=0)
        #a1.legend(bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
        a1.grid(color='b', linestyle='-', linewidth=0.1)

        a2 = self.plot_figure.add_subplot(222, title='polynomial fitting', xlim=(0, pixel_end),
                                          xlabel='pixel', ylabel='Intensity')
        a2.plot(phi[0:pixel_end], label='original structure factor') #  = profile-background
        a2.plot(yfit[0:pixel_end], label='fitting')
        #a2.legend(loc='lower right')
        a2.legend(loc=0)
        #plt.plot(parameter_s[0:pixel_end], phi[0:pixel_end])
        #plt.plot(parameter_s[0:pixel_end], yfit[0:pixel_end])
        #plt.title('G')
        #plt.xlim(0, pixel_end)
        #plt.xlim((parameter_s[0], parameter_s[pixel_end - 1]))
        a2.grid(color='b', linestyle='-', linewidth=0.1)
        plot_end = np.size(RIF_pristine)
        print(plot_end)
        if RIF_nosmo is not None:
            plot_end = np.minimum(plot_end, np.size(RIF_nosmo))
        if RIF_damped is not None:
            plot_end = np.minimum(plot_end, np.size(RIF_damped))

        print(plot_end)
        a3 = self.plot_figure.add_subplot(223, xlim=(0, plot_end),
                                      title='corrected structure factor', xlabel='pixel',
                                      ylabel='Intensity')

        if RIF_nosmo is not None:
            if RIF_damped is None:

                a3.plot(RIF_nosmo[0:plot_end], label='corrected structure factor')
                a3.plot(RIF_pristine[0:plot_end], label='factor with smoothing')
                a3.legend(loc=0)
            else:
                a3.plot(RIF_nosmo[0:plot_end], label='corrected structure factor')
                a3.plot(RIF_damped[0:plot_end], label='smoothed and damped')
                a3.legend(loc=0)
        else:
            if RIF_damped is not None:
                a3.plot(RIF_nosmo[0:plot_end], label='corrected structure factor')
                a3.plot(RIF_damped[0:plot_end], label='factor with damping')
                a3.legend(loc=0)
            else:
                a3.plot(RIF_pristine[0:plot_end], label='corrected structure factor')
                a3.legend(loc=0)

        a3.grid(color='b', linestyle='-', linewidth=0.1)

        a4 = self.plot_figure.add_subplot(224, xlim=(r[0], r[-1]), title='RDF',
                                          xlabel='Å', ylabel='Intensity')
        if G_corrected_nod is None:
            a4.plot(r, G_corrected, label='without damping')
        else:
            a4.plot(r, G_corrected_nod, label='without damping')
            a4.plot(r, G_corrected, label='with damping')
            a4.legend(loc=0)
        a4.grid(color='b', linestyle='-', linewidth=0.1, which='major', alpha=1)
        #a4.grid(color='b', linestyle='-', linewidth=0.1, which='minor', alpha=0.2)

    def update_rdf_para(self):
        #try:
            self.element_sv.set(' '.join(str(e) for e in self.rdfpara_dict['element']))
            self.composition_sv.set(' '.join(str(e) for e in self.rdfpara_dict['composition']))
            self.autodark_en_int.set(self.rdfpara_dict['autodark_en'])
            self.phi_fit_begin_sv.set(self.rdfpara_dict['phi_fit_begin'])
            self.phi_fit_end_sv.set(self.rdfpara_dict['phi_fit_end'])
            self.pixel_begin_sv.set(self.rdfpara_dict['pixel_begin'])
            self.pixel_end_sv.set(self.rdfpara_dict['pixel_end'])
            self.pixel_adjust_sv.set(self.rdfpara_dict['pixel_adjust'])
            self.smooth_en_int.set(self.rdfpara_dict['smooth_en'])
            self.smooth_range_sv.set(self.rdfpara_dict['smooth_range'])
            self.smooth_strength_sv.set(self.rdfpara_dict['smooth_strength'])
            self.polyfitN_sv.set(self.rdfpara_dict['polyfitN'])
            self.califactor_sv.set(self.rdfpara_dict['califactor'])
            self.damping_en_int.set(self.rdfpara_dict['damping_en'])
            self.damping_start_point_sv.set(self.rdfpara_dict['damping_start_point'])
            self.damping_strength_sv.set(self.rdfpara_dict['damping_strength'])
            self.L_sv.set(self.rdfpara_dict['L'])
            self.rn_sv.set(self.rdfpara_dict['rn'])
            print('success')

    def save_para(self):
        self.read_input()
        path = asksaveasfilename(filetypes=(("csv File", "*.csv"), ("All Files", "*.*")))
        with open(path, 'wb+') as file:
            pickle.dump(self.rdfpara_dict, file)
        '''
        with open(path, 'w') as f:
            for key in self.rdfpara_dict.keys():
                value = ' '.join(str(e) for e in self.rdfpara_dict[key])
                f.write("%s,%s\n" % (key, value))
        '''

    def load_para(self):
        path = askopenfilename(filetypes=(("csv File", "*.csv"), ("All Files", "*.*")),
                               title='Choose a file')
        if path is not '':
            file = open(path, 'rb+')
            self.rdfpara_dict = pickle.load(file)
            self.update_rdf_para()
            '''
                para_dict = {}
                para_dict_list = {}
                with open(path) as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        values = str2list(row[1])
                        para_dict[row[0]] = values
                self.rdfpara_dict = para_dict
                self.update_rdf_para()
            '''

    def autodark_en_function(self):
        if self.autodark_en_int.get():
            self.phi_fit_begin_entry.config(state='normal')
            self.phi_fit_end_entry.config(state='normal')
        else:
            self.phi_fit_begin_entry.config(state='disabled')
            self.phi_fit_end_entry.config(state='disabled')

    def smooth_en_function(self):
        if self.smooth_en_int.get():
            self.smooth_strength_entry.config(state='normal')
            self.smooth_range_entry.config(state='normal')
        else:
            self.smooth_strength_entry.config(state='disabled')
            self.smooth_range_entry.config(state='disabled')

    def damping_en_function(self):
        if self.damping_en_int.get():
            self.damping_start_point_entry.config(state='normal')
            self.damping_strength_entry.config(state='normal')
        else:
            self.damping_start_point_entry.config(state='disabled')
            self.damping_strength_entry.config(state='disabled')

    def transfer_para(self):
        self.parent.rdf_return_info(self.rdfpara_dict)

    def read_input(self):
        self.userinfo = {
            "element": str2list(self.element_entry.get()),
            "composition": str2list(self.composition_entry.get()),
            "autodark_en": str2list(self.autodark_en_int.get()),
            "phi_fit_begin": str2list(self.phi_fit_begin_entry.get()),
            "phi_fit_end": str2list(self.phi_fit_end_entry.get()),
            "pixel_begin": str2list(self.pixel_begin_entry.get()),
            "pixel_end": str2list(self.pixel_end_entry.get()),
            'pixel_adjust': str2list(self.pixel_adjust_entry.get()),
            "smooth_en": str2list(self.smooth_en_int.get()),
            "smooth_range": str2list(self.smooth_range_entry.get()),
            "smooth_strength": str2list(self.smooth_strength_entry.get()),
            "polyfitN": str2list(self.polyfitN_entry.get()),
            "califactor": str2list(self.califactor_entry.get()),
            "L": str2list(self.L_entry.get()),
            "rn": str2list(self.rn_entry.get()),
            "damping_en": str2list(self.damping_en_int.get()),
            "damping_strength": str2list(self.damping_strength_entry.get()),
            "damping_start_point": str2list(self.damping_start_point_entry.get())
        }
        for key in self.userinfo.keys():
            value = self.userinfo[key]
            try:
                if len(value) == 1 and key is not 'element' and key is not 'composition':
                    self.userinfo[key] = value[0]
            except:
                pass
        self.parent.rdf_window_return_info = self.userinfo
        return self.userinfo

    def save_rdf_data(self):
        data_format = '%.18e'
        path = asksaveasfilename(filetypes=(("npy File", "*.npy"),
                                            ("mrc File", "*.mrc"),
                                            ("text File", "*.txt"),
                                            ("All Files", "*.*")))
        if self.plot_dict['G_corrected_nod'] is None:
            rdf_data = self.plot_dict['G_corrected']
        else:
                rdf_data = self.plot_dict['G_corrected_nod']
        
        if self.plot_dict['r'] is not None and self.with_x_int.get():
            with open(path, 'w') as f:
                 np.savetxt(f, np.stack((self.plot_dict['r'], rdf_data), axis=-1), 
                            delimiter=' ', fmt=data_format)            
        
        if path[-3:] == 'npy':
           with open(path, 'wb+') as file: np.save(file, rdf_data)
        elif path[-3:] == 'mrc':
            with mrcfile.new(path, overwrite=True) as mrc:
                 mrc.set_data(rdf_data.astype(np.float32).reshape(1,len(rdf_data)))
        else: 
            with open(path, 'wb+') as file: np.savetxt(file, rdf_data, 
                                               delimiter='\n', fmt=data_format)
                

    def save_structure_factor(self):
        data_format = '%.18e'
        path = asksaveasfilename(filetypes=(("text file", "*.txt"), ("All Files", "*.*")))
        if self.plot_dict['RIF_damped'] is not None:
           data = self.plot_dict['RIF_damped']
        else:
           data = self.plot_dict['RIF_pristine']
        data[np.where(np.isnan(data))] = 0
        
        if path[-3:] == 'mrc':
            with mrcfile.new(path, overwrite=True) as mrc:
                 mrc.set_data(data.astype(np.float32).reshape(1,len(data)))
        else: 
            with open(path, 'wb+') as file: np.savetxt(file, data, 
                                           delimiter='\n', fmt=data_format)


class ProfileMapFrame(tk.Frame):
    def __init__(self, frame, parent):
        super().__init__(frame)
        self.parent = parent
        self.path = None
        #self.data_4d = None
        self._profile_map = None
        self._profile = None
        self.center_map = None
        self.beam_stopper = None
        self.shape_beam_stopper = None
        self.scale_value_left = 0
        self.scale_value_right = 0
        self.process_size = None
        self.delete_mask = None
        self.click_count = 0
        self.click_position1 = [0, 0]
        self.click_position2 = [0, 0]

        self.flag_click_count = 1
        self.flag_use_mouse = 0

        self.frame_left = tk.Frame(self, width=10)
        self.frame_left.pack(side=tk.LEFT)

        self.frame_right = tk.Frame(self)
        self.frame_right.pack(side=tk.RIGHT, fill='both', expand=1)

        self.frame_tool = tk.Frame(self.frame_right)
        self.frame_tool.pack(side=tk.TOP, fill='both', expand=1)
        self.frame_tool_left = tk.Frame(self.frame_tool)
        self.frame_tool_left.pack(side=tk.LEFT)

        self.setup_tool(self.frame_tool_left)
        self.frame_tool_right = tk.Frame(self.frame_tool)
        self.frame_tool_right.pack(side=tk.RIGHT)
        self.mode2_int = tk.IntVar(value=0)


        row0 = tk.Frame(self.frame_tool_right)
        row0.pack(side=tk.TOP)
        row1 = tk.Frame(self.frame_tool_right)
        row1.pack(side=tk.TOP)

        tk.Label(row1, text='display minimum').pack(side=tk.LEFT)
        self.sv_map_min = tk.StringVar(value=0)
        self.entry_map_min = tk.Entry(row1, textvariable=self.sv_map_min, width=5)
        self.entry_map_min.pack(side=tk.LEFT)

        tk.Label(row0, text='display maximum').pack(side=tk.LEFT)
        self.sv_map_max = tk.StringVar(value=0)
        self.entry_map_max = tk.Entry(row0, textvariable=self.sv_map_max, width=5)
        self.entry_map_max.pack(side=tk.LEFT)

        row2 = tk.Frame(self.frame_tool_right)
        row2.pack(side=tk.TOP)
        self.check_button_mode2 = tk.Checkbutton(row2,
                                                 variable=self.mode2_int,
                                                 text='choose integration area\nin right map')
        self.check_button_mode2.pack(side=tk.RIGHT)

        self.frame_display_window = tk.Frame(self.frame_right)
        self.frame_display_window.pack(side=tk.RIGHT, fill='both', expand=1)
        self.setup_command_window(self.frame_left)

        self.frame_rangingbar = tk.Frame(self.frame_right)
        self.frame_rangingbar.pack(side=tk.LEFT)
        self.profile_scale_bar = tk.Scale(self.frame_rangingbar, orient=tk.VERTICAL,
                                   resolution=0.1,
                                   length=400, sliderlength=20,
                                   command=self.scale_profile)
        self.profile_scale_bar.bind("<ButtonRelease-1>", self.plot_map)
        self.profile_scale_bar.pack(side=tk.RIGHT)
        self.profile_scale_bar.set(100)

        self.plot_figure = Figure()
        self.plot_figure.subplots_adjust(left=0.08, bottom=0.10,
                                         right=0.96, top=0.94,
                                         wspace=0.13, hspace=0.4)
        self.axes1 = self.plot_figure.add_subplot(121)
        self.axes2 = self.plot_figure.add_subplot(122)
        self.plot_canvas = self.build_canvas(self.frame_display_window, self.plot_figure)
        self.plot_canvas.mpl_connect('button_press_event', handlerAdaptor(self.plot_event))
        #self.plot_canvas.mpl_connect('button_release_event', handlerAdaptor(self.plot_event))
        #self.plot_canvas.mpl_connect('button_press_event', handlerAdaptor(self.plot_event))

    def setup_command_window(self, frame):
        '''
        tk.Button(frame, text='read data', command=self.read_data).pack()
        tk.Button(frame, text='read\nbeam stopper', command=self.read_beam_stopper).pack()
        tk.Button(frame, text='import map', command=self.import_data).pack()
        tk.Button(frame, text='save map', command=self.save_profile_data).pack()
        '''
        row1 = tk.Frame(frame)
        #row1.pack()
        tk.Label(row1, text='X: ')#.pack(side=tk.LEFT)
        self.sv_center_y = tk.StringVar(value='0.00')
        self.entry_center_y = tk.Entry(row1, textvariable=self.sv_center_y, width=5)
        self.entry_center_y.pack(side=tk.LEFT, fill='x', expand=1)

        row2 = tk.Frame(frame)
        #row2.pack()
        tk.Label(row2, text='Y: ')#.pack(side=tk.LEFT)
        self.sv_center_x = tk.StringVar(value='0.00')
        self.entry_center_x = tk.Entry(row2, textvariable=self.sv_center_x, width=5)
        
        row21 = tk.Frame(frame)
        tk.Label(row21, text='R: ')
        self.sv_radius = tk.StringVar(value='0.00')
        self.label_radius = tk.Entry(row21, textvariable=self.sv_radius, width=5)

        tk.Label(frame, text='shape of data:')#.pack()

        row3 = tk.Frame(frame)
        #row3.pack()
        self.sv_shape_x = tk.StringVar(value='1')
        self.entry_shape_x = tk.Entry(row3, textvariable=self.sv_shape_x, width=4)
        #self.entry_shape_x.grid(row=0, column=0)

        #tk.Label(row3, text=', ').grid(row=0, column=1)
        self.sv_shape_y = tk.StringVar(value='1')
        self.entry_shape_y = tk.Entry(row3, textvariable=self.sv_shape_y, width=4)
        #self.entry_shape_y.grid(row=0, column=2)

        #tk.Button(frame, text='calculate\n profile map', command=self.calculate_profile_map).pack()
        #tk.Button(frame, text='transfer map',
        #          command=lambda fun=self.transfer_data, mode=2: fun(mode)).pack()

    def build_canvas(self, master_frame, plot_frame=None):
        if plot_frame is None:
            return None
        canvas = FigureCanvasTkAgg(plot_frame, master=master_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, master_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def setup_tool(self, frame):
        frame_left = tk.Frame(frame)
        frame_left.pack(side=tk.LEFT, fill='both', expand=1)
        frame_right = tk.Frame(frame)
        frame_right.pack(side=tk.RIGHT, fill='both', expand=1)

        row0 = tk.Frame(frame_left)
        row0.pack()
        tk.Label(row0, text='width').pack(side=tk.LEFT)
        self.scale_right = tk.Scale(row0, orient=tk.HORIZONTAL,
                                    resolution=0.1,
                                    length=200, sliderlength=20,
                                    command=self.scale_event_right)
        self.scale_right.bind("<ButtonRelease-1>", self.plot_map)
        self.scale_right.pack(side=tk.RIGHT)
        row1 = tk.Frame(frame_left)
        row1.pack()
        tk.Label(row1, text='slides').pack(side=tk.LEFT)
        self.scale_left = tk.Scale(row1, orient=tk.HORIZONTAL,
                                   resolution=0.1,
                                   length=200, sliderlength=20,
                                   command=self.scale_event_left)
        self.scale_left.bind("<ButtonRelease-1>", self.plot_map)
        self.scale_left.pack(side=tk.RIGHT)

        tk.Button(frame_right, text='save\n3d profile cube', command=self.save_profile_data).pack(side=tk.BOTTOM)
        tk.Button(frame_right, text='import\n3d profile cube', command=self.import_data).pack(side=tk.BOTTOM)
        tk.Button(frame_right, text='transfer data',
                  command=lambda fun=self.transfer_data, mode=1: fun(mode)).pack(side=tk.BOTTOM)

    def read_data(self):
        path = askopenfilename(filetypes=(('Merlin Image Binary File', '*.mib'),
                                          ('Tagged Image File Format File', '*.tif'),
                                          ("All Files", "*.*")),
                               title="Choose a file.")
        if path[-3:] == 'mib':
            self.path = path
            self.mode = 'mib'
            print('mib')
        elif path[-3:] == 'tif':
            self.path = path
            self.mode = 'tif'
            print('tif')
        elif path[-3:] == 'npy':
            self.path = path
            self.mode = 'npy'
            print('npy')

    def read_beam_stopper(self):
        path = askopenfilename(filetypes=(('Tagged Image File Format File', '*.tif'),
                                          ("All Files", "*.*")),
                               title="Choose a file")
        if path[-3:] == 'tif':
            beam_stopper1 = Image.open(path)
            beam_stopper1 = np.array(beam_stopper1)
            beam_stopper = np.full(np.shape(beam_stopper1), np.nan)
            beam_stopper[np.where(beam_stopper1 == 1)] = 1
            self.beam_stopper = beam_stopper
            self.shape_beam_stopper = np.shape(self.beam_stopper)

    def import_data(self):
        path = askopenfilename(filetypes=(('npy file', '*.npy'),
                                          ('mat file', '*.mat'),
                                          ("All Files", "*.*")),
                               title="Choose a file")
        if path[-3:] == 'mat':
            print(path)
            self._profile_map = sio.loadmat(path)['data']
        elif path[-3:] == 'npy':
            #with open(path, 'r') as file:
            self._profile_map = np.load(path)
        shape = np.shape(self._profile_map)
        print(shape)
        self.sv_shape_x.set(shape[0])
        self.sv_shape_y.set(shape[1])
        self.plot_map()

    def save_profile_data(self):
        data_format = float
        path = asksaveasfilename(filetypes=(("npy File", "*.npy"),
                                            ("text File", "*.mrc"),
                                            ("text File", "*.txt"),
                                            ("All Files", "*.*")))
        if path[-3:] == 'npy':
            with open(path, 'wb+') as file:
                np.save(file, self._profile_map)
        elif path[-3:] == 'txt':
            with open(path, 'wb+') as file:
                data = self._profile_map.reshape(np.size(self._profile_map))
                np.savetxt(file, data, delimiter='\n', fmt=data_format)
        elif path[-3:] == 'mrc':
           # with open(path, 'wb+') as file:
               profile_map_temp = np.transpose(self._profile_map,axes=(2,0,1))
               profile_map_temp = profile_map_temp[:,::-1,:]
               with mrcfile.new(path, overwrite=True) as mrc:
                    mrc.set_data(profile_map_temp.astype(np.float32))
        else:
            print("\n A file extension is necessary \n")

    def calculate_profile_map(self, process_area=None):
        # Calculates a 3d profile cube from 4d data. 
        # Can constrict to process_area and uses center_map if present
        a=time.time()
        if self.center_map is not None:
            radius = float(self.label_radius.get())
            if self.center_map.shape[1:3] == data_4d.shape[0:2]:
                center_map = self.center_map+1
            else:
                center_map = None
            
        else:
            center_map = None
            try:
                x = float(self.entry_center_x.get())
                y = float(self.entry_center_y.get())
                radius = float(self.label_radius.get())
            except ValueError:
                print('invalid center')
                return
            center = [y+1, x+1]
        x_start = 0
        y_start = 0
        x_end = data_4d.shape[0]
        y_end = data_4d.shape[1]

        if process_area is not None:
            if process_area[0] is not None and process_area[1] is not None:
                x_start = process_area[0][1]
                x_end = process_area[1][1] + 1
                y_start = process_area[0][0]
                y_end = process_area[1][0] + 1
        # set maximum radius to half of sensor diagonal
        profile_length = int(((data_4d.shape[2]/2)**2+(data_4d.shape[3]/2)**2)**0.5)
        _profile_map = np.zeros((x_end-x_start, y_end-y_start, profile_length))
        for i in np.arange(x_start, x_end):
            print(a-time.time())
            a=time.time()
            for j in np.arange(y_start, y_end):
                single_pattern = data_4d[i, j, :, :]
                if self.shape_beam_stopper == np.shape(single_pattern):
                    single_pattern = np.multiply(single_pattern, self.beam_stopper)
                if self.delete_mask is not None:
                    single_pattern = np.multiply(single_pattern, self.delete_mask)
                if center_map is not None:
                    center = [center_map[0][i,j]+1,center_map[1][i,j]+1]
                profile = RDF_preparation.intensity_average(single_pattern, center)
                profile[0:int(radius)] = 0 ##################################
                if profile.shape[0] == profile_length:
                    _profile_map[i-x_start,
                        j-y_start,
                        :] = profile
                elif profile.shape[0] < profile_length:
                    _profile_map[i-x_start,
                        j-y_start,
                        :profile.shape] = profile
                else:
                    _profile_map[i-x_start,
                        j-y_start,
                        :] = profile[:profile_length]
        self._profile_map = _profile_map
        self.plot_map()

    def plot_profile(self):
        if self.flag_use_mouse:
            line1 = int(self.last_click_position)
            line2 = int(self.current_click_position)
            line1, line2 = min(line1, line2), max(line1, line2)
        else:
            line1 = max(0, self.scale_value_left)
            line2 = min(1, self.scale_value_right + self.scale_value_left)

        temp = 1 - np.power(1 - self.profile_scale_bar.get()/100, 0.1)

        _position1, _position2 = swap_position(self.click_position1, self.click_position2)
        if _position1 is not None and _position2 is not None and self._profile_map is not None:
            self.axes1.cla()
            self.axes1.title.set_text('profile')
            _profile = np.sum(self._profile_map[_position1[1]:_position2[1] + 1,
                                                _position1[0]:_position2[0] + 1,
                                                :], 0)
            _profile = np.sum(_profile, 0)
            self._profile = _profile
            self.axes1.plot(_profile)
            self.axes1.set_ylim([np.min(_profile), np.max(_profile)*temp])
            if line1 < line2:
                if self.flag_use_mouse:
                    self.axes1 = self.profile_mask(_profile, line1, line2, self.axes1, 0)
                else:
                    self.axes1 = self.profile_mask(_profile, line1, line2, self.axes1)
            self.plot_canvas.draw()

    def plot_map(self, event=None, max=0, min=0):
        if self._profile_map is None:
            return
        temp = copy.deepcopy(self._profile_map)
        temp[np.isnan(temp)] = 0

        if self.scale_value_right > 0:
            start = self.scale_value_left * np.shape(self._profile_map)[2]
            #print(self.scale_value_right, self.scale_value_left)
            range = self.scale_value_right + self.scale_value_left
            if range > 1:
                range = 1
            end = range * np.shape(self._profile_map)[2]
            profile_map = np.sum(temp[:, :, int(start):int(end)+1], 2)
        else:
            profile_map = np.sum(temp, 2)
        del temp

        try:
            min = float(self.sv_map_min.get())
            max = float(self.sv_map_max.get())
            print(min, max)
        except:
            print('error min max')

        if max > min:
            profile_map = np.minimum(profile_map, max)
            profile_map = np.maximum(profile_map, min)
            flag = 1
        else:
            flag = 0
        self.plot_figure.clf()
        self.axes1 = self.plot_figure.add_subplot(121, title='profile')
        self.axes2 = self.plot_figure.add_subplot(122, title='mapping')

        self.plot_profile()
        if flag:
            self.colorbar_area = self.axes2.imshow(profile_map, vmax=max, vmin=min)
        else:
            self.colorbar_area = self.axes2.imshow(profile_map)

        position1 = self.click_position1
        position2 = self.click_position2

        _position1, _position2 = swap_position(position1, position2)
        if _position1 is not None and _position2 is not None:
            mask_figure = np.zeros(np.shape(profile_map))
            mask_figure[np.where(mask_figure == 0)] = None
            cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'red'], 256)

            mask_figure[_position1[1]:_position2[1]+1, _position1[0]:_position2[0]+1] = 1
            mask_figure[_position1[1]+1:_position2[1], _position1[0]+1:_position2[0]] = None
            self.axes2.imshow(mask_figure, cmap=cmap1)
            if flag:
                self.axes2.imshow(profile_map, alpha=0, vmin=min, vmax=max)
            else:
                self.axes2.imshow(profile_map, alpha=0)

        self.plot_figure.colorbar(self.colorbar_area)
        self.plot_canvas.draw()
        
    def plot_event(self, event):
        if event.inaxes is self.axes1:
            print(event.xdata)
            print('flag', self.flag_click_count)
            if self.flag_click_count == 1:
                self.last_click_position = event.xdata
                self.flag_click_count *= -1
            else:
                self.flag_use_mouse = 1
                self.current_click_position = event.xdata
                self.plot_map()
                self.flag_click_count *= -1

        elif event.inaxes is self.axes2:
            if not self.mode2_int.get():
                try:
                    if not self.mode2_int.get():
                        self.click_count = 0
                        self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                        self.click_position2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                        self.plot_map()
                except ValueError:
                    pass
            else:
                if self.click_count == 0:
                    self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.click_count += 1
                else:
                    self.click_position2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.plot_map()
                    self.click_count = 0
            self.parent.rdf_return_position(self.click_position1, self.click_position2)

    def scale_event_left(self, event):
        self.flag_use_mouse = 0
        self.scale_value_left = self.scale_left.get()/100
        self.scale_value_right = self.scale_right.get() / 100
        self.plot_profile()

    def scale_event_right(self, event):
        self.flag_use_mouse = 0
        self.scale_value_right = self.scale_right.get()/100
        self.scale_value_left = self.scale_left.get() / 100
        self.plot_profile()

    def scale_profile(self, event):
        self.plot_profile()

    def profile_mask(self, profile, line1, line2, axes, use_scale=1):
        if use_scale:
            x1 = np.size(profile) * line1
            x2 = np.size(profile) * line2
        else:
            x1 = line1
            x2 = line2
        profile[np.isnan(profile)] = 0

        profile_max = np.amax(profile)
        profile_min = np.amin(profile)

        step = (x2 - x1)/100
        a = np.arange(x1, x2, step)
        axes.fill_between(a, profile_min, profile_max, alpha=0.2, color='r')
        axes.plot([x1, x1], [profile_min, profile_max], 'r')
        axes.plot([x2, x2], [profile_min, profile_max], 'r')
        return axes

    def transfer_data(self, mode=1):
        if self.process_size is None:
            scanning_shape = [int(self.sv_shape_x.get()), int(self.sv_shape_y.get())]
        else:
            scanning_shape = self.process_size
        print('scanning_shape', scanning_shape)
        dict_info = {'profile': self._profile,
                     'profile_map': self._profile_map,
                     'scanning_shape': scanning_shape}
        self.parent.profile_map_return_info(mode=mode, dict_info=dict_info)


class RdfMapFrame(tk.Frame):
    def __init__(self, frame, parent):
        super().__init__(frame)
        self.parent = parent

        self.click_position1 = [0, 0]
        self.click_position2 = [0, 0]
        self.scale_value_left = 0
        self.scale_value_right = 0
        self.path = None
        self._rdf_map = None
        self.rdfpara_dict = None
        self.scale_factor = 1

        self.frame_left = tk.Frame(self, width=10)
        self.frame_left.pack(side=tk.LEFT, fill='y')

        self.frame_right = tk.Frame(self)
        self.frame_right.pack(side=tk.RIGHT, fill='both', expand=1)

        self.frame_tool = tk.Frame(self.frame_right)
        self.frame_tool.pack(side=tk.TOP, fill='both', expand=1)
        self.frame_tool_left = tk.Frame(self.frame_tool)
        self.frame_tool_left.pack(side=tk.LEFT)

        self.setup_tool(self.frame_tool_left)
        self.frame_tool_right = tk.Frame(self.frame_tool)
        self.frame_tool_right.pack(side=tk.RIGHT)

        row1 = tk.Frame(self.frame_tool_right)
        row1.pack(side=tk.BOTTOM, fill='both', expand=1)
        self.mode2_int = tk.IntVar(value=0)
        self.check_button_mode2 = tk.Checkbutton(row1,
                                                 variable=self.mode2_int,
                                                 text='choose integration area\nin right map')
        self.check_button_mode2.pack(side=tk.RIGHT)
        row2 = tk.Frame(self.frame_tool_right)
        row2.pack(side=tk.BOTTOM, fill='both', expand=1)
        map_frame1 = tk.Frame(row2)
        map_frame1.pack(side=tk.BOTTOM, expand=1, fill='both')
        map_frame2 = tk.Frame(row2)
        map_frame2.pack(side=tk.BOTTOM, expand=1, fill='both')

        tk.Label(map_frame1, text='minimum of map:').pack(side=tk.LEFT)
        self.sv_map_min = tk.StringVar(value=0)
        self.entry_map_min = tk.Entry(map_frame1, textvariable=self.sv_map_min, width=5)
        self.entry_map_min.pack(side=tk.LEFT)

        tk.Label(map_frame2, text='maximum of map:').pack(side=tk.LEFT)
        self.sv_map_max = tk.StringVar(value=0)
        self.entry_map_max = tk.Entry(map_frame2, textvariable=self.sv_map_max, width=5)
        self.entry_map_max.pack(side=tk.LEFT)

        self.frame_display_window = tk.Frame(self.frame_right)
        self.frame_display_window.pack(side=tk.BOTTOM, fill='both', expand=1)
        temp_frame = tk.Frame(self.frame_left)
        temp_frame.pack(side=tk.LEFT, fill='x')
        #self.sv_info = tk.StringVar(value='waiting')
        #tk.Label(self.frame_left, textvariable=self.sv_info).pack(side=tk.TOP)
        self.setup_command_window(temp_frame)

        self.plot_figure = Figure()
        self.plot_figure.subplots_adjust(left=0.08, bottom=0.10,
                                         right=0.96, top=0.94,
                                         wspace=0.13, hspace=0.4)
        self.axes1 = self.plot_figure.add_subplot(121)
        self.axes2 = self.plot_figure.add_subplot(122)
        self.plot_canvas = self.build_canvas(self.frame_display_window, self.plot_figure)
        self.plot_canvas.mpl_connect('button_press_event', handlerAdaptor(self.plot_event))

    def setup_command_window(self, frame):
        '''
        tk.Button(frame, text='read data', command=self.read_data).pack()
        tk.Button(frame, text='import\nparameter', command=self.import_parameter).pack()
        tk.Button(frame, text='import map', command=self.import_map).pack()
        tk.Button(frame, text='save map', command=self.save_rdf_data).pack()

        tk.Label(frame, text='shape of data:').pack()
        '''
        row3 = tk.Frame(frame)
        #row3.pack()

        self.sv_shape_x = tk.StringVar(value='1')
        self.entry_shape_x = tk.Entry(row3, textvariable=self.sv_shape_x, width=4)
        #self.entry_shape_x.grid(row=0, column=0)

        #tk.Label(row3, text=', ').grid(row=0, column=1)

        self.sv_shape_y = tk.StringVar(value='1')
        self.entry_shape_y = tk.Entry(row3, textvariable=self.sv_shape_y, width=4)
        #self.entry_shape_y.grid(row=0, column=2)

        #tk.Button(frame, text='calculate\n rdf map', command=self.calcualte_rdf_map).pack()

    def build_canvas(self, master_frame, plot_frame=None):
        if plot_frame is None:
            return None
        canvas = FigureCanvasTkAgg(plot_frame, master=master_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, master_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        return canvas

    def setup_tool(self, frame):
        frame_left = tk.Frame(frame)
        frame_left.pack(side=tk.LEFT, fill='both', expand=1)
        frame_right = tk.Frame(frame)
        frame_right.pack(side=tk.RIGHT, fill='both', expand=1)
        row1 = tk.Frame(frame_left)
        row1.pack()
        tk.Label(row1, text='width').pack(side=tk.LEFT)
        self.scale_left = tk.Scale(row1, orient=tk.HORIZONTAL,
                                   resolution=0.1,
                                   length=200, sliderlength=20,
                                   command=self.scale_event_left)
        self.scale_left.bind("<ButtonRelease-1>", self.plot_map)
        self.scale_left.pack(side=tk.RIGHT)

        row0 = tk.Frame(frame_left)
        row0.pack()
        tk.Label(row0, text='slides').pack(side=tk.LEFT)
        self.scale_right = tk.Scale(row0, orient=tk.HORIZONTAL,
                                    resolution=0.1,
                                    length=200, sliderlength=20,
                                    command=self.scale_event_right)
        self.scale_right.bind("<ButtonRelease-1>", self.plot_map)

        self.scale_right.pack(side=tk.RIGHT)


        tk.Button(frame_right, text='save\n3d rdf cube', command=self.save_rdf_data).pack()
        tk.Button(frame_right, text='load\n3d rdf cube', command=self.import_map).pack()

    def read_data(self, path=None):
        if path is None:
            path = askopenfilename(filetypes=(('csv file', '*.csv'),
                                              ("All Files", "*.*")),
                                   title="Choose a file")
        if path[-3:] == 'csv':
            self.mode = 'csv'
            print('csv')
        if path[-3:] == 'mib':
            self.mode = 'mib'
            print('mib')
        elif path[-3:] == 'tif':
            self.mode = 'tif'
            print('tif')
        elif path[-3:] == 'npy':
            self.mode = 'npy'
            print('npy')
        self.path = path

    def import_map(self):
        path = askopenfilename(filetypes=(('csv file', '*.csv'),
                                          ("All Files", "*.*")),
                               title="Choose a file")
        with open(path, 'rb+') as file:
            self._rdf_map = pickle.load(file)
        self.plot_map()

    def import_parameter(self):
        path = askopenfilename(filetypes=(("csv File", "*.csv"), ("All Files", "*.*")),
                               title='Choose a file')
        if path is not '':
            file = open(path, 'rb+')
            self.rdfpara_dict = pickle.load(file)
            #self.update_rdf_para()

    def save_rdf_data(self):
        data_format = '%.18e'
        path = asksaveasfilename(filetypes=(("mrc File", "*.mrc"),
                                            ("npy File", "*.npy"),
                                            ("text File", "*.txt"),
                                            ("All Files", "*.*")))
        data_shape = np.shape(self._rdf_map)
        datainfo = str(data_shape[2]) + '_' + str(data_shape[0]) + '_' + str(data_shape[1])
        path = path[:-4] + '_size_' + datainfo + path[-4:] # for RDFmap
        path2 = path[:-4]+'_RIF_'+path[-4:] # for structure factor map
        
        rdf_map_temp = self._rdf_map
        rdf_map_temp[np.isnan(rdf_map_temp)] = 0
        
        if path[-3:] == 'mrc':
             rdf_map_temp = np.transpose(rdf_map_temp,axes=(2,0,1))
             rdf_map_temp = rdf_map_temp[:,::-1,:]
             with mrcfile.new(path, overwrite=True) as mrc:
                  mrc.set_data(rdf_map_temp.astype(np.float32))
        elif path[-3:] == 'npy':
            with open(path, 'wb+') as file:
                np.save(file, rdf_map_temp)
        elif path[-3:] == 'txt':
            with open(path, 'wb+') as file:
                data = rdf_map_temp.reshape(np.size(rdf_map_temp))
                np.savetxt(file, data, delimiter=",", newline="\r\n", fmt=data_format)

        RIF_map_temp = self._RIF_map
        RIF_map_temp[np.isnan(RIF_map_temp)] = 0
        if path2[-3:] == 'mrc':
            RIF_map_temp = np.transpose(RIF_map_temp,axes=(2,0,1))
            RIF_map_temp = RIF_map_temp[:,::-1,:]
            with mrcfile.new(path2, overwrite=True) as mrc:
                 mrc.set_data(RIF_map_temp.astype(np.float32))
        elif path[-3:] == 'npy':
            with open(path, 'wb+') as file:
                np.save(file, RIF_map_temp)
        else: 
            with open(path2, 'wb+') as file:
                RIF_map_temp = RIF_map_temp.reshape(np.size(RIF_map_temp))
                np.savetxt(file, RIF_map_temp, delimiter=",", newline="\r\n", fmt=data_format)

    def calculate_rdf_map(self):
        try:
            shape_x = int(self.entry_shape_x.get())
            shape_y = int(self.entry_shape_y.get())
        except ValueError:
            print('invalid shape')
            return
        if self.path is not None:
            self.profile_map, shape_x, shape_y = self.get_profile_data(self.path, self.mode, shape_x, shape_y)
        else:
            if self.profile_map is None:
                return
        if self.rdfpara_dict is not None:
            temp = dict2struct(self.rdfpara_dict, RdfParaData(True))
        else:
            return
        temp.aver_inten = self.profile_map[0, 0, :]
        rdf_data_cube = RDF_Package.rdf_cal(temp)
        #if rdf_data_cube['G_corrected_nod'] is not None:
        #    rdf_test = rdf_data_cube['G_corrected_nod']
        #else:
        #    rdf_test = rdf_data_cube['G_corrected']
        rdf_test = rdf_data_cube['G_corrected']
        if rdf_data_cube['RIF_damped'] is not None:
            RIF_test = rdf_data_cube['RIF_damped']
        else:
            RIF_test = rdf_data_cube['RIF_pristine']

        self.r = rdf_data_cube['r']
        rdf_length = np.size(rdf_test)
        _rdf_map = np.zeros((shape_x, shape_y, rdf_length))
        _RIF_map = np.zeros((shape_x, shape_y, np.size(RIF_test)))
        #yfit_nosmo = rdf_data_cube['RIF_pristine']   ############
        #_struct_map = np.zeros((shape_x, shape_y, np.size(yfit_nosmo)))#######
        count = 0
        total_num = shape_x * shape_y

        for i in np.arange(shape_x):
            for j in np.arange(shape_y):
                count += 1
                single_pattern = self.profile_map[i, j, :]
                single_pattern[np.isnan(single_pattern)] = 0
                temp.aver_inten = single_pattern
                _data_set = RDF_Package.rdf_cal(temp)

                #_struct_map[i, j, :] = _data_set['RIF_pristine'] ##########
                
                #if _data_set['G_corrected_nod'] is not None:
                #    _rdf_map[i, j, :] = _data_set['G_corrected_nod']
                #else:
                #    _rdf_map[i, j, :] = _data_set['G_corrected']
                _rdf_map[i, j, :] = _data_set['G_corrected']
                if _data_set['RIF_damped'] is not None:
                    _RIF_map[i, j, :] = _data_set['RIF_damped']
                else:
                    _RIF_map[i, j, :] = _data_set['RIF_pristine']

                '''
                if count % 100 == 0:
                    current_time = time.time()
                    current_num = j+1+i*shape_x
                    used_time = current_time-start_time
                    remaining_time = used_time/current_num * total_num - used_time
                    print('calculating %d / %d\n'
                          'used time: %d\n'
                          'remaining time: %d' % (current_num, total_num, used_time, remaining_time))
                '''
        self._RIF_map = _RIF_map
        self._rdf_map = _rdf_map
        #self._struct_map = _struct_map
        self.plot_map()

    def get_profile_data(self, path=None, mode=None, shape_x=None, shape_y=None):
        if mode == 'csv':
            with open(path, 'rb+') as file:
                _rdf_map =pickle.load(file)
            if shape_x is not None and shape_y is not None:
                map_shape = np.shape(_rdf_map)
                shape_x = np.minimum(shape_x, map_shape[0])
                shape_y = np.minimum(shape_y, map_shape[1])
                self.sv_shape_x.set(shape_x)
                self.sv_shape_y.set(shape_y)
                return _rdf_map[0:shape_x+1, 0:shape_y+1, :], shape_x, shape_y
            else:
                return _rdf_map, shape_x, shape_y
        else:
            return None, None, None

    def plot_event(self, event):
        if event.inaxes is self.axes1:
            pass
        elif event.inaxes is self.axes2:
            if not self.mode2_int.get():
                try:
                    if not self.mode2_int.get():
                        self.click_count = 0
                        self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                        self.click_position2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                        self.plot_map()
                except ValueError:
                    pass
            else:
                if self.click_count == 0:
                    self.click_position1 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.click_count += 1
                else:
                    self.click_position2 = [int(event.xdata + 0.5), int(event.ydata + 0.5)]
                    self.plot_map()
                    self.click_count = 0
            self.return_click_position()

    def plot_rdf(self):
        try:
            a = self.rdfpara_dict['L']
            b = self.rdfpara_dict['rn']
            self.scale_factor = b/a
        except TypeError:
            pass
        line1 = self.scale_value_right / self.scale_factor
        line2 = min(1, self.scale_value_right + self.scale_value_left) / self.scale_factor
        # line1 = self.scale_value_left
        # line2 = min(1, self.scale_value_right + self.scale_value_left)

        _position1, _position2 = swap_position(self.click_position1, self.click_position2)
        if _position1 is not None and _position2 is not None and self._rdf_map is not None:
            self.axes1.cla()
            self.axes1.title.set_text('profile')
            _profile = np.sum(self._rdf_map[_position1[1]:_position2[1] + 1,
                                            _position1[0]:_position2[0] + 1,
                                            :], 0)
            _profile = np.sum(_profile, 0)
            try:
                self.axes1.plot(self.r, _profile)
            except AttributeError:
                self.axes1.plot(_profile)

            if line1 < line2:
                self.axes1 = self.profile_mask(_profile, line1, line2, self.axes1)
            self.plot_canvas.draw()

    def plot_map(self, event=None, max=0, min=0):
        if self._rdf_map is None:
            return
        try:
            max = float(self.sv_map_max.get())
            min = float(self.sv_map_min.get())
        except ValueError:
            max = 1
            min = 0

        temp = copy.deepcopy(self._rdf_map)
        temp[np.isnan(temp)] = 0
        if self.scale_value_left > 0 or self.scale_value_right > 0:
            print(self.scale_value_left, self.scale_value_right)
            start = self.scale_value_right * np.shape(self._rdf_map)[2]

            end = np.minimum(self.scale_value_left + self.scale_value_right, 1.0) * np.shape(self._rdf_map)[2]
            print(start, end)
            if start == end:
                rdf_map = np.average(temp, 2)
            else:
                rdf_map = np.average(temp[:, :, int(start):int(end) + 1], 2)
        else:
            rdf_map = np.average(temp, 2)
        del temp

        if max > min:
            rdf_map = np.minimum(rdf_map, max)
            rdf_map = np.maximum(rdf_map, min)
            flag = 1
        else:
            flag = 0

        self.plot_figure.clf()
        self.axes1 = self.plot_figure.add_subplot(121, title='rdf')
        self.axes2 = self.plot_figure.add_subplot(122, title='mapping')

        self.plot_rdf()
        if flag:
            self.colorbar_area = self.axes2.imshow(rdf_map, vmax=max, vmin=min)
        else:
            self.colorbar_area = self.axes2.imshow(rdf_map)

        position1 = self.click_position1
        position2 = self.click_position2
        _position1, _position2 = swap_position(position1, position2)
        if _position1 is not None and _position2 is not None:
            mask_figure = np.zeros(np.shape(rdf_map))
            mask_figure[np.where(mask_figure == 0)] = None
            cmap1 = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap', ['red', 'red'], 256)

            mask_figure[_position1[1]:_position2[1] + 1, _position1[0]:_position2[0] + 1] = 1
            mask_figure[_position1[1] + 1:_position2[1], _position1[0] + 1:_position2[0]] = None
            self.axes2.imshow(mask_figure, cmap=cmap1)
            if flag:
                self.axes2.imshow(rdf_map, alpha=0, vmin=min,vmax=max,)
            else:
                self.axes2.imshow(rdf_map, alpha=0)

        self.plot_figure.colorbar(self.colorbar_area)
        self.plot_canvas.draw()

    def scale_event_left(self, event):
        self.scale_value_left = self.scale_left.get()/100
        self.plot_rdf()

    def scale_event_right(self, event):
        self.scale_value_right = self.scale_right.get()/100
        self.plot_rdf()

    def profile_mask(self, profile, line1, line2, axes):
        x1 = np.size(profile) * line1
        x2 = np.size(profile) * line2
        profile[np.isnan(profile)] = 0

        profile_max = np.amax(profile)
        profile_min = np.amin(profile)

        step = (x2 - x1)/100
        a = np.arange(x1, x2, step)
        axes.fill_between(a, profile_min, profile_max, alpha=0.2, color='r')
        axes.plot([x1, x1], [profile_min, profile_max], 'r')
        axes.plot([x2, x2], [profile_min, profile_max], 'r')
        return axes

    def rdf_mask(self, profile, line1, line2, axes):
        x1 = np.size(profile) * line1
        x2 = np.size(profile) * line2
        profile[np.isnan(profile)] = 0

        profile_max = np.amax(profile)
        profile_min = np.amin(profile)

        step = (x2 - x1)/100
        a = np.arange(x1, x2, step)
        axes.fill_between(a, profile_min, profile_max, alpha=0.2, color='r')
        axes.plot([x1, x1], [profile_min, profile_max], 'r')
        axes.plot([x2, x2], [profile_min, profile_max], 'r')
        return axes

    def get_dict_info(self, dict_info):
        self.path = None
        self.profile_map = dict_info['profile_map']
        self.sv_shape_x.set(dict_info['scanning_shape'][0])
        self.sv_shape_y.set(dict_info['scanning_shape'][1])

    def return_click_position(self):
        self.parent.rdf_return_position(self.click_position1, self.click_position2)

class SimulationFrame(tk.Frame):
    def __init__(self, frame, parent):
        super().__init__(frame)
        self.parent = parent

        self.path = None
        self.recal_para = None
        self.atom_kinds = None
        self.frame_left = tk.Frame(self, width=10)
        self.frame_left.pack(side=tk.LEFT)
        self.frame_right = tk.Frame(self, width=10)
        self.frame_right.pack(side=tk.RIGHT, fill='both', expand=1)

        self.frame_right_bottom = tk.Frame(self.frame_right)
        self.frame_right_bottom.pack(side=tk.BOTTOM, fill='both', expand=1)
        self.frame_right_top = tk.Frame(self.frame_right)
        self.frame_right_top.pack(side=tk.TOP, fill='x')

        self.setup_left_frame(self.frame_left)
        self.setup_right_frame(self.frame_right_bottom)

        self.pdf_frame = tk.Frame(self.frame_right_top)
        self.pdf_frame.pack(side=tk.LEFT, fill='y')
        self.rdf_frame = tk.Frame(self.frame_right_top)
        self.rdf_frame.pack(side=tk.RIGHT, fill='y')

        self.setup_pdf_frame(self.pdf_frame)
        self.setup_rdf_frame(self.rdf_frame)

    def setup_pdf_frame(self, frame):
        line1 = tk.Frame(frame)
        line1.pack()
        tk.Label(line1, text='cali-factor').grid(row=0, column=0)
        self.cali_sv = tk.StringVar(value='0.00111625')
        self.cali_entry = tk.Entry(line1, textvariable=self.cali_sv, width=10)
        self.cali_entry.grid(row=0, column=1, columnspan=2, sticky='w')

        tk.Label(line1, text='maxAngle').grid(row=1, column=0)
        self.maxangle_sv = tk.StringVar(value='10')
        self.maxangle_entry = tk.Entry(line1, textvariable=self.maxangle_sv, width=6)
        self.maxangle_entry.grid(row=1, column=1)
        tk.Label(line1, text='(Å^-1)').grid(row=1, column=2)

        tk.Label(line1, text='Damping for Str. Factors').grid(row=2, column=0)
        self.pdf_damp_sv = tk.StringVar(value='0.25')
        self.pdf_damp_entry = tk.Entry(line1, textvariable=self.pdf_damp_sv, width=6)
        self.pdf_damp_entry.grid(row=2, column=1)


    def setup_rdf_frame(self, frame):
        line1 = tk.Frame(frame)
        line1.pack()
        tk.Label(line1, text='start').grid(row=0, column=0)
        self.window_start_sv = tk.StringVar(value='0.3')
        self.window_start_entry = tk.Entry(line1, textvariable=self.window_start_sv, width=6)
        self.window_start_entry.grid(row=0, column=1)
        tk.Label(line1 ,text='(Å^-1)').grid(row=0, column=2)
        tk.Label(line1, text='end').grid(row=0, column=3)
        self.window_end_sv = tk.StringVar(value='3')
        self.window_end_entry = tk.Entry(line1, textvariable=self.window_end_sv, width=6)
        self.window_end_entry.grid(row=0, column=4)
        tk.Label(line1, text='(Å^-1)').grid(row=0, column=5)

        tk.Label(line1, text='Max range of RDF').grid(row=1, column=0)
        self.rdf_max_range_sv = tk.StringVar(value='10')
        self.rdf_max_range_entry = tk.Entry(line1, textvariable=self.rdf_max_range_sv, width=6)
        self.rdf_max_range_entry.grid(row=1, column=1)
        tk.Label(line1, text='(Å)').grid(row=1, column=2)

        tk.Label(line1, text='Step length').grid(row=1, column=3)
        self.rdf_step_length_sv = tk.StringVar(value='0.01')
        self.rdf_step_length_entry = tk.Entry(line1, textvariable=self.rdf_step_length_sv, width=6)
        self.rdf_step_length_entry.grid(row=1, column=4)
        tk.Label(line1, text='(Å)').grid(row=1, column=5)

        tk.Label(line1, text='Damping for RDFs').grid(row=2, column=0)
        self.rdf_damp_sv = tk.StringVar(value='0.015')
        self.rdf_damp_entry = tk.Entry(line1, textvariable=self.rdf_damp_sv, width=6)
        self.rdf_damp_entry.grid(row=2, column=1)
        tk.Button(line1, text='update', command=self.re_cal_rdf).grid(row=2, column=3, columnspan=2)

    def setup_left_frame(self, frame):
        tk.Button(frame, text='load file', command=self.load_xyz).pack()
        tk.Button(frame, text='start\n simulation', command=self.rdf_simulation).pack()
        tk.Button(frame, text='save\n simulation results', command=self.save_sim).pack()

    def setup_right_frame(self, frame):
        self.disp_figure = Figure()
        a = self.disp_figure.add_subplot(111)
        canvas_disp = FigureCanvasTkAgg(self.disp_figure, master=frame)
        canvas_disp.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas_disp, frame)
        toolbar.update()
        canvas_disp._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas_disp.draw()
        self.canvas_disp = canvas_disp

    def re_cal_rdf(self):
        parameter = self.get_parameter()
        if self.recal_para is not None:
            PDF, G, Gtot, r, atom_kinds, s, diffs, diftot, recal_para = simulation.recal_damp(self.recal_para, parameter)
            self.recal_para = recal_para
            self.Gtot = Gtot
            self.G = G
            self.r = r
            self.atom_kinds = atom_kinds
            self.s = s
            self.diffs = diffs
            self.diftot = diftot
            self.plot_result()
        else:
            return

    def load_xyz(self):
        path = askopenfilename(filetypes=(('xyz File', '*.xyz'), ("All Files", "*.*")),
                               title="Choose a file.")
        self.path = path

    def save_sim(self):
        path = asksaveasfilename(filetypes=(('Excel File', '*.xlsx'),
                                            ('txt File', '*.txt'),
                                            ("All Files", "*.*")),
                                 title="Choose a file.")
        if path is None or self.atom_kinds is None:
            return
        atom_kinds = self.atom_kinds

        suffix = ['total']
        for i, item in enumerate(atom_kinds):
            for j, item2 in enumerate(atom_kinds[i:]):
                suffix.append(atom_kinds[i] + '-' + atom_kinds[i+j])
        save_data_rdf = [self.Gtot]
        for i in range(np.shape(self.G)[1]):
            save_data_rdf.append(self.G[:, i])
        save_data_pdf = [self.diftot]
        for i in range(np.shape(self.diffs)[1]):
            save_data_pdf.append(self.diffs[:, i])
        if path[-3:] == 'txt':
            self.save_sim_txt(path, suffix, save_data_pdf, save_data_rdf)

        elif path[-4:] == 'xlsx':
            self.save_sim_excel(path, suffix, save_data_pdf, save_data_rdf)

    def save_sim_txt(self, path, suffix, pdf_data, rdf_data):
        save_path_rdf = []
        save_path_pdf = []
        for i, item in enumerate(suffix):
            save_path_pdf.append(path[:-4] + '_pdf_' + suffix[i] + path[-4:])
            save_path_rdf.append(path[:-4] + '_rdf_' + suffix[i] + path[-4:])
        with open(path[:-4] + '_pdf_r' + path[-4:], 'wb+') as file:
            np.savetxt(file, self.r)
        with open(path[:-4] + '_rdf_s' + path[-4:], 'wb+') as file:
            np.savetxt(file, self.s)
        for i, item in enumerate(save_path_rdf):
            with open(save_path_pdf[i], 'wb+') as file:
                np.savetxt(file, pdf_data[i], delimiter=",", newline="\r\n")
            with open(item, 'wb+') as file:
                np.savetxt(file, rdf_data[i], delimiter=",", newline="\r\n")

    def save_sim_excel(self, path, suffix, save_data_pdf, save_data_rdf):
        temp_rdf_data = {'r(Å)': self.r}
        temp_pdf_data = {'s(Å^-1)': self.s}
        for i, item in enumerate(suffix):
            temp_pdf_data[item] = save_data_pdf[i]
            temp_rdf_data[item] = save_data_rdf[i]
        rdf_data = pandas.DataFrame(temp_rdf_data)
        pdf_data = pandas.DataFrame(temp_pdf_data)

        with pandas.ExcelWriter(path) as writer:
            pdf_data.to_excel(writer, sheet_name='PDF Data', index=False)
            rdf_data.to_excel(writer, sheet_name='RDF Data', index=False)

    def rdf_simulation(self):
        if self.path is not None:
            parameter = self.get_parameter()
            PDF, G, Gtot, r, atom_kinds, s, diffs, diftot, recal_para = simulation.simulation_with_xyz(self.path, parameter)
        else:
            return
        self.recal_para = recal_para
        self.Gtot = Gtot
        self.G = G
        self.r = r
        self.atom_kinds = atom_kinds
        self.s = s
        self.diffs = diffs
        self.diftot = diftot
        self.plot_result()

    def get_parameter(self):
        parameter = {}
        try:
            califactor = float(self.cali_sv.get())
        except ValueError:
            califactor = 0.00111625
            self.cali_sv.set(califactor)
        parameter['cali'] = califactor

        try:
            maxangle = float(self.maxangle_sv.get())
        except ValueError:
            maxangle = 10
            self.maxangle_sv.set(maxangle)
        parameter['maxangle'] = maxangle

        try:
            pdf_damp = float(self.pdf_damp_sv.get())
        except ValueError:
            pdf_damp = 0.25
            self.pdf_damp_sv.set(pdf_damp)
        parameter['pdf_damp'] = pdf_damp

        try:
            window_start = float(self.window_start_sv.get())
        except ValueError:
            window_start = 0.3
            self.window_start_sv.set(window_start)
        parameter['window_start'] = window_start

        try:
            window_end = float(self.window_end_sv.get())
        except ValueError:
            window_end = 3
            self.window_end_sv.set(window_end)
        parameter['window_end'] = window_end

        try:
            max_range = float(self.rdf_max_range_sv.get())
        except ValueError:
            max_range = 10
            self.rdf_max_range_sv.set(max_range)
        parameter['max_range'] = max_range

        try:
            step_length = float(self.rdf_step_length_sv.get())
        except ValueError:
            step_length = 0.01
            self.rdf_step_length_sv.set(step_length)
        parameter['step_length'] = step_length

        try:
            rdf_damp = float(self.rdf_damp_sv.get())
        except ValueError:
            rdf_damp = 0.015
            self.rdf_damp_sv.set(rdf_damp)
        parameter['rdf_damp'] = rdf_damp
        return parameter

    def plot_result(self):
        s = self.s
        diftot = self.diftot
        diffs = self.diffs
        atom_kinds = self.atom_kinds
        G = self.G
        r = self.r
        Gtot = self.Gtot

        # print(atom_kinds)
        self.disp_figure.clf()
        ds = s[1] - s[0]
        xlimit = [0.1, 3]
        xlim1 = int(xlimit[0] / ds)
        xlim2 = int(xlimit[1] / ds)
        ylimit = [diftot[xlim2] - 0.1 * (diftot[xlim1] - diftot[xlim2]),
                  diftot[xlim1] + 0.3 * (diftot[xlim1] - diftot[xlim2])]
        f1 = self.disp_figure.add_subplot(221, title='Diffraction', xlim=xlimit, xlabel='Å^-1',
                                          ylim=ylimit)
        f1.plot(s, diftot)

        f2 = self.disp_figure.add_subplot(222, title='total RDF', xlabel='Å')
        f2.plot(r, Gtot)

        ymax = np.amax(diffs[xlim1:xlim2, :])
        ymin = np.amin(diffs[xlim1:xlim2, :])

        ylimit = [ymin - 0.1 * (ymax - ymin),
                  ymax + 0.3 * (ymax - ymin)]
        f3 = self.disp_figure.add_subplot(223, title='Structure Factors', xlim=xlimit, xlabel='Å^-1',
                                          ylim=ylimit)
        count = 0
        for i, item1 in enumerate(atom_kinds):
            for j, item2 in enumerate(atom_kinds[i:]):
                title = atom_kinds[i] + '-' + atom_kinds[i + j]
                f3.plot(s, diffs[:, count], label=title)
                count += 1
        f3.legend(loc=1)

        f4 = self.disp_figure.add_subplot(224, title='partial RDFs', xlabel='Å')
        count = 0
        for i, item1 in enumerate(atom_kinds):
            for j, item2 in enumerate(atom_kinds[i:]):
                title = atom_kinds[i] + '-' + atom_kinds[j + i]
                f4.plot(r, G[:, count], label=title)
                count += 1
        # f2.plot(r, G[:, 0], label=atom_kinds[0] + '-' + atom_kinds[0])
        # f2.plot(r, G[:, 1], label=atom_kinds[0] + '-' + atom_kinds[1])
        # f2.plot(r, G[:, 2], label=atom_kinds[1] + '-' + atom_kinds[1])
        f4.legend(loc=1)
        self.disp_figure.subplots_adjust(left=0.06, bottom=0.06,
                                         right=0.95, top=0.95,
                                         wspace=0.2, hspace=0.28)
        self.canvas_disp.draw()



class CompareFrame(tk.Frame):
    def __init__(self, frame, parent):
        super().__init__(frame)
        tk.Label(self, text=111).pack(side=tk.RIGHT)
        self.parent = parent


class ImageWindow(tk.Toplevel):
    def __init__(self, fig):
        super().__init__()
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        canvas.draw()


class RdfParaData:
    def __init__(self, initial_condition):
        if initial_condition:
            self.element = [26, 5, 14]
            self.composition = [0.8, 0.2, 0.1]
            self.phi_fit_begin = 1200
            self.phi_fit_end = 1500
            self.pixel_begin = 60
            self.pixel_end = 1600
            self.pixel_adjust_begin = 0
            self.pixel_adjust_end = 0
            self.pixel_adjust = 0
            self.smooth_en = 1
            self.smooth_range = 0.25
            self.smooth_strength = 3
            self.polyfitN_en = 1
            self.polyfitN = 5
            self.califactor = 0.0021745
            self.damping_en = 0
            self.damping_strength = 0
            self.damping_start_point = 10000

            self.autodark_en = 1

            self.aver_inten = None
            self.L = 10
            self.rn = 1000


class RdfParaDict:
    dict = {
        'element': [26, 5, 14],
        'composition': [0.8, 0.2, 0.1],
        'phi_fit_begin': 1200,
        'phi_fit_end': 1500,
        'pixel_begin': 60,
        'pixel_end': 1600,
        'pixel_adjust_begin': 0,
        'pixel_adjust_end': 0,
        'pixel_adjust': 0,
        'smooth_en': 1,
        'smooth_range': 0.25,
        'smooth_strength': 3,
        'polyfitN': 5,
        'califactor': 0.0021745,
        'damping_en': 0,
        'damping_strength': 0,
        'damping_start_point': 10000,

        'autodark_en': 1,

        'aver_inten': None,
        'L': 10,
        'rn': 1000
    }

    @classmethod
    def set_value(cls, key, value):
            cls.dict[key] = value


class CommandButton:
    def __init__(self, parent, frame):
        self.frame = frame
        self.parent = parent

    def add_new_button(self, name, text=None):
        if text is None:
            text = name
        setattr(self, name, tk.Button(self.frame, text=text,
                                      width=12, height=3,
                                      command=eval('self.parent.%s' % name)))
        eval('self.%s.pack()' % name)


class ErrorWindow(tk.Toplevel):
    def __init__(self, error_number='000'):
        super().__init__()
        self.title('Error')
        self.error_number = error_number
        tk.Label(self, text=error_list(self.error_number)).pack()


class GetDataShape(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title('get data shape')

        self.userinfo = [1, 1]
        width = self.winfo_screenwidth()
        height = self.winfo_screenheight()
        self.geometry("%dx%d+%d+%d" % (0.1 * width, 0.1 * height, 0.45 * width, 0.45 * height))

        self.frame = tk.Frame(self)
        self.frame.pack()
        self.x_data = tk.StringVar(self.frame, value='1')
        self.y_data = tk.StringVar(self.frame, value='1')
        tk.Label(self.frame, text='enter data shape').pack()
        row0 = tk.Frame(self.frame)
        row0.pack(fill='x', expand=1)
        self.entry1 = tk.Entry(row0, textvariable=self.x_data, width=4)
        self.entry1.pack(side=tk.LEFT)
        tk.Label(row0, text=',').pack(side=tk.LEFT)
        self.entry2 = tk.Entry(row0, textvariable=self.y_data, width=4)
        self.entry2.pack(side=tk.LEFT)

        row2 = tk.Frame(self.frame)
        row2.pack(fill='x', expand=1)
        tk.Button(row2, command=self.confirm, text='ok').pack(side=tk.LEFT)
        tk.Button(row2, command=self.cancel, text='cancel').pack(side=tk.RIGHT)

    def confirm(self):
        x = 1
        y = 1
        try:
            x = self.x_data.get()
            y = self.y_data.get()
            x = int(x)
            y = int(y)
            print(x, y)
        except:
            self.destroy()

        if x > 0 and y > 0:
            self.userinfo = [x, y]
        self.destroy()


    def cancel(self):
        self.destroy()


def error_list(error_number='000'):
    error = {
        '000': "ERROR CODE: 000\nunknown error",

        '001': "ERROR CODE: 001\n"
               "the numbers of the elements\n "
               "and compositions are not equal",

        '002': "ERROR CODE: 002\n"
               "images has not been loaded",

        '003': "ERROR CODE: 003\n"
               "images has not been loaded or\n"
               "beam stopper area has been not set",

        '004': "ERROR CODE: 004\n"
               "radial intensity profile do not exist",

        '005': "ERROR CODE: 005\n"
               "file can not been loaded",
        "006": "ERROR CODE: 006\n"
               "can not open file",

        '007': "ERROR CODE: 007\n" 
               "recheck the parameters table",

        '008': "ERROR CODE: 008\n"
               "update_error"
    }
    try:
        return error[error_number]
    except:
        return error['000']


def swap_position(position1=None, position2=None):
    _position1 = None
    _position2 = None
    if position1 is not None and position2 is not None:
        _position1 = [np.minimum(position1[0], position2[0]),
                      np.minimum(position1[1], position2[1])]
        _position2 = [np.maximum(position1[0], position2[0]),
                      np.maximum(position1[1], position2[1])]
    return _position1, _position2


def handlerAdaptor(fun, **kwargs):
    return lambda event, fun=fun, kwds=kwargs: fun(event, **kwds)


def dict2struct(dict, master):
    for i, item in enumerate(list(dict.values())):
        if not (item is ''):
            try:
                result = str2list(item)
                if len(result) > 1:
                    setattr(master, list(dict.keys())[i], result)
                else:
                    setattr(master, list(dict.keys())[i], result[0])
            except:
                setattr(master, list(dict.keys())[i], result)
    return master


def str2list(string):
    """Converts a string of space seperated numbers to a list"""
    try:
        string = ' ' + string + ' '
    except:
        return string
    num_list = []
    begin = 0
    int_part = None
    decimal_part = None
    deci_en = 0
    for i, item1 in enumerate(string[0: -1]):
        item2 = string[i+1]
        try:
            int(item1)
            try:
                int(item2)
                continue
            except:
                if item2 is '.':
                    int_part = int(string[begin: i+1])
                    deci_en = 1
                else:
                    if deci_en:
                        decimal_part = np.divide(int(string[begin:i+1]), np.power(10, i+1-begin))
                        num = int_part + decimal_part
                    else:
                        num = int(string[begin: i+1])
                    num_list.append(num)
                    int_part = None
                    decimal_part = None
                    deci_en = 0
        except:
            try:
                int(item2)
                if item1 is '.':
                    begin = i+1
                else:
                    begin = i
            except:
                if item2 is '.':
                    int_part = 0
                    deci_en = 1
                    continue
                if not (int_part is None and decimal_part is None):
                    if int_part is None:
                        num = decimal_part
                    elif decimal_part is None:
                        num = int_part
                    else:
                        num = decimal_part + int_part
                    num_list.append(num)
                    int_part = None
                    decimal_part = None
                    deci_en = 0
    return num_list


def display_histogram(fig, mode=1):
    line = np.array(fig).reshape(np.size(fig))
    if mode == 1:
        value, index = np.histogram(line, bins=np.array(np.arange(np.amin(line), np.amax(line))))
    else:
        _max = np.amax(line)
        _min = np.amin(line)
        value, index = np.histogram(line, bins=np.array(np.arange(_min,
                                                                  _max,
                                                                  (_max-_min)/300)))
    a = Figure()
    a.add_subplot(111).plot(index[0:-1], value)
    ImageWindow(a)


if __name__ == '__main__':
    try:
        ctx =lt.Context()
    except:
        print("Libertem is missing")
    window = MainGui()
    window.mainloop()
    

