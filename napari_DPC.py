# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 00:16:58 2022

"""

import napari
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QSplitter, QWidget, QPushButton
from qtpy.QtWidgets import QComboBox,QLabel, QFormLayout, QVBoxLayout, QSpinBox, QDoubleSpinBox, QCheckBox

from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import sys

import numpy as np

import tifffile

from utils import pDPC_reconstruction, normalize99, binning_img #, ImgToReg 

# from skimage.morphology import remove_small_objects

from cellpose import models
import torch

import cv2 as cv

#-----------------------------------------------------------------------                
class Settings():
    ''' 
    Auxilliary class to create an object with a corresponding Qwidget,
    and update its value as a property (self.val)-
    - name of the QWidget (it contain a label)
    - dtype: Currently supported for int and float 
    - initial_value: stored in the @property self.val
    - vmin, vmax: min and max values of the QWidget
    - layout: parent Qlayout    
    - read function: not implemented
    - write_function is executed on value change of the QWidget
    
    '''
    
    def __init__(self, name ='settings_name',
                 dtype = int,
                 initial_value = 0,
                 vmin = 0,
                 vmax = 2**16-1,
                 layout = None,
                 write_function = None,
                 read_function = None):
        
        self.name= name
        self._val = initial_value
        self.write_function = write_function
        self.read_function = read_function
        self.create_spin_box(layout, dtype, vmin, vmax)
        
    @property    
    def val(self):
        self._val = self.sbox.value()
        return self._val 
    
    @val.setter 
    def val(self, new_val):
        self.sbox.setValue(new_val)
        self._val = new_val
        
    def create_spin_box(self, layout, dtype, vmin, vmax):
        name = self.name
        val = self._val
        if dtype == int:
            sbox = QSpinBox()
            sbox.setMaximum(vmax)
            sbox.setMinimum(vmin)
        elif dtype == float:
            sbox = QDoubleSpinBox()
            sbox.setDecimals(3)
            sbox.setSingleStep(0.1)
            sbox.setMaximum(2**16-1)
        
        else: raise(TypeError, 'Specified setting type not supported')
        sbox.setValue(val)
        if self.write_function is not None:
            sbox.valueChanged.connect(self.write_function)
        settingLayout = QFormLayout()
        settingLayout.addRow(QLabel(name), sbox)
        layout.addLayout(settingLayout)
        self.sbox = sbox
#-----------------------------------------------------------------------                        
class DPC_widget(QWidget):
        
    SGM_FLUOR_MODE_DICT = {0:'DPC', 1:'fluor'}
    
    def __init__(self, viewer:napari.Viewer,
                 ):
        self.viewer = viewer
        super().__init__()
        
        self.dpc_reconstruction_settings_dict = {'seq_case':int(7),
                          'binning':int(1),
                          'mag':float(10.00),
                          'NA_img':float(0.25),
                          'NA_illu':float(0.55),
                          'NA_inner':float(0.00),                                                 
                          'wavelength_um':float(0.6),
                          'px_cam_um':float(3.45)
                          }    
    
        self.cellpose_segmentation_settings_dict = {
                         'diameter':float(30.0),
                         'cellprob_threshold':float(0.00),
                         'flow_threshold':float(0.4),
                         'min_size_pix':float(400),
                         }    
    
        self.cellpose_model_dict = {0:'cyto', 1:'nuclei'}
        self.model = None
    
        self.pDPCsol = pDPC_reconstruction()
        
        self.img_optosplit = None
        self.img_trinoc = None
        self.img_dpc = None
        
        self.shape = None        
        self.img_fluos = None
        self.img_dpcs = None
        
        self.current_filename = None
        
        self.setup_ui()                 
        
        # hardcoded for test example
        self.affines = {'optosplit_left': np.asarray([[ 1.61741444e+00,  2.27390700e-02, -2.31129972e+02],
            [-4.88988964e-04,  1.57668493e+00, -5.69609046e+02]]), 'trinoc': np.asarray([[ 9.45148412e-01,  1.08530112e-02, -3.32527084e+02],
            [-9.17187269e-03,  9.45341990e-01, -2.41914795e+02]]), 'optosplit_right': np.asarray([[ 1.62779431e+00,  1.29697257e-02, -1.36556917e+03],
            [-2.90882726e-03,  1.58147100e+00, -5.67097946e+02]])}
     #-----------------------------------------------------------------------           
    def setup_ui(self):     
        
        def add_section(_layout,_title):
            splitter = QSplitter(Qt.Vertical)
            _layout.addWidget(splitter)
            _layout.addWidget(QLabel(_title))
            
        # initialize layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.flip_trinoc_checkbox = QCheckBox("Flip trinoc")
        self.flip_trinoc_checkbox.setChecked(True)
        layout.addWidget(self.flip_trinoc_checkbox)
        
        btn = QPushButton('Load registration chart')
        btn.clicked.connect(self.load_registration_chart)
        layout.addWidget(btn)
        
        btn = QPushButton('Perform registration')
        btn.clicked.connect(self.perform_registration)
        layout.addWidget(btn)   
        
        add_section(layout,'')  
        
        btn = QPushButton('Load FOV')
        btn.clicked.connect(self.load_FOV_image)
        layout.addWidget(btn)  

        add_section(layout,'')        

        add_section(layout,'DPC reconstruction settings')
        DPC_settings_layout = QVBoxLayout()
        layout.addLayout(DPC_settings_layout)
        self.create_Settings(DPC_settings_layout, self.dpc_reconstruction_settings_dict)
         
        add_section(layout,'')
        add_section(layout,'Cellpose segmentation source:')      
        self.sgm_source_combo = QComboBox()
        self.sgm_source_combo.addItems(list(self.SGM_FLUOR_MODE_DICT.values()))
        layout.addWidget(self.sgm_source_combo)        
                
        add_section(layout,'model_type:')
        self.modeltype_combo = QComboBox()
        self.modeltype_combo.addItems(list(self.cellpose_model_dict.values()))
        self.modeltype_combo.currentIndexChanged.connect(self.on_model_type_change)
        layout.addWidget(self.modeltype_combo)
        self.on_model_type_change()
              
        sgm_settings_layout = QVBoxLayout()
        layout.addLayout(sgm_settings_layout)
        self.create_Settings(sgm_settings_layout,self.cellpose_segmentation_settings_dict)
        
        self.cellpose_resample_checkbox = QCheckBox("resample")
        self.cellpose_resample_checkbox.setChecked(True)
        layout.addWidget(self.cellpose_resample_checkbox) 
                
        add_section(layout,'')
        
        btn = QPushButton('Process frame')
        btn.clicked.connect(self.process_current_frame)
        layout.addWidget(btn)
        
        btn = QPushButton('Process current FOV')
        btn.clicked.connect(self.process_current_FOV)
        layout.addWidget(btn)
        
        btn = QPushButton('Process FOVs batch')
        btn.clicked.connect(self.process_FOVs_batch)
        layout.addWidget(btn)
        
        add_section(layout,'')
    #-----------------------------------------------------------------------                
    def create_Settings(self, slayout, s_dict):
        for key, val in s_dict.items():
            new_setting = Settings(name=key, dtype=type(val), initial_value=val,
                                    layout=slayout,
                                    write_function=None) #?
            setattr(self, key, new_setting)              
    #-----------------------------------------------------------------------                
    def process_current_frame(self):
        self.updateTransferFunc()
        ind = self.viewer.dims.current_step[0]
        img, img_dpc_rec, fluor_ch1, fluor_ch2, masks = self.process_frame(ind)
        # ?
        image_layer = self.viewer.add_image(img, name='f = ' + str(ind)) 
        labels_layer = self.viewer.add_labels(masks, name='sgm f = ' + str(ind))
    #-----------------------------------------------------------------------                        
    def process_frame(self,ind):    
        fluor_ch1 = None
        fluor_ch2 = None
        masks = None
        img = None
        try:                                    
            img_dpc = np.squeeze(self.img_dpcs[ind,:,:])           
                        
            img_dpc_rec = self.pDPCsol.reconstructPDPC(img_dpc, self.seq_case.val, self.binning.val)
            #
            shape = img_dpc_rec.shape[::-1]
            img = self.img_fluos[ind,:,:]
            fluor_ch1 = cv.warpAffine(img.astype('float32'), self.affines['optosplit_left'], shape)
            fluor_ch2 = cv.warpAffine(img.astype('float32'), self.affines['optosplit_right'], shape)
                                    
            if 'DPC'==self.sgm_source_combo.currentText():
                img = img_dpc_rec
            else:
                if 'fluor'==self.sgm_source_combo.currentText():
                    img = fluor_ch1
            
            img = normalize99(img)             
            
            masks, flows, styles, diams = self.model.eval(img, 
            diameter = self.diameter.val, 
            cellprob_threshold = self.cellprob_threshold.val,
            flow_threshold = self.flow_threshold.val,
            resample = self.cellpose_resample_checkbox.isChecked(),            
            channels=[0,0])                                   
        except:
            print('error!')
        return img, img_dpc_rec, fluor_ch1, fluor_ch2, masks            
    #-----------------------------------------------------------------------
    def process_current_FOV(self):        
        # presuming, the image is loaded
        try:                        
            self.updateTransferFunc()
            
            n_frames = self.img_dpcs.shape[0]
            
            v = np.array(0)
                        
            for i in napari.utils.progress(range(n_frames)):
                                
                img, img_dpc_rec, fluor_ch1, fluor_ch2, masks = self.process_frame(i)
                
                if 1==v.size:
                    w,h = img_dpc_rec.shape
                    v = np.zeros((n_frames,4,w,h),dtype=np.float32)               
                
                v[i,0,:,:] = fluor_ch1 
                v[i,1,:,:] = fluor_ch2
                v[i,2,:,:] = img_dpc_rec
                v[i,3,:,:] = masks
                #    
                print(i)
            
            extension = '.tif'
            if '.ome.tif' in  self.current_filename:
                extension = '.ome.tif'
            # 
            savename = self.current_filename.replace(extension, '_out_' + extension)                        
            tifffile.imwrite(savename,v)       
            
        except:
            print('error!')
    #-----------------------------------------------------------------------        
    def process_FOVs_batch(self):
        files = self.openFileNamesDialog()  
        try:        
            for fname in files:       
                self.load_fov_image(fname,False)
                self.process_current_FOV()                
        except:
                print('process_FOVs_batch - error')
    #-----------------------------------------------------------------------                      
    def load_FOV_image(self):
        self.load_fov_image(None,True) 
    #-----------------------------------------------------------------------                
    def load_fov_image(self,filename = None, verbose = False): 
       try:
           if None != filename:
              self.current_filename = filename
           else:                   
               self.current_filename = self.openFileNameDialog()             
           imgs = tifffile.imread(self.current_filename)
           if len(imgs.shape)==4:
                self.img_fluos = np.flip(np.squeeze(imgs[:,0,:,:]), axis = 1)
                self.img_dpcs = np.squeeze(imgs[:,1,:,:])
           elif len(imgs.shape)==3:
                self.img_fluos = np.flip(np.squeeze(imgs[::2,:,:]), axis = 1)
                self.img_dpcs = np.squeeze(imgs[1::2,:,:])                
           else:
                print(imgs.shape)      
               
           self.shape = (self.img_dpcs.shape[1]//2,self.img_dpcs.shape[2]//2)
           
           if verbose:              
               self.viewer.add_image(self.img_dpcs, name='dpcs raw')
               self.viewer.add_image(self.img_fluos, name='fluos raw')
                
       except IOError as e:
            print('Unable to open file : error '  + e.strerror)                
    #-----------------------------------------------------------------------                                   
    def updateTransferFunc(self):
        if self.shape==None:
            print(sys._getframe().f_code.co_name)
            return
        else:
            self.pDPCsol.updateTransferFunc(self.shape, self.mag.val, 
                                        self.NA_img.val, self.NA_illu.val, self.NA_inner.val, 
                                        self.wavelength_um.val, self.px_cam_um.val)    
    #-----------------------------------------------------------------------                                
    def on_model_type_change(self):
       GPU = True 
       if not torch.cuda.is_available():
          print('not using GPU')
          GPU = False
       self.model = models.Cellpose(gpu=GPU,model_type = self.modeltype_combo.currentText())        
    #-----------------------------------------------------------------------                
    def openFileNamesDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        files, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Python Files (*.py)", options=options)
        if files:
            return files
        return None
    #-----------------------------------------------------------------------        
    def openFileNameDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            return fileName
        return None
    #-----------------------------------------------------------------------                                       
    def load_registration_chart(self):
        filename = self.openFileNameDialog()  
        try:
            img = tifffile.imread(filename)
            self.img_optosplit = np.flipud(np.squeeze(img[0,:,:]))
            if self.flip_trinoc_checkbox.isChecked():
                self.img_trinoc = np.fliplr(np.squeeze(img[2,:,:]))
            else:
                self.img_trinoc = np.squeeze(img[2,:,:])
            self.img_dpc = binning_img(np.squeeze(img[1,:,:]), 2) 
            self.viewer.add_image(self.img_optosplit, name='chart optosplit')
            self.viewer.add_image(self.img_trinoc, name='chart trinoc')
            self.viewer.add_image(self.img_dpc, name='chart dpc')
        except IOError as e:
            print('Unable to open file : error '  + e.strerror)
            # error_log.write('Unable to open foo : %s\n' % e)      
    #-----------------------------------------------------------------------            
    def perform_registration(self):            
        print('...') 
        
if __name__ == '__main__':
   
    viewer = napari.Viewer()
    widget = DPC_widget(viewer)
    viewer.window.add_dock_widget(widget,
                                  name = 'DPC @Imperial',
                                  add_vertical_stretch = True)
    napari.run()      