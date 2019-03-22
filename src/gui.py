from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollArea, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QMainWindow, QFormLayout, QMessageBox)

from functools import partial

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import sys
import random
import numpy as np
import os

import src.detTOD as tod
import src.loaddata as ld
import src.mapmaker as mp

class MainWindow(QTabWidget):
    
    def __init__(self, parent = None):
        super(MainWindow, self).__init__(parent)
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
            
        self.addTab(self.tab1,"Parameters and Maps")
        self.addTab(self.tab2,"Detector TOD")
        self.ParamMapLayout()
        self.TODLayout()
        self.setWindowTitle("Naive MapMaker")
        
        self.show()

    def ParamMapLayout(self):

        self.createAstroGroup()
        self.createExperimentGroup()
        self.createDataRepository()
        self.plotbutton = QPushButton('Plot')
        self.createMapPlotGroup()

        scroll = QScrollArea()
        scroll.setWidget(self.ExperimentGroup)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(100)

        ExperimentGroup_Scroll = QGroupBox("Experiment Parameters")
        ExperimentGroup_Scroll.setLayout(QVBoxLayout())
        ExperimentGroup_Scroll.layout().addWidget(scroll)

        self.plotbutton.clicked.connect(self.load_func)
        
        mainlayout = QGridLayout()
        mainlayout.addWidget(self.DataRepository, 0, 0)
        mainlayout.addWidget(self.AstroGroup, 1, 0)
        mainlayout.addWidget(ExperimentGroup_Scroll, 2, 0)
        mainlayout.addWidget(self.plotbutton)
        mainlayout.addWidget(self.MapPlotGroup, 0, 1, 2, 1)
        
        self.tab1.setLayout(mainlayout)

    def createDataRepository(self):
        self.DataRepository = QGroupBox("Data Repository")
        
        self.detpath = QLineEdit('')
        self.detpathlabel = QLabel("Detector Path:")
        self.detpathlabel.setBuddy(self.detpath)

        self.detname = QLineEdit('')
        self.detnamelabel = QLabel("Detector Name:")
        self.detnamelabel.setBuddy(self.detname)

        self.detvalue = np.array([])

        self.roachnumber = QLineEdit('')
        self.roachnumberlabel = QLabel("Roach Number:")
        self.roachnumberlabel.setBuddy(self.roachnumber)

        self.coordpath = QLineEdit('')
        self.coordpathlabel = QLabel("Coordinate Path:")
        self.coordpathlabel.setBuddy(self.coordpath)

        self.coord1value = np.array([])
        self.coord2value = np.array([])

        self.DettableCheckBox = QCheckBox("Use Detector Table")
        self.DettableCheckBox.setChecked(False)
        self.dettablepath = QLineEdit('')
        self.dettablepathlabel = QLabel("Detector Table Path:")
        self.dettablepathlabel.setBuddy(self.dettablepath)

        self.DettableCheckBox.toggled.connect(self.dettablepathlabel.setVisible)
        self.DettableCheckBox.toggled.connect(self.dettablepath.setVisible)

        self.PointingOffsetCheckBox = QCheckBox("Use Pointing Offset")
        self.PointingOffsetCheckBox.setChecked(False)
        self.pointingoffsetpath = QLineEdit('')
        self.pointingoffsetpathlabel = QLabel("Pointing Offset Table Path:")
        self.pointingoffsetpathlabel.setBuddy(self.pointingoffsetpath)

        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetpathlabel.setVisible)
        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetpath.setVisible)

        self.layout = QGridLayout()

        self.layout.addWidget(self.detpathlabel, 0, 0)
        self.layout.addWidget(self.detpath, 0, 1, 1, 2)
        self.layout.addWidget(self.detnamelabel, 1, 0)
        self.layout.addWidget(self.detname, 1, 1, 1, 2)
        self.layout.addWidget(self.roachnumberlabel, 2, 0)
        self.layout.addWidget(self.roachnumber, 2, 1, 1, 2)
        self.layout.addWidget(self.coordpathlabel, 3, 0)
        self.layout.addWidget(self.coordpath, 3, 1, 1, 2)
        self.layout.addWidget(self.DettableCheckBox, 4, 0)
        self.layout.addWidget(self.dettablepathlabel, 5, 1)
        self.layout.addWidget(self.dettablepath, 5, 2)
        self.layout.addWidget(self.PointingOffsetCheckBox, 6, 0)
        self.layout.addWidget(self.pointingoffsetpathlabel, 7, 1)
        self.layout.addWidget(self.pointingoffsetpath, 7, 2)

        self.dettablepathlabel.setVisible(False)
        self.dettablepath.setVisible(False)
        self.pointingoffsetpathlabel.setVisible(False)
        self.pointingoffsetpath.setVisible(False)

        self.DataRepository.setLayout(self.layout)

    def createAstroGroup(self):
        self.AstroGroup = QGroupBox("Astronomy Parameters")
    
        self.coordchoice = QComboBox()
        self.coordchoice.addItem('RA and DEC')
        self.coordchoice.addItem('AZ and EL')
        self.coordchoice.addItem('CROSS-EL and EL')
        coordLabel = QLabel("Coordinates System:")
        coordLabel.setBuddy(self.coordchoice)

        self.convchoice = QComboBox()
        self.convchoice.addItem('Not Apply')
        self.convchoice.addItem('Gaussian')
        convLabel = QLabel("Map Convolution:")
        convLabel.setBuddy(self.convchoice)

        self.GaussianSTD = QLineEdit('')
        self.gaussianLabel = QLabel("Convolution STD (in arcsec):")
        self.gaussianLabel.setBuddy(self.GaussianSTD)

        self.crpix1 = QLineEdit('')
        self.crpix2 = QLineEdit('')
        self.crpixlabel = QLabel("CRpix of the Map:")
        self.crpixlabel.setBuddy(self.crpix1)

        self.cdelt1 = QLineEdit('')
        self.cdelt2 = QLineEdit('')
        self.cdeltlabel = QLabel("Cdelt of the Map in deg:")
        self.cdeltlabel.setBuddy(self.cdelt1)

        self.cval1 = QLineEdit('')
        self.cval2 = QLineEdit('')
        self.cvallabel = QLabel("Cval of the Map in deg:")
        self.cvallabel.setBuddy(self.cval1)

        self.pixnum1 = QLineEdit('')
        self.pixnum2 = QLineEdit('')
        self.pixnumlabel = QLabel("Pixel Number:")
        self.pixnumlabel.setBuddy(self.pixnum1)

        self.ICheckBox = QCheckBox("Map only I")
        self.ICheckBox.setChecked(True)
        
        self.convchoice.activated[str].connect(self.updateGaussian)
           
        self.layout = QGridLayout()
        self.layout.addWidget(coordLabel, 0, 0)
        self.layout.addWidget(self.coordchoice, 0, 1, 1, 2)
        self.layout.addWidget(convLabel, 1, 0)
        self.layout.addWidget(self.convchoice, 1, 1, 1, 2)
        self.layout.addWidget(self.gaussianLabel, 2, 1)
        self.layout.addWidget(self.GaussianSTD, 2, 2)
        self.layout.addWidget(self.crpixlabel, 3, 0)
        self.layout.addWidget(self.crpix1, 3, 1)
        self.layout.addWidget(self.crpix2, 3, 2)
        self.layout.addWidget(self.cdeltlabel, 4, 0)
        self.layout.addWidget(self.cdelt1, 4, 1)
        self.layout.addWidget(self.cdelt2, 4, 2)
        self.layout.addWidget(self.cvallabel, 5, 0)
        self.layout.addWidget(self.cval1, 5, 1)
        self.layout.addWidget(self.cval2, 5, 2)
        self.layout.addWidget(self.pixnumlabel, 6, 0)
        self.layout.addWidget(self.pixnum1, 6, 1)
        self.layout.addWidget(self.pixnum2, 6, 2)
        self.layout.addWidget(self.ICheckBox, 7, 0)
        self.GaussianSTD.setVisible(False)
        self.gaussianLabel.setVisible(False)
        
        self.AstroGroup.setLayout(self.layout)

    def updateGaussian(self, text=None):
        if text is None:
            text = self.convchoice.currentText()

        if text == 'Gaussian': 
            self.GaussianSTD.setVisible(True)
            self.GaussianSTD.setEnabled(True)
            self.gaussianLabel.setVisible(True)
            self.gaussianLabel.setEnabled(True)
        else: 
            self.GaussianSTD.setVisible(False)
            self.GaussianSTD.setEnabled(False)
            self.gaussianLabel.setVisible(False)
            self.gaussianLabel.setEnabled(False)

    def createExperimentGroup(self):
        self.ExperimentGroup = QGroupBox()

        self.experiment = QComboBox()
        self.experiment.addItem('BLASTPol')
        self.experiment.addItem('BLAST-TNG')
        self.experimentLabel = QLabel("Experiment:")
        self.experimentLabel.setBuddy(self.experiment)

        self.detfreq = QLineEdit('')
        self.detfreqlabel = QLabel("Detector Frequency Sample")
        self.detfreqlabel.setBuddy(self.detfreq)
        self.acsfreq = QLineEdit('')
        self.acsfreqlabel = QLabel("ACS Frequency Sample")
        self.acsfreqlabel.setBuddy(self.acsfreq)

        self.highpassfreq = QLineEdit('')
        self.highpassfreqlabel = QLabel("High Pass Filter cutoff frequency")
        self.highpassfreqlabel.setBuddy(self.highpassfreq)

        self.detframe = QLineEdit('')
        self.detframelabel = QLabel("Detector Samples per Frame")
        self.detframelabel.setBuddy(self.detframe)
        self.acsframe = QLineEdit('')
        self.acsframelabel = QLabel("ACS Sample Samples per Frame")
        self.acsframelabel.setBuddy(self.acsframe)

        self.startframe = QLineEdit('')
        self.endframe = QLineEdit('')
        self.numberframelabel = QLabel('Starting and Ending Frames')
        self.numberframelabel.setBuddy(self.startframe)

        self.dettype = QLineEdit('')
        self.dettypelabel = QLabel("Detector DIRFILE data type")
        self.dettypelabel.setBuddy(self.dettype)
        self.coord1type = QLineEdit('')
        self.coord1typelabel = QLabel("Coordinate 1 DIRFILE data type")
        self.coord1typelabel.setBuddy(self.coord1type)
        self.coord2type = QLineEdit('')
        self.coord2typelabel = QLabel("Coordinate 2 DIRFILE data type")
        self.coord2typelabel.setBuddy(self.coord2type)


        self.DirConvCheckBox = QCheckBox("DIRFILE Conversion factors")
        self.DirConvCheckBox.setChecked(False)

        self.adetconv = QLineEdit('')
        self.bdetconv = QLineEdit('')
        self.detconv = QLabel('Detector conversion factors')
        self.detconv.setBuddy(self.adetconv)

        self.acoord1conv = QLineEdit('')
        self.bcoord1conv = QLineEdit('')
        self.coord1conv = QLabel('Coordinate 1 conversion factors')
        self.coord1conv.setBuddy(self.acoord1conv)

        self.acoord2conv = QLineEdit('')
        self.bcoord2conv = QLineEdit('')
        self.coord2conv = QLabel('Coordinate 2 conversion factors')
        self.coord2conv.setBuddy(self.acoord2conv)

        
        self.DirConvCheckBox.toggled.connect(self.detconv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.adetconv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bdetconv.setVisible)

        self.DirConvCheckBox.toggled.connect(self.coord1conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.acoord1conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bcoord1conv.setVisible)

        self.DirConvCheckBox.toggled.connect(self.coord2conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.acoord2conv.setVisible)
        self.DirConvCheckBox.toggled.connect(self.bcoord2conv.setVisible)

        self.layout = QGridLayout()
        self.layout.addWidget(self.experimentLabel, 0, 0)
        self.layout.addWidget(self.experiment, 0, 1, 1, 3)
        self.layout.addWidget(self.detfreqlabel, 1, 0)
        self.layout.addWidget(self.detfreq, 1, 1, 1, 3)
        self.layout.addWidget(self.acsfreqlabel, 2, 0)
        self.layout.addWidget(self.acsfreq, 2, 1, 1, 3)
        self.layout.addWidget(self.highpassfreqlabel, 3, 0)
        self.layout.addWidget(self.highpassfreq, 3, 1, 1, 3)
        self.layout.addWidget(self.detframelabel, 4, 0)
        self.layout.addWidget(self.detframe, 4, 1, 1, 3)
        self.layout.addWidget(self.acsframelabel, 5, 0)
        self.layout.addWidget(self.acsframe, 5, 1, 1, 3)
        self.layout.addWidget(self.dettypelabel, 6, 0)
        self.layout.addWidget(self.dettype, 6, 1, 1, 3)
        self.layout.addWidget(self.coord1typelabel, 7, 0)
        self.layout.addWidget(self.coord1type, 7, 1, 1, 3)
        self.layout.addWidget(self.coord2typelabel, 8, 0)
        self.layout.addWidget(self.coord2type, 8, 1, 1, 3)
        self.layout.addWidget(self.numberframelabel, 9, 0)
        self.layout.addWidget(self.startframe, 9, 2, 1, 1)
        self.layout.addWidget(self.endframe, 9, 3, 1, 1)

        self.layout.addWidget(self.DirConvCheckBox, 10, 0)
        self.layout.addWidget(self.detconv, 11, 1)
        self.layout.addWidget(self.adetconv, 11, 2)
        self.layout.addWidget(self.bdetconv, 11, 3)
        self.layout.addWidget(self.coord1conv, 12, 1)
        self.layout.addWidget(self.acoord1conv, 12, 2)
        self.layout.addWidget(self.bcoord1conv, 12, 3)
        self.layout.addWidget(self.coord2conv, 13, 1)
        self.layout.addWidget(self.acoord2conv, 13, 2)
        self.layout.addWidget(self.bcoord2conv, 13, 3)
        self.detconv.setVisible(False)
        self.adetconv.setVisible(False)
        self.bdetconv.setVisible(False)
        self.coord1conv.setVisible(False)
        self.acoord1conv.setVisible(False)
        self.bcoord1conv.setVisible(False)
        self.coord2conv.setVisible(False)
        self.acoord2conv.setVisible(False)
        self.bcoord2conv.setVisible(False)
        self.ExperimentGroup.setContentsMargins(5, 5, 5, 5)

        self.ExperimentGroup.setLayout(self.layout)

    def createMapPlotGroup(self):
        self.MapPlotGroup = QGroupBox("Map")

        self.layout = QVBoxLayout()  
        self.matplotlibWidget_Map = MatplotlibWidget(self)
        self.axis_map = self.matplotlibWidget_Map.figure.add_subplot(111)
        self.axis_map.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_Map)
        self.plotbutton.clicked.connect(self.map2d)

        self.MapPlotGroup.setLayout(self.layout)
           
    def map2d(self, data=None):
        data = [random.random() for i in range(25)]

        self.ctype = self.coordchoice.text()

        self.crpix = np.array([self.crpix1.text(),self.crpix2.text()])
        self.cdelt = np.array([self.cdelt1.text(),self.cdelt2.text()])
        self.crval = np.array([self.crval1.text(),self.crval2.text()])

        if self.convchoice.currentText().lower() == 'gaussian':
            self.convolution = True
            self.std = self.GaussianSTD.text()
        else:
            self.convolution = False
            self.std = 0

        maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
                       self.cleaned_data, self.coord1slice, self.coord2slice, \
                       self.convolution, self.std)

        maps.wcs_proj()

        map_value = maps.map2d()
        
        self.axis_map.set_axis_on()
        self.axis_map.clear()
        self.axis_map.set_title('PyQt Matplotlib Example')
        
        max_index = np.where(np.abs(map_value) == np.amax((np.abs(map_value))))

        interval = np.flip(np.linspace(0, 1, levels+1))

        map_levels = map_value[max_index]*(1-interval)

        extent = (np.amin(coord1), np.amax(coord1), np.amin(coord2), np.amax(coord2))
        self.axis_map.imshow(map_value, extent = extent, origin='lower', cmap=plt.cm.viridis)
        self.cbar = plt.colorbar(im)

        if self.ctype.lower() == 'RA and DEC':
            self.axis_map.set_xlabel('RA (deg)')
            self.axis_map.set_ylabel('Dec (deg)')
        elif self.ctype.lower() == 'AZ and EL':
            self.axis_map.set_xlabel('Azimuth (deg)')
            self.axis_map.set_ylabel('Elevation (deg)')
        elif self.ctype.lower() == 'CROSS-EL and EL':
            self.axis_map.set_xlabel('Cross Elevation (deg)')
            self.axis_map.set_ylabel('Elevation (deg)')

        self.matplotlibWidget_Map.canvas.draw()

    def TODLayout(self):

        self.createTODplot()
        self.createTODcleanedplot()

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.TODplot, 0, 0)
        mainlayout.addWidget(self.TODcleanedplot, 0, 1)
        
        self.tab2.setLayout(mainlayout)

    def createTODplot(self, data = None):
        self.TODplot = QGroupBox("Detector TOD")
        self.layout = QVBoxLayout()

        self.matplotlibWidget_TOD = MatplotlibWidget(self)
        self.axis_TOD = self.matplotlibWidget_TOD.figure.add_subplot(111)
        self.axis_TOD.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_TOD)        
        self.plotbutton.clicked.connect(partial(self.draw_TOD, data))

        self.TODplot.setLayout(self.layout)

    def draw_TOD(self, data = None):
        self.axis_TOD.set_axis_on()
        self.axis_TOD.clear()
        try:
            self.axis_TOD.plot(self.detslice)
        except AttributeError:
            pass
        self.axis_TOD.set_title('detTOD')
        self.matplotlibWidget_TOD.canvas.draw()

    def createTODcleanedplot(self, data = None):
        self.TODcleanedplot = QGroupBox("Detector Cleaned TOD")
        self.layout = QVBoxLayout()

        self.matplotlibWidget_cleaned_TOD = MatplotlibWidget(self)
        self.axis_cleaned_TOD = self.matplotlibWidget_cleaned_TOD.figure.add_subplot(111)
        self.axis_cleaned_TOD.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_cleaned_TOD)
        self.plotbutton.clicked.connect(partial(self.draw_cleaned_TOD, data))

        self.TODcleanedplot.setLayout(self.layout)

    def draw_cleaned_TOD(self, data = None):
        
        self.axis_cleaned_TOD.set_axis_on()
        self.axis_cleaned_TOD.clear()
        try:
            det_tod = tod.data_cleaned(self.detslice, self.detfreq.text(), self.highpassfreq.text())
            self.cleaned_data = det_tod.data_clean()
            self.axis_cleaned_TOD.plot(self.cleaned_data)
        except AttributeError or NameError or TypeError:
            pass
        self.axis_cleaned_TOD.set_title('Cleaned Data')
        self.matplotlibWidget_cleaned_TOD.canvas.draw()

    def load_func(self):
        
        label_final = []

        coord_type = self.coordchoice.currentText()
        if coord_type == 'RA and DEC':
            self.coord1 = str('RA')
            self.coord2 = str('DEC')
        elif coord_type == 'AZ and EL':
            self.coord1 = str('AZ')
            self.coord2 = str('EL')
        elif coord_type == 'CROSS-EL and EL':
            self.coord1 = str('CROSS-EL')
            self.coord2 = str('EL')

        try:
            detfile = os.stat(self.detpath.text()+'/'+self.detname.text())
        except OSError:
            label = self.detpathlabel.text()
            label_final.append(label)
        if self.experiment.currentText().lower() == 'blast-tng':    
            try:
                coordfile = os.stat(self.coordpath.text())
            except OSError:
                label = self.coordpathlabel.text()
                label_final.append(label)
        elif self.experiment.currentText().lower() == 'blastpol':
            try:
                coordfile = os.stat(self.coordpath.text()+'/'+self.coord1.lower())
            except OSError:
                label = self.coord1.lower()+' coordinate'
                label_final.append(label)
            try:
                coordfile = os.stat(self.coordpath.text()+'/'+self.coord2.lower())
            except OSError:
                label = self.coord2.lower()+' coordinate'
                label_final.append(label)

        if np.size(label_final) != 0:
            self.warningbox = QMessageBox()
            self.warningbox.setIcon(QMessageBox.Warning)
            self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            self.warningbox.setWindowTitle('Warning')

            
            msg = 'Incorrect Path(s): \n'
            for i in range(len(label_final)): 
                msg += (str(label_final[i][:-1])) +'\n'
            
            self.warningbox.setText(msg)        
        
            self.warningbox.exec_()

        else:
            dataload = ld.data_value(self.detpath.text(), self.detname.text(), self.coordpath.text(), \
                                     self.coord1, self.coord2, self.dettype.text(), \
                                     self.coord1type.text(), self.coord2type.text(), \
                                     self.experiment.currentText())

            self.det_data, self.coord1_data, self.coord2_data = dataload.values()

            self.DirConvCheckBox.toggled.connect(self.dirfile_conversion)

            if self.experiment.currentText().lower() == 'blast-tng':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  roach_number = self.roachnumber.text(), \
                                                  roach_pps_path = self.coord_path.text())
            elif self.experiment.currentText().lower() == 'blastpol':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText())

            self.timemap, self.detslice, self.coord1slice, \
                                         self.coord2slice = zoomsyncdata.sync_data()

            print 'tests', self.detslice

    def dirfile_conversion(self):

        det_conv = ld.convert_dirfile(self.det_data, self.adetconv.text(), self.bdetconv.text())
        coord1_conv = ld.convert_dirfile(self.coord1_data, self.acoord1conv.text(), \
                                         self.bcoord1conv.text())
        coord2_conv = ld.convert_dirfile(self.coord2_data, self.acoord2conv.text(), \
                                         self.bcoord2conv.text())

        self.det_data = det_conv.conversion()
        self.coord1_data = coord1_data.conversion()
        self.coord2_data = coord2_data.conversion()


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layoutVertical = QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)
        self.layoutVertical.addWidget(self.toolbar)
