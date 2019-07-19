<<<<<<< HEAD
<<<<<<< HEAD
from PyQt5.QtCore import QDateTime, Qt, QTimer, pyqtSlot
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QScrollArea, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit,
        QVBoxLayout, QWidget, QMainWindow, QFormLayout, QMessageBox)

=======
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
from functools import partial
=======
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
from PyQt5.QtGui import *
>>>>>>> 8989c24... Correct calculation of coordinates

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
<<<<<<< HEAD
<<<<<<< HEAD

import numpy as np
import os
import configparser
=======
=======
>>>>>>> 11cb7f9... removed test button
from astropy.io import fits
from functools import partial

import numpy as np
import os
import pickle
import configparser
import gc
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 6acdf4e... Solved a memory leak when trying to replot with different parameters
=======
import copy
>>>>>>> 77760bc... Add TOD timing offset

import src.detTOD as tod
import src.loaddata as ld
import src.mapmaker as mp
import src.beam as bm
import src.pointing as pt 

<<<<<<< HEAD
<<<<<<< HEAD
class MainWindow(QTabWidget):
    
=======

class App(QMainWindow):

    '''
    Class to create the app
    '''

    def __init__(self):
        super().__init__()
        self.title = 'NaMap'

        self.setWindowTitle(self.title)
        
        self.TabLayout = MainWindowTab(self)
        self.setCentralWidget(self.TabLayout)

    def closeEvent(self,event):

        '''
        This function contains the code that is run when the application is closed.
        In this case, deleting the pickles file created.
        '''

        result = QMessageBox.question(self,
                                      "Confirm Exit...",
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes:
            
            directory = 'pickles_object/'
            pkl_list = os.listdir(directory)

            if np.size(pkl_list) > 0:
                for i in range(len(pkl_list)):
                    path = directory+pkl_list[i]
                    os.remove(path)

            event.accept()

class MainWindowTab(QTabWidget):

<<<<<<< HEAD
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
=======
    '''
    General layout of the application 
    '''

>>>>>>> 651e1e6... Commented files
    def __init__(self, parent = None):
        super(MainWindowTab, self).__init__(parent)
        self.tab1 = ParamMapTab()
        self.tab2 = TODTab()
        #self.tab3 = BeamTab()

        self.data = np.array([])
        self.cleandata = np.array([])

        self.tab1.plotbutton.clicked.connect(self.updatedata)
        self.tab1.fitsbutton.clicked.connect(self.save2fits)

        self.addTab(self.tab1,"Parameters and Maps")
        self.addTab(self.tab2,"Detector TOD")
        #self.addTab(self.tab3, "Beam")

    def updatedata(self):
        '''
        This function updates the map values everytime that the plot button is pushed
        '''

        #functions to compute the updated values
        self.tab1.load_func()

        self.data = self.tab1.detslice 

        self.cleandata = self.tab1.cleaned_data

        self.tab2.draw_TOD(self.data)
        self.tab2.draw_cleaned_TOD(self.cleandata)

        try:
            self.tab1.mapvalues(self.cleandata)

            #Update Maps
            maps = self.tab1.map_value
<<<<<<< HEAD
            wcs = self.tab1.proj
            print(wcs)
=======
>>>>>>> 6ecb5ac... Added FITS file generator and field
            mp_ini = self.tab1.createMapPlotGroup
            mp_ini.updateTab(data=maps)
            #Update Offset
            self.tab1.updateOffsetValue()

        except AttributeError:
            pass

    
    def save2fits(self): #function to save the map as a FITS file
        hdr = self.tab1.proj.to_header() #grabs the projection information for header
        maps = self.tab1.map_value #grabs the actual map for the fits img
        hdu = fits.PrimaryHDU(maps, header = hdr)
        hdu.writeto('./'+self.tab1.fitsname.text())

        
<<<<<<< HEAD
        self.show()

    def ParamMapLayout(self):
=======

class AppWindow(QMainWindow):

    '''
    Class to create the app
    '''

    def __init__(self):
        super().__init__()
        self.title = 'NaMap'

        self.setWindowTitle(self.title)

        self.TabLayout = MainWindowTab(self)
        self.setCentralWidget(self.TabLayout)

        self.current_name()
        self.TabLayout.tab1.experiment.activated[str].connect(self.current_name)

        menubar = self.menuBar()

        self.beammenu = menubar.addMenu('Beam Functions')
        beamfit = QAction('Fitting Parameters', self)
        self.beammenu.addAction(beamfit)
        beamfit.triggered.connect(self.beam_fit_param_menu)
        
        self.detmenu = menubar.addMenu('TOD Functions')
        todfunc = QAction('Time offset between TOD and Pointing',  self)
        self.detmenu.addAction(todfunc)
        todfunc.triggered.connect(self.tod_func_offset)

        self.pointingmenu = menubar.addMenu('Pointing Functions')
        lstlatfunc = QAction('LAT and LST paramters',  self)
        self.pointingmenu.addAction(lstlatfunc)
        lstlatfunc.triggered.connect(self.lstlat_func_offset)
        refpoint = QAction('Reference Point', self)
        self.pointingmenu.addAction(refpoint)
        refpoint.triggered.connect(self.refpoint_func)

    def current_name(self):

        self.experiment_name = self.TabLayout.experiment_name

    @pyqtSlot()
    def lstlat_func_offset(self):
        dialog = LST_LAT_Param(experiment = self.experiment_name)
        dialog.LSTtype_signal.connect(self.connection_LST_type)
        dialog.LATtype_signal.connect(self.connection_LAT_type)
        dialog.LSTconv_signal.connect(self.connection_LST_conv)
        dialog.LATconv_signal.connect(self.connection_LAT_conv)
        dialog.lstlatfreqsignal.connect(self.connection_LSTLAT_freq)
        dialog.lstlatsamplesignal.connect(self.connection_LSTLAT_sample)
        dialog.exec_()
    
    @pyqtSlot()
    def refpoint_func(self):
        dialog = REFPOINT_Param()
        dialog.radecsignal.connect(self.connection_ref_point)
        dialog.exec_()

    @pyqtSlot()
    def tod_func_offset(self):
        dialog = TODoffsetWindow()
        try:
            if self.TabLayout.todoffsetvalue is not None:
                dialog.todoffsetvalue.setText(str(self.TabLayout.todoffsetvalue))
            else:
                dialog.todoffsetvalue.setText('0.0')
        except AttributeError:
            pass
        dialog.todoffsetsignal.connect(self.connection_tod_off)
        dialog.exec_()

    @pyqtSlot()
    def beam_fit_param_menu(self):
        dialog = BeamFitParamWindow()
        dialog.fitparamsignal.connect(self.connection_beam_param)
        dialog.exec_()

    @pyqtSlot(np.ndarray)
    def connection_beam_param(self, val):
        self.TabLayout.beamparam = val.copy()
    
    @pyqtSlot(float)
    def connection_tod_off(self, val):
        self.TabLayout.todoffsetvalue = copy.copy(val)

    @pyqtSlot(str)
    def connection_LST_type(self, val):
        self.TabLayout.LSTtype = copy.copy(val)
    
    @pyqtSlot(str)
    def connection_LAT_type(self, val):
        self.TabLayout.LATtype = copy.copy(val)

    @pyqtSlot(np.ndarray)
    def connection_LST_conv(self, val):
        self.TabLayout.LSTconv = val.copy()

    @pyqtSlot(np.ndarray)
    def connection_LAT_conv(self, val):
        self.TabLayout.LATconv = val.copy()

    @pyqtSlot(float)
    def connection_LSTLAT_freq(self, val):
        self.TabLayout.lstlatfreq = copy.copy(val)

    @pyqtSlot(float)
    def connection_LSTLAT_sample(self, val):
        self.TabLayout.lstlatsampleframe = copy.copy(val)

    @pyqtSlot(np.ndarray)
    def connection_ref_point(self, val):
        self.TabLayout.refpoint = val.copy()    


    def closeEvent(self,event):

        '''
        This function contains the code that is run when the application is closed.
        In this case, deleting the pickles file created.
        '''

        result = QMessageBox.question(self,
                                      "Confirm Exit...",
                                      "Are you sure you want to exit ?",
                                      QMessageBox.Yes| QMessageBox.No)
        event.ignore()

        if result == QMessageBox.Yes:
            
            directory = 'pickles_object/'
            pkl_list = os.listdir(directory)

            if np.size(pkl_list) > 0:
                for i in range(len(pkl_list)):
                    path = directory+pkl_list[i]
                    os.remove(path)

            event.accept()

class LST_LAT_Param(QDialog):

    LSTtype_signal = pyqtSignal(str)
    LATtype_signal = pyqtSignal(str)
    LSTconv_signal = pyqtSignal(np.ndarray)
    LATconv_signal = pyqtSignal(np.ndarray)
    lstlatfreqsignal = pyqtSignal(float)
    lstlatsamplesignal = pyqtSignal(float)

    def __init__(self, experiment, parent = None):
        super(QDialog, self).__init__(parent)
        
        self.setWindowTitle('LAT and LST Parameters')

        self.LSTtype = QLineEdit('')
        self.LSTlabel = QLabel('LST File Type')

        self.LATtype = QLineEdit('')
        self.LATlabel = QLabel('LAT File Type')

        self.aLSTconv = QLineEdit('')
        self.bLSTconv = QLineEdit('')
        self.LSTconv = QLabel('LST DIRFILE conversion factors')
        self.LSTconv.setBuddy(self.aLSTconv)

        self.aLATconv = QLineEdit('')
        self.bLATconv = QLineEdit('')
        self.LATconv = QLabel('LAT DIRFILE conversion factors')
        self.LATconv.setBuddy(self.aLATconv)

        self.LSTLATfreq = QLineEdit('')
        self.LSTLATfreqlabel = QLabel('LST/LAT frequency Sample')

        self.LSTLATsample = QLineEdit('')
        self.LSTLATsamplelabel = QLabel('LST/LAT Samples per frame')

        self.savebutton = QPushButton('Write Parameters')
        self.savebutton.clicked.connect(self.updateParamValues)

        layout = QGridLayout(self)

        layout.addWidget(self.LSTlabel, 0, 0)
        layout.addWidget(self.LSTtype, 0, 1)
        layout.addWidget(self.LATlabel, 1, 0)
        layout.addWidget(self.LATtype, 1, 1)
        layout.addWidget(self.LSTconv, 2, 0)
        layout.addWidget(self.aLSTconv, 2, 1)
        layout.addWidget(self.bLSTconv, 2, 2)
        layout.addWidget(self.LATconv, 3, 0)
        layout.addWidget(self.aLATconv, 3, 1)
        layout.addWidget(self.bLATconv, 3, 2)
        layout.addWidget(self.LSTLATfreqlabel, 4, 0)
        layout.addWidget(self.LSTLATfreq, 4, 1)
        layout.addWidget(self.LSTLATsamplelabel, 5, 0)
        layout.addWidget(self.LSTLATsample, 5, 1)

        layout.addWidget(self.savebutton)

        self.configuration_value(experiment=experiment)

        self.setLayout(layout)

    def configuration_value(self, experiment):
    
        dir_path = os.getcwd()+'/config/'
        
        filepath = dir_path+experiment.lower()+'.cfg'
        model = configparser.ConfigParser()

        model.read(filepath)
        sections = model.sections()

        for section in sections:
            if section.lower() == 'lst_lat parameters':
                lstlatfreq_config = float(model.get(section, 'LSTLATFREQ').split('#')[0])
                lst_dir_conv = model.get(section,'LST_DIR_CONV').split('#')[0].strip()
                lstconv_config = np.array(lst_dir_conv.split(',')).astype(float)
                lat_dir_conv = model.get(section,'LAT_DIR_CONV').split('#')[0].strip()
                latconv_config = np.array(lat_dir_conv.split(',')).astype(float)
                lstlatframe_config = float(model.get(section, 'LSTLAT_SAMP_FRAME').split('#')[0])
                lsttype_config = model.get(section,'LST_FILE_TYPE').split('#')[0].strip()
                lattype_config = model.get(section,'LAT_FILE_TYPE').split('#')[0].strip()

        self.LSTLATfreq.setText(str(lstlatfreq_config))
        self.LSTLATsample.setText(str(lstlatframe_config))
        self.LSTtype.setText(str(lsttype_config))
        self.LATtype.setText(str(lattype_config))
        self.aLSTconv.setText(str(lstconv_config[0]))
        self.bLSTconv.setText(str(lstconv_config[1]))
        self.aLATconv.setText(str(latconv_config[0]))
        self.bLATconv.setText(str(latconv_config[1]))


    def updateParamValues(self):

        self.LSTtype_signal.emit(self.LSTtype.text())
        self.LATtype_signal.emit(self.LATtype.text())

        LST_array = np.array([float(self.aLSTconv.text()), \
                              float(self.bLSTconv.text())])

        LAT_array = np.array([float(self.aLATconv.text()), \
                              float(self.bLATconv.text())])

        self.LSTconv_signal.emit(LST_array)
        self.LATconv_signal.emit(LAT_array)
        self.lstlatfreqsignal.emit(float(self.LSTLATfreq.text()))
        self.lstlatsamplesignal.emit(float(self.LSTLATsample.text()))

        self.close()

class REFPOINT_Param(QDialog):

    '''
    Class to create a dialog input for the coordinates of the reference point.
    The reference point is used to compute the pointing offset
    '''

    radecsignal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(QDialog, self).__init__(parent)

        self.setWindowTitle('Reference Point for pointing offset')

        self.ra = QLineEdit('')
        self.ralabel = QLabel('RA in degree')
        self.dec = QLineEdit('')
        self.declabel = QLabel('DEC in degree')

        self.savebutton = QPushButton('Write Parameters')
        self.savebutton.clicked.connect(self.updateParamValues)

        self.layout = QGridLayout()

        self.layout.addWidget(self.ralabel, 0, 0)
        self.layout.addWidget(self.ra, 0, 1)
        self.layout.addWidget(self.declabel, 1, 0)
        self.layout.addWidget(self.dec, 1, 1)
        self.layout.addWidget(self.savebutton)

        self.setLayout(self.layout)

    def updateParamValues(self):

        self.ra_value = float(self.ra.text())
        self.dec_value = float(self.dec.text())

        self.radecsignal.emit(np.array([self.ra_value,self.dec_value]))
        self.close() 

class BeamFitParamWindow(QDialog):

    '''
    Dialog window for adding manually beam fitting parameters
    '''

    fitparamsignal = pyqtSignal(np.ndarray)

    def __init__(self, parent = None):
        super(QDialog, self).__init__(parent)
        
        self.setWindowTitle('Beam Fitting Parameters')

        self.peak_numbers = QLineEdit('')
        self.peak_numbers_label = QLabel('Number of Gaussin to be fitted')

        self.savebutton = QPushButton('Write Parameters')

        self.table = QTableWidget()

        self.peak_numbers.textChanged[str].connect(self.updateTable)
        self.savebutton.clicked.connect(self.updateParamValues)

        layout = QGridLayout(self)

        layout.addWidget(self.peak_numbers_label, 0, 0)
        layout.addWidget(self.peak_numbers, 0, 1)
        layout.addWidget(self.table, 1, 0, 1, 2)
        layout.addWidget(self.savebutton)

        self.setLayout(layout)

        self.resize(640, 480)
        
    def configureTable(self, table, rows):
        table.setColumnCount(6)
        table.setRowCount(rows)
        table.setHorizontalHeaderItem(0, QTableWidgetItem("Amplitude"))
        table.setHorizontalHeaderItem(1, QTableWidgetItem("X0 (in pixel)"))
        table.setHorizontalHeaderItem(2, QTableWidgetItem("Y0 (in pixel)"))
        table.setHorizontalHeaderItem(3, QTableWidgetItem("SigmaX (in pixel)"))
        table.setHorizontalHeaderItem(4, QTableWidgetItem("SigmaY (in pixel)"))
        table.setHorizontalHeaderItem(5, QTableWidgetItem("Theta"))
        #table.horizontalHeader().setStretchLastSection(True)

    def updateTable(self):
        try:
            rows = int(self.peak_numbers.text().strip())
            self.configureTable(self.table, rows)
        except ValueError:
            pass

    def updateParamValues(self):

        values = np.array([])
        rows_number = self.table.rowCount()
        column_number = self.table.columnCount()

        for i in range(rows_number):
            for j in range(column_number):
                values = np.append(values, float(self.table.item(i,j).text()))

        self.fitparam = values.copy()
        self.fitparamsignal.emit(values)

    # def closeEvent(self,event):

    #     '''
    #     This function contains the code that is run when the application is closed.
    #     In this case, deleting the pickles file created.
    #     '''

    #     result = QMessageBox.question(self,
    #                                   "Confirm Exit...",
    #                                   "Are you sure you want to exit ?",
    #                                   QMessageBox.Yes| QMessageBox.No)
    #     event.ignore()

    #     if result == QMessageBox.Yes:
        
    #         event.accept()

class TODoffsetWindow(QDialog):

    '''
    Dialog window for adding manually time offset between the 
    detector TOD and the pointing solution
    '''

    todoffsetsignal = pyqtSignal(float)

    def __init__(self, parent = None):
        super(QDialog, self).__init__(parent)
        #w = QDialog(self)

        self.setWindowTitle('Detectors TOD timing offset')

        self.todoffsetvalue = QLineEdit('')
        self.todoffset_label = QLabel('TOD timing offset value (ms)')

        self.savebutton = QPushButton('Write Parameters')

        layout = QGridLayout(self)

        layout.addWidget(self.todoffset_label, 0, 0)
        layout.addWidget(self.todoffsetvalue, 0, 1)
        layout.addWidget(self.savebutton)

        self.setLayout(layout)

        self.savebutton.clicked.connect(self.updateParamValues)

    def updateParamValues(self):

        self.value = float(self.todoffsetvalue.text())

        self.todoffsetsignal.emit(self.value)
        self.close()

class MainWindowTab(QTabWidget):

    '''
    General layout of the application 
    '''

    def __init__(self, parent = None):
        super(MainWindowTab, self).__init__(parent)
        self.tab1 = ParamMapTab()
        self.tab2 = TODTab()

        self.addTab(self.tab1,"Parameters and Maps")
        self.addTab(self.tab2,"Detector TOD")

        checkI = self.tab1.ICheckBox

        self.emit_name()
        self.tab1.experiment.activated[str].connect(self.emit_name)
        
        self.data = np.array([])
        self.cleandata = np.array([])

        self.beamparam = None
        self.todoffsetvalue = None

        #LAT and LST parameters
        self.LSTtype = None
        self.LATtype = None
        self.LSTconv = np.array([1., 0.])
        self.LATconv = np.array([1., 0.])
        self.lstlatfreq = None
        self.lstlatsampleframe = None

        self.refpoint = None

        self.tab3 = BeamTab(checkbox=checkI)
        self.addTab(self.tab3, "Beam")

        self.tab1.plotbutton.clicked.connect(self.updatedata)
        self.tab1.fitsbutton.clicked.connect(self.printvalue)

    def emit_name(self):

        self.experiment_name = (self.tab1.experiment.currentText())

    def updatedata(self):
        '''
        This function updates the map values everytime that the plot button is pushed
        '''

        if self.tab1.PointingOffsetCheckBox.isChecked():
            correction = True
        else:
            correction = False

        #functions to compute the updated values
        self.tab1.load_func(offset = self.todoffsetvalue, correction = correction, \
                            LSTtype=self.LSTtype, LATtype=self.LATtype,\
                            LSTconv=self.LSTconv, LATconv=self.LATconv, \
                            lstlatfreq=self.lstlatfreq, lstlatsample = self.lstlatsampleframe)

        self.data = self.tab1.detslice
        self.lst = self.tab1.lstslice
        self.lat = self.tab1.latslice

        self.cleandata = self.tab1.cleaned_data

        self.tab2.draw_TOD(self.data)
        self.tab2.draw_cleaned_TOD(self.cleandata)

        try:
            self.tab1.mapvalues(self.cleandata)

            #Update Maps
            maps = self.tab1.map_value
            mp_ini = self.tab1.createMapPlotGroup
            mp_ini.updateTab(data=maps)
            #Update Offset

            if self.tab1.PointingOffsetCalculationCheckBox.isChecked():
                if self.refpoint is not None:
                    self.tab1.updateOffsetValue(self.refpoint[0], self.refpoint[1])
                else:
                    self.tab1.updateOffsetValue()

            checkBeam = self.tab1.BeamCheckBox

            #Create Beams
            if checkBeam.isChecked():
                beam_value = bm.beam(maps, param = self.beamparam)
                beam_map = beam_value.beam_fit()

                beams = self.tab3.beammaps

                if isinstance(beam_map[0], str):
                    self.warningbox = QMessageBox()
                    self.warningbox.setIcon(QMessageBox.Warning)
                    self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                    self.warningbox.setWindowTitle('Warning')

                    msg = 'Fit not converged'
                    
                    self.warningbox.setText(msg)        
                
                    self.warningbox.exec_()

                else:
                    beams.updateTab(data=beam_map[0])

        except AttributeError:
            pass
    
    def save2fits(self): #function to save the map as a FITS file
        hdr = self.tab1.proj.to_header() #grabs the projection information for header
        maps = self.tab1.map_value #grabs the actual map for the fits img
        hdu = fits.PrimaryHDU(maps, header = hdr)
        hdu.writeto('./'+self.tab1.fitsname.text())

    def printvalue(self):
        print('OFFset', self.todoffsetvalue)
        print('REF Point', self.refpoint)
        print(self.LSTtype)
        print(self.LATtype)
        print(self.LSTconv)
        print(self.LATconv)
        print(self.lstlatfreq)
        print(self.lstlatsampleframe)
        print('offset Checkbox', self.tab1.PointingOffsetCheckBox.isChecked())

class ParamMapTab(QWidget):

    '''
    Create the layout of the first tab containing the various input parameters and 
    the final maps
    '''

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.detslice = np.array([])         #Detector TOD between the frames of interest
        self.latslice = np.array([])
        self.lstslice = np.array([])
        self.cleaned_data = np.array([])     #Detector TOD cleaned (despiked and highpassed) between the frame of interest
        self.proj = None                     #WCS projection of the map
        self.map_value = np.array([])        #Final map values
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
class ParamMapTab(QWidget):
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs

    '''
    Create the layout of the first tab containing the various input parameters and 
    the final maps
    '''

    def __init__(self, parent=None):
        super(QWidget, self).__init__(parent)

        self.detslice = np.array([])         #Detector TOD between the frames of interest
        self.cleaned_data = np.array([])     #Detector TOD cleaned (despiked and highpassed) between the frame of interest
        self.proj = np.array([])             #WCS projection of the map
        self.map_value = np.array([])        #Final map values
=======
        #self.ICheckBox=None
>>>>>>> 1531050... Added option to choose beam fitting parameters
=======
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation

        self.createAstroGroup()
        self.createExperimentGroup()
        self.createDataRepository()
<<<<<<< HEAD
<<<<<<< HEAD
=======
        
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
        self.plotbutton = QPushButton('Plot')
        self.button = QPushButton('Test')
        self.fitsbutton = QPushButton('Save as Fits')
        
        #self.plotbutton.clicked.connect(self.load_func)
        #self.plotbutton.clicked.connect(self.mapvalues)
        #self.plotbutton.clicked.connect(self.clean_func)

        self.createOffsetGroup()
<<<<<<< HEAD
=======
        
        self.plotbutton = QPushButton('Plot')
        self.fitsbutton = QPushButton('Save as Fits')

        self.createOffsetGroup()
        mainlayout = QGridLayout(self)
        self.createMapPlotGroup = (MapPlotsGroup(checkbox=self.ICheckBox, data=self.map_value, \
                                   projection = self.proj))

        self.fitsname = QLineEdit('')
        self.fitsnamelabel = QLabel("FITS name")
        self.fitsnamelabel.setBuddy(self.fitsname)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        mainlayout = QGridLayout(self)
<<<<<<< HEAD
        self.MapPlotGroup = MapPlotsGroup(checkbox=self.ICheckBox)
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
=======
        self.createMapPlotGroup = MapPlotsGroup(checkbox=self.ICheckBox, data=self.map_value)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

        self.fitsname = QLineEdit('')
        self.fitsnamelabel = QLabel("FITS name")
        self.fitsnamelabel.setBuddy(self.fitsname)

        scroll = QScrollArea()
        scroll.setWidget(self.ExperimentGroup)
        scroll.setWidgetResizable(True)
<<<<<<< HEAD
<<<<<<< HEAD
        scroll.setFixedHeight(100)
=======
        scroll.setFixedHeight(200)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        scroll.setFixedHeight(200)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

        ExperimentGroup_Scroll = QGroupBox("Experiment Parameters")
        ExperimentGroup_Scroll.setLayout(QVBoxLayout())
        ExperimentGroup_Scroll.layout().addWidget(scroll)
<<<<<<< HEAD
<<<<<<< HEAD
   
        mainlayout = QGridLayout()
=======
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
        mainlayout.addWidget(self.DataRepository, 0, 0)
        mainlayout.addWidget(self.AstroGroup, 1, 0)
        mainlayout.addWidget(ExperimentGroup_Scroll, 2, 0)
        mainlayout.addWidget(self.plotbutton, 3, 0)
        mainlayout.addWidget(self.createMapPlotGroup, 0, 1, 2, 1)
        mainlayout.addWidget(self.OffsetGroup, 2, 1)
        mainlayout.addWidget(self.fitsbutton,3,1)
        mainlayout.addWidget(self.fitsname)
        mainlayout.addWidget(self.fitsnamelabel)
        
        self.setLayout(mainlayout)        

    def createDataRepository(self):

        '''
        Function for the layout and input of the Data Repository group.
        This includes:
        - Paths to the data
        - Name of the detectors
        - Possibility to use pointing offset and detectortables
        - Roach Number
        '''

        self.DataRepository = QGroupBox("Data Repository")
        
        self.detpath = QLineEdit('/Users/ian/AnacondaProjects/BLASTpolData/bolo_data/')
        self.detpathlabel = QLabel("Detector Path:")
        self.detpathlabel.setBuddy(self.detpath)

<<<<<<< HEAD
        self.detname = QLineEdit('')
=======
        mainlayout.addWidget(self.DataRepository, 0, 0)
        mainlayout.addWidget(self.AstroGroup, 1, 0)
        mainlayout.addWidget(ExperimentGroup_Scroll, 2, 0)
        mainlayout.addWidget(self.plotbutton, 3, 0)
        mainlayout.addWidget(self.createMapPlotGroup, 0, 1, 2, 1)
        mainlayout.addWidget(self.OffsetGroup, 2, 1)
        mainlayout.addWidget(self.fitsbutton,3,1)
        mainlayout.addWidget(self.fitsname)
        mainlayout.addWidget(self.fitsnamelabel)
        
        self.setLayout(mainlayout)        

    def createDataRepository(self):

        '''
        Function for the layout and input of the Data Repository group.
        This includes:
        - Paths to the data
        - Name of the detectors
        - Possibility to use pointing offset and detectortables
        - Roach Number
        '''

        self.DataRepository = QGroupBox("Data Repository")
        
        self.detpath = QLineEdit('/mnt/c/Users/gabri/Documents/GitHub/mapmaking/2012_data/bolo_data/')
        self.detpathlabel = QLabel("Detector Path:")
        self.detpathlabel.setBuddy(self.detpath)

        self.detname = QLineEdit('n31c04')
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        self.detname = QLineEdit('n31c04')
>>>>>>> 6ecb5ac... Added FITS file generator and field
        self.detnamelabel = QLabel("Detector Name:")
        self.detnamelabel.setBuddy(self.detname)

        self.detvalue = np.array([])

        self.roachnumber = QLineEdit('')
        self.roachnumberlabel = QLabel("Roach Number:")
        self.roachnumberlabel.setBuddy(self.roachnumber)

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        self.coordpath = QLineEdit('')
=======
        self.coordpath = QLineEdit('/Users/ian/AnacondaProjects/BLASTpolData/')
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        self.coordpath = QLineEdit('/Users/ian/AnacondaProjects/BLASTpolData/')
>>>>>>> 6ecb5ac... Added FITS file generator and field
=======
        self.coordpath = QLineEdit('/mnt/c/Users/gabri/Documents/GitHub/mapmaking/2012_data/')
>>>>>>> cbc2d94... Solved some bugs in computing offset
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
        self.pointingoffsetnumber = QLineEdit('')
        self.pointingoffsetnumberlabel = QLabel("StarCamera used for pointing offset:")
        self.pointingoffsetnumberlabel.setBuddy(self.pointingoffsetnumber)

        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetnumberlabel.setVisible)
        self.PointingOffsetCheckBox.toggled.connect(self.pointingoffsetnumber.setVisible)

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
        self.layout.addWidget(self.pointingoffsetnumberlabel, 7, 1)
        self.layout.addWidget(self.pointingoffsetnumber, 7, 2)

        self.dettablepathlabel.setVisible(False)
        self.dettablepath.setVisible(False)
        self.pointingoffsetnumberlabel.setVisible(False)
        self.pointingoffsetnumber.setVisible(False)

        self.DataRepository.setLayout(self.layout)

    def createAstroGroup(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files

        '''
        Function for the layout and input of the Astronometry parameters group.
        This includes:
        - Coordinates system
        - Standard WCS parameters to create a map
        - If the maps need to be convolved
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
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

        self.crpix1 = QLineEdit('50')
        self.crpix2 = QLineEdit('50')
        self.crpixlabel = QLabel("CRpix of the Map:")
        self.crpixlabel.setBuddy(self.crpix1)

        self.cdelt1 = QLineEdit('0.00189')
        self.cdelt2 = QLineEdit('0.00189')
        self.cdeltlabel = QLabel("Cdelt of the Map in deg:")
        self.cdeltlabel.setBuddy(self.cdelt1)

        self.crval1 = QLineEdit('132.20')
        self.crval2 = QLineEdit('-42.54')
        self.crvallabel = QLabel("Cval of the Map in deg:")
        self.crvallabel.setBuddy(self.crval1)

        self.pixnum1 = QLineEdit('100')
        self.pixnum2 = QLineEdit('100')
        self.pixnumlabel = QLabel("Pixel Number:")
        self.pixnumlabel.setBuddy(self.pixnum1)

        self.ICheckBox = QCheckBox("Map only I")
        self.ICheckBox.setChecked(True)

        self.BeamCheckBox = QCheckBox("Beam Analysis")
        self.BeamCheckBox.setChecked(False)

        self.PointingOffsetCalculationCheckBox = QCheckBox("Calculate Pointing Offset")
        self.PointingOffsetCalculationCheckBox.setChecked(False)
        
        self.convchoice.activated[str].connect(self.updateGaussian)
<<<<<<< HEAD
<<<<<<< HEAD
=======
        self.coordchoice.activated[str].connect(self.configuration_update)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        self.coordchoice.activated[str].connect(self.configuration_update)
>>>>>>> 4ee3dbe... Fixed bug in selecting data
           
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
        self.layout.addWidget(self.crvallabel, 5, 0)
        self.layout.addWidget(self.crval1, 5, 1)
        self.layout.addWidget(self.crval2, 5, 2)
        self.layout.addWidget(self.pixnumlabel, 6, 0)
        self.layout.addWidget(self.pixnum1, 6, 1)
        self.layout.addWidget(self.pixnum2, 6, 2)
        self.layout.addWidget(self.ICheckBox, 7, 0)
        self.layout.addWidget(self.BeamCheckBox, 8, 0)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
=======
        self.layout.addWidget(self.PointingOffsetCalculationCheckBox, 9, 0)
>>>>>>> 63d0b03... Added pointing offset calculation

>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
        self.GaussianSTD.setVisible(False)
        self.gaussianLabel.setVisible(False)
        
        self.AstroGroup.setLayout(self.layout)

    def updateGaussian(self, text=None):
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files

        '''
        Function to update the layout of the group, to add a line 
        to input the std of the gaussian convolution if the convolution parameter
        is set to gaussian
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
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
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files

        '''
        Function for the layout and input of the Experiment parameters group.
        This includes:
        - Frequency sampling of detectors and ACSs 
        - Experiment to be analyzed 
        - Frames of interests
        - High pass filter cutoff frequency
        - If DIRFILE conversion needs to be performed. If so, the parameters to 
          use for the conversion
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
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

        self.highpassfreq = QLineEdit('0.1')
        self.highpassfreqlabel = QLabel("High Pass Filter cutoff frequency")
        self.highpassfreqlabel.setBuddy(self.highpassfreq)

        self.detframe = QLineEdit('')
        self.detframelabel = QLabel("Detector Samples per Frame")        
        self.detframelabel.setBuddy(self.detframe)
        self.acsframe = QLineEdit('')
        self.acsframelabel = QLabel("ACS Sample Samples per Frame")
        self.acsframelabel.setBuddy(self.acsframe)

<<<<<<< HEAD
<<<<<<< HEAD
        self.startframe = QLineEdit('')
        self.endframe = QLineEdit('')
=======
        self.startframe = QLineEdit('1918381')
        self.endframe = QLineEdit('1922092')
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        self.startframe = QLineEdit('1918381')
        self.endframe = QLineEdit('1922092')
>>>>>>> 6ecb5ac... Added FITS file generator and field
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
        self.DirConvCheckBox.setChecked(True)

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

<<<<<<< HEAD
<<<<<<< HEAD
        self.configuration_value()
        self.experiment.activated[str].connect(self.configuration_value)
=======
        self.configuration_update()
        self.experiment.activated[str].connect(self.configuration_update)
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        self.configuration_update()
        self.experiment.activated[str].connect(self.configuration_update)
>>>>>>> 4ee3dbe... Fixed bug in selecting data

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
        self.detconv.setVisible(True)
        self.adetconv.setVisible(True)
        self.bdetconv.setVisible(True)
        self.coord1conv.setVisible(True)
        self.acoord1conv.setVisible(True)
        self.bcoord1conv.setVisible(True)
        self.coord2conv.setVisible(True)
        self.acoord2conv.setVisible(True)
        self.bcoord2conv.setVisible(True)

        self.ExperimentGroup.setContentsMargins(5, 5, 5, 5)

        self.ExperimentGroup.setLayout(self.layout)

<<<<<<< HEAD
<<<<<<< HEAD
    def configuration_value(self):
        text = self.experiment.currentText()
=======
    def configuration_update(self):

        '''
        Function to update the experiment parameters based on some templates.
        It requires the coordinates system and the experiment name
        '''

        text = self.experiment.currentText()
=======
    def configuration_update(self):

        '''
        Function to update the experiment parameters based on some templates.
        It requires the coordinates system and the experiment name
        '''

        text = self.experiment.currentText()
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        coord_text = self.coordchoice.currentText()

        self.configuration_value(text, coord_text)

    def configuration_value(self, text, coord_text):
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> 651e1e6... Commented files

        '''
        Function to read the experiment parameters from the template
        '''
        
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        dir_path = os.getcwd()+'/config/'
        
        filepath = dir_path+text.lower()+'.cfg'
        model = configparser.ConfigParser()

        model.read(filepath)
        sections = model.sections()

        for section in sections:
            if section.lower() == 'experiment parameters':
                self.detfreq_config = float(model.get(section, 'DETFREQ').split('#')[0])
<<<<<<< HEAD
<<<<<<< HEAD
                self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
                det_dir_conv = model.get(section,'DET_DIR_CONV').split('#')[0].strip()
                self.detconv_config = np.array(det_dir_conv.split(',')).astype(float)
                self.detframe_config = float(model.get(section, 'DET_SAMP_FRAME').split('#')[0])
                self.dettype_config = model.get(section,'DET_FILE_TYPE').split('#')[0].strip()

<<<<<<< HEAD
=======
                det_dir_conv = model.get(section,'DET_DIR_CONV').split('#')[0].strip()
                self.detconv_config = np.array(det_dir_conv.split(',')).astype(float)
                self.detframe_config = float(model.get(section, 'DET_SAMP_FRAME').split('#')[0])
                self.dettype_config = model.get(section,'DET_FILE_TYPE').split('#')[0].strip()

=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
            elif section.lower() == 'ra_dec parameters':
                if coord_text.lower() == 'ra and dec':
                    self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
                    coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
                    self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
                    coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
                    self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
                    self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
                    self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
                    self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()
                else:
                    pass

            elif section.lower() == 'az_el parameters':
                if coord_text.lower() == 'ra and dec':
                    pass
                else:
                    self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
                    coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
                    self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
                    coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
                    self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
                    self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
                    self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
                    self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()
            
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
        self.detfreq.setText(str(self.detfreq_config))
        self.acsfreq.setText(str(self.acsfreq_config))
        self.detframe.setText(str(self.detframe_config))
        self.acsframe.setText(str(self.acsframe_config))
        self.dettype.setText(str(self.dettype_config))
        self.coord1type.setText(str(self.coord1type_config))
        self.coord2type.setText(str(self.coord2type_config))
        self.adetconv.setText(str(self.detconv_config[0]))
        self.bdetconv.setText(str(self.detconv_config[1]))
        self.acoord1conv.setText(str(self.coord1conv_config[0]))
        self.bcoord1conv.setText(str(self.coord1conv_config[1]))
        self.acoord2conv.setText(str(self.coord2conv_config[0]))
        self.bcoord2conv.setText(str(self.coord2conv_config[1]))
<<<<<<< HEAD
<<<<<<< HEAD

    def createMapPlotGroup(self):
        self.MapPlotGroup = QTabWidget()

        self.mapTab1 = QWidget()
        self.mapTab2 = QWidget()
        self.mapTab3 = QWidget()

        self.layout = QVBoxLayout()  
        self.matplotlibWidget_Map = MatplotlibWidget(self)
        self.axis_map = self.matplotlibWidget_Map.figure.add_subplot(111)
        self.axis_map.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_Map)
        self.plotbutton.clicked.connect(self.map2d)

        self.MapPlotGroup.setLayout(self.layout)
           
=======
          
<<<<<<< HEAD
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
    def map2d(self, data=None):
        data = [random.random() for i in range(25)]

        self.ctype = self.coordchoice.currentText()

        self.crpix = np.array([int(float(self.crpix1.text())),\
                               int(float(self.crpix2.text()))])
        self.cdelt = np.array([float(self.cdelt1.text()),\
                               float(self.cdelt2.text())])
        self.crval = np.array([float(self.crval1.text()),\
                               float(self.crval2.text())])

        if self.convchoice.currentText().lower() == 'gaussian':
            self.convolution = True
            self.std = self.GaussianSTD.text()
        else:
            self.convolution = False
            self.std = 0

        self.maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
                       self.cleaned_data, self.coord1slice, self.coord2slice, \
                       self.convolution, self.std)

        self.maps.wcs_proj()

        self.map_value = self.maps.map2d()
        
        self.axis_map.set_axis_on()
        self.axis_map.clear()
        self.axis_map.set_title('Maps')
        
        max_index = np.where(np.abs(self.map_value) == np.amax((np.abs(self.map_value))))

        levels = 5

        interval = np.flip(np.linspace(0, 1, levels+1))

        map_levels = self.map_value[max_index]*(1-interval)

        extent = (np.amin(self.coord1slice), np.amax(self.coord1slice), \
                  np.amin(self.coord2slice), np.amax(self.coord2slice))
        im = self.axis_map.imshow(self.map_value, extent = extent, origin='lower', cmap=plt.cm.viridis)
        plt.colorbar(im)

        if self.ctype == 'RA and DEC':
            self.axis_map.set_xlabel('RA (deg)')
            self.axis_map.set_ylabel('Dec (deg)')
        elif self.ctype == 'AZ and EL':
            self.axis_map.set_xlabel('Azimuth (deg)')
            self.axis_map.set_ylabel('Elevation (deg)')
        elif self.ctype == 'CROSS-EL and EL':
            self.axis_map.set_xlabel('Cross Elevation (deg)')
            self.axis_map.set_ylabel('Elevation (deg)')

        self.matplotlibWidget_Map.canvas.draw()

=======
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
    def createOffsetGroup(self):
=======
          
    def createOffsetGroup(self):

        '''
        Function to create the layout and the output of the offset group.
        Check the pointing.py for offset calculation
        '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files

        self.OffsetGroup = QGroupBox("Detector Offset")

        self.CROSSELoffsetlabel = QLabel('Cross Elevation (deg)')
        self.ELxoffsetlabel = QLabel('Elevation (deg)')
        self.CROSSELoffset = QLineEdit('')
        self.ELxoffset = QLineEdit('')

<<<<<<< HEAD
        self.coordchoice.activated[str].connect(self.updateOffsetLabel)
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
        self.plotbutton.clicked.connect(self.updateOffsetValue)
=======
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
        #self.plotbutton.clicked.connect(self.updateOffsetValue)
>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)
=======
>>>>>>> 4ee3dbe... Fixed bug in selecting data
=======
        #self.coordchoice.activated[str].connect(self.updateOffsetLabel)

        #self.updateOffsetLabel()
>>>>>>> 63d0b03... Added pointing offset calculation

        self.PointingOffsetCalculationCheckBox.toggled.connect(self.updateOffsetLabel)
        self.updateOffsetLabel()

        self.layout = QGridLayout()
        self.layout.addWidget(self.CROSSELoffsetlabel, 0, 0)
        self.layout.addWidget(self.CROSSELoffset, 0, 1)
        self.layout.addWidget(self.ELxoffsetlabel, 1, 0)
        self.layout.addWidget(self.ELxoffset, 1, 1)

        self.ELxoffset.setEnabled(False)
        self.CROSSELoffset.setEnabled(False)
        
        self.OffsetGroup.setLayout(self.layout)

    def updateOffsetLabel(self):

<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files
        '''
        Update the offset labels based on the coordinate system choice
        '''
        if self.PointingOffsetCalculationCheckBox.isChecked():
            self.OffsetGroup.setVisible(True)
        else:
            self.OffsetGroup.setVisible(False)

<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
        self.ctype = self.coordchoice.currentText()

        if self.ctype == 'RA and DEC':
            self.RADECvisibile = True
            self.AZELvisibile = False
            self.CROSSEL_ELvisibile = False

        elif self.ctype == 'AZ and EL':
            self.RADECvisibile = False
            self.AZELvisibile = True
            self.CROSSEL_ELvisibile = False
            
        elif self.ctype == 'CROSS-EL and EL':
            self.RADECvisibile = False
            self.AZELvisibile = False
            self.CROSSEL_ELvisibile = True


        self.RAoffset.setVisible(self.RADECvisibile)
        self.DECoffset.setVisible(self.RADECvisibile)
        self.RAoffsetlabel.setVisible(self.RADECvisibile)
        self.DECoffsetlabel.setVisible(self.RADECvisibile)

        self.AZoffset.setVisible(self.AZELvisibile)
        self.ELoffset.setVisible(self.AZELvisibile)
        self.AZoffsetlabel.setVisible(self.AZELvisibile)
        self.ELoffsetlabel.setVisible(self.AZELvisibile)      

        self.CROSSELoffset.setVisible(self.CROSSEL_ELvisibile)
        self.ELxoffset.setVisible(self.CROSSEL_ELvisibile)
        self.CROSSELoffsetlabel.setVisible(self.CROSSEL_ELvisibile)
        self.ELxoffsetlabel.setVisible(self.CROSSEL_ELvisibile) 

    def updateOffsetValue(self):
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> 651e1e6... Commented files
=======
    def updateOffsetValue(self, coord1_ref=None, coord2_ref=None):
>>>>>>> 63d0b03... Added pointing offset calculation

        '''
        Calculate and update the offset value based on the coordinate system choice
        '''
<<<<<<< HEAD
<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
        
        ctype_map = self.coordchoice.currentText()
=======
>>>>>>> 63d0b03... Added pointing offset calculation

        if self.ctype.lower() == 'ra and dec':
            coord1_ref = copy.copy(float(self.crval1.text()))
            coord2_ref = copy.copy(float(self.crval2.text()))
        else:
            coord1_ref = coord1_ref
            coord2_ref = coord2_ref

        print('REF',coord1_ref, coord2_ref)

        offset = pt.compute_offset(coord1_ref, coord2_ref, self.map_value, self.w[:,0], self.w[:,1],\
                                   self.proj, self.ctype, self.lstslice, self.latslice)

<<<<<<< HEAD
        if self.ctype == 'RA and DEC':
<<<<<<< HEAD
<<<<<<< HEAD
            self.RAoffset = QLineEdit(str(self.offset_angle[0]))
            self.DECoffset = QLineEdit(str(self.offset_angle[1]))
=======
            self.RAoffset.setText(str(self.offset_angle[0]))
            self.DECoffset.setText(str(self.offset_angle[1]))
>>>>>>> ccf3a8f... Added output for offset calculation
=======
        
        xel_offset, el_offset = offset.value()
<<<<<<< HEAD
>>>>>>> 63d0b03... Added pointing offset calculation
=======
        print('OFFSET', xel_offset, el_offset)
>>>>>>> cbc2d94... Solved some bugs in computing offset

        self.CROSSELoffset.setText(str(xel_offset))
        self.ELxoffset.setText(str(el_offset))

    def load_func(self, offset = None, correction=False, LSTtype=None, LATtype=None,\
                  LSTconv=None, LATconv=None, lstlatfreq=None, lstlatsample = None):


        '''
        Wrapper function to loaddata.py to read the DIRFILEs.
        If the paths are not correct a warning is generated. To reduce the time to 
        re-run the code everytime, a new DIRFILE is loaded a pickle object is created so it can be 
        loaded again when the plot button is pushed. The pickles object are deleted when
        the software is closed
        '''
        
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
            os.stat(self.detpath.text()+'/'+self.detname.text())
        except OSError:
            label = self.detpathlabel.text()[:-1]
            label_final.append(label)
        if self.experiment.currentText().lower() == 'blast-tng':    
            try:
                os.stat(self.coordpath.text())
            except OSError:
                label = self.coordpathlabel.text()
                label_final.append(label)
        elif self.experiment.currentText().lower() == 'blastpol':
            try:
                if self.coord1 == 'RA':
                    os.stat(self.coordpath.text()+'/'+self.coord1.lower())
                else:
                    os.stat(self.coordpath.text()+'/az')
            except OSError:
                label = self.coord1.lower()+' coordinate'
                label_final.append(label)
            try:
                os.stat(self.coordpath.text()+'/'+self.coord2.lower())
            except OSError:
                label = self.coord2.lower()+' coordinate'
                label_final.append(label)

        label_lst = []
        if correction:
            try:
                os.stat(os.getcwd()+'/xsc_'+self.pointingoffsetnumber.text()+'.txt')
            except OSError:
                label = 'StarCamera'
                label_final.append(label)
        
        if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
            print('TEST')
            try:
                os.stat(self.coordpath.text()+'/'+'lst')
            except OSError:
                label = 'LST'
                label_final.append(label)
            try:
                os.stat(self.coordpath.text()+'/'+'lat')
            except OSError:
                label = 'LAT'
                label_final.append(label)

            if LSTtype is None:
                print('OK_TEST')
                label_lst = 'Write LST and LAT Parameters from the menubar'

        if np.size(label_final)+np.size(label_lst) != 0:
            
            if np.size(label_final) != 0:
                self.warningbox = QMessageBox()
                self.warningbox.setIcon(QMessageBox.Warning)
                self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                self.warningbox.setWindowTitle('Warning')

                msg = 'Incorrect Path(s): \n'
                for i in range(len(label_final)): 
                    msg += (str(label_final[i])) +'\n'
                
                self.warningbox.setText(msg)        
            
                self.warningbox.exec_()
            
            if np.size(label_lst) !=0:
                self.lstwarningbox = QMessageBox()
                self.lstwarningbox.setIcon(QMessageBox.Warning)
                self.lstwarningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

                self.lstwarningbox.setWindowTitle('Warning')
                
                self.lstwarningbox.setText(label_lst)        
            
                self.lstwarningbox.exec_()


        else:
            
            os.makedirs(os.path.dirname('pickles_object/'), exist_ok=True)

            det_path_pickle = 'pickles_object/'+self.detname.text()
            coord1_path_pickle = 'pickles_object/'+self.coord1
            coord2_path_pickle = 'pickles_object/'+self.coord2
            lat_path_pickle = 'pickles_object/lat'
            lst_path_pickle = 'pickles_object/lst'

            try:               
                self.det_data = pickle.load(open(det_path_pickle, 'rb'))  
                self.coord1_data = pickle.load(open(coord1_path_pickle, 'rb'))
                self.coord2_data = pickle.load(open(coord2_path_pickle, 'rb'))
                
                if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
                    self.lst_data = pickle.load(open(lst_path_pickle, 'rb'))
                    self.lat_data = pickle.load(open(lat_path_pickle, 'rb'))
                else:
                    self.lst_data = None
                    self.lat_data = None

            except FileNotFoundError:

                if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
                    lat_file_type = LSTtype
                    lst_file_type = LATtype
                    print('OK', lat_file_type)
                else: 
                    lat_file_type = None
                    lst_file_type = None


<<<<<<< HEAD
            except FileNotFoundError:               
=======
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation
                dataload = ld.data_value(self.detpath.text(), self.detname.text(), self.coordpath.text(), \
                                         self.coord1, self.coord2, self.dettype.text(), \
                                         self.coord1type.text(), self.coord2type.text(), \
                                         self.experiment.currentText(), lst_file_type, lat_file_type)

                if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
                    self.det_data, self.coord1_data, self.coord2_data, self.lst_data, self.lat_data = dataload.values()
                    pickle.dump(self.lst_data, open(lst_path_pickle, 'wb'))
                    pickle.dump(self.lat_data, open(lat_path_pickle, 'wb'))
                else:
                    self.det_data, self.coord1_data, self.coord2_data = dataload.values()
                    self.lst_data = None
                    self.lat_data = None

                pickle.dump(self.det_data, open(det_path_pickle, 'wb'))  
                pickle.dump(self.coord1_data, open(coord1_path_pickle, 'wb'))
                pickle.dump(self.coord2_data, open(coord2_path_pickle, 'wb'))
                

                del dataload
                gc.collect()
 
            if self.experiment.currentText().lower() == 'blast-tng':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  self.lst_data, self.lat_data, lstlatfreq, \
                                                  lstlatsample, offset=offset, \
                                                  roach_number = self.roachnumber.text(), \
                                                  roach_pps_path = self.coord_path.text())
            elif self.experiment.currentText().lower() == 'blastpol':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
<<<<<<< HEAD
                                                  self.endframe.text(), self.experiment.currentText(), offset)

            (self.timemap, self.detslice, self.coord1slice, \
             self.coord2slice) = zoomsyncdata.sync_data()
=======
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  self.lst_data, self.lat_data, lstlatfreq, \
                                                  lstlatsample, offset)

            if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
                (self.timemap, self.detslice, self.coord1slice, \
                 self.coord2slice, self.lstslice, self.latslice) = zoomsyncdata.sync_data()
            else:
                (self.timemap, self.detslice, self.coord1slice, \
                 self.coord2slice) = zoomsyncdata.sync_data()
<<<<<<< HEAD
>>>>>>> 3f224e8... Added pointing input dialogs and caluclation
=======
                self.lstslice = None
                self.latslice = None

<<<<<<< HEAD
            print('BEFORE', self.coord2slice)
>>>>>>> cbc2d94... Solved some bugs in computing offset

=======
>>>>>>> dfc2e45... Correct pointing offset calculation
            if self.DirConvCheckBox.isChecked:
                self.dirfile_conversion(correction = correction, LSTconv = LSTconv, \
                                        LATconv = LATconv)
            
            print('AFTER', self.coord1slice, self.coord2slice)
            
            if self.coord1.lower() == 'cross-el':
                self.coord1slice = self.coord1slice*np.cos(np.radians(self.coord2slice))
            # else:
            #     self.coord1slice = self.coord1slice
            
            if correction is True:

                xsc_file = ld.xsc_offset(self.pointingoffsetnumber.text(), self.startframe.text(), self.endframe.text())

                xsc_offset = xsc_file.read_file()   

                corr = pt.apply_offset(self.coord1slice, self.coord2slice, coord_type,\
                                       xsc_offset, lst = self.lstslice, lat = self.latslice)

                self.coord1slice, self.coord2slice = corr.correction()
            else:
                if self.coord1.lower() == 'ra':
                    self.coord1slice = self.coord1slice*15. #Conversion between hours to degree
           
            
            del self.det_data
            del self.coord1_data
            del self.coord2_data 
            del zoomsyncdata
            gc.collect()

            self.clean_func()

    def clean_func(self):

        '''
        Function to compute the cleaned detector TOD
        '''

        det_tod = tod.data_cleaned(self.detslice, self.detfreq.text(), self.highpassfreq.text())
        self.cleaned_data = det_tod.data_clean()

    def dirfile_conversion(self, correction=False, LSTconv=None, LATconv=None):

        '''
        Function to convert the DIRFILE data.
        '''

        det_conv = ld.convert_dirfile(self.detslice, float(self.adetconv.text()), \
                                      float(self.bdetconv.text()))
        coord1_conv = ld.convert_dirfile(self.coord1slice, float(self.acoord1conv.text()), \
                                         float(self.bcoord1conv.text()))
        coord2_conv = ld.convert_dirfile(self.coord2slice, float(self.acoord2conv.text()), \
                                         float(self.bcoord2conv.text()))

        det_conv.conversion()
        coord1_conv.conversion()
        coord2_conv.conversion()

        self.detslice = det_conv.data
        self.coord1slice = coord1_conv.data
        self.coord2slice = coord2_conv.data

        if (correction and self.coord1.lower() == 'ra') or self.PointingOffsetCalculationCheckBox.isChecked():
            print('TEST_CORRECTION')
            lst_conv = ld.convert_dirfile(self.lstslice, float(LSTconv[0]), \
                                          float(LSTconv[1]))
            lat_conv = ld.convert_dirfile(self.latslice, float(LATconv[0]), \
                                          float(LATconv[1]))

            lst_conv.conversion()
            lat_conv.conversion()

            self.lstslice = lst_conv.data
            self.latslice = lat_conv.data

            
            
    def mapvalues(self, data):

        '''
        Function to compute the maps
        '''

        self.ctype = self.coordchoice.currentText()

        self.crpix = np.array([int(float(self.crpix1.text())),\
                               int(float(self.crpix2.text()))])
        self.cdelt = np.array([float(self.cdelt1.text()),\
                               float(self.cdelt2.text())])
        self.crval = np.array([float(self.crval1.text()),\
                               float(self.crval2.text())])

        if self.convchoice.currentText().lower() == 'gaussian':
            self.convolution = True
            self.std = self.GaussianSTD.text()
        else:
            self.convolution = False
            self.std = 0

        self.maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
                            data, self.coord1slice, self.coord2slice, \
                            self.convolution, self.std, self.ICheckBox.isChecked())

        self.maps.wcs_proj()

        self.proj = self.maps.proj
        self.w = self.maps.w

        self.map_value = self.maps.map2d()

class TODTab(QWidget):

    '''
    Layout Class for the TOD tab
    '''

    def __init__(self, parent=None):

        super(QWidget, self).__init__(parent)

        self.c = ParamMapTab()

        self.createTODplot()
        self.createTODcleanedplot()

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.TODplot, 0, 0)
        mainlayout.addWidget(self.TODcleanedplot, 0, 1)
        
        self.setLayout(mainlayout)

    def createTODplot(self, data = None):

        '''
        Function to create the TOD empty plot
        '''

        self.TODplot = QGroupBox("Detector TOD")
        TODlayout = QGridLayout()

        self.matplotlibWidget_TOD = MatplotlibWidget(self)
        self.axis_TOD = self.matplotlibWidget_TOD.figure.add_subplot(111)
        self.axis_TOD.set_axis_off()
        TODlayout.addWidget(self.matplotlibWidget_TOD)        

        self.TODplot.setLayout(TODlayout)

    def draw_TOD(self, data = None):

        '''
        Function to draw the TOD when the plot button is pushed.
        The plotted TOD is the one between the frame of interest
        '''
        
        self.axis_TOD.set_axis_on()
        self.axis_TOD.clear()
        try:
            self.axis_TOD.plot(data)
        except AttributeError:
            pass
        self.axis_TOD.set_title('detTOD')
        self.matplotlibWidget_TOD.canvas.draw()

    def createTODcleanedplot(self, data = None):

        '''
        Same of createTODPlot but for the cleanedTOD
        '''

        self.TODcleanedplot = QGroupBox("Detector Cleaned TOD")
        self.layout = QVBoxLayout()

        self.matplotlibWidget_cleaned_TOD = MatplotlibWidget(self)
        self.axis_cleaned_TOD = self.matplotlibWidget_cleaned_TOD.figure.add_subplot(111)
        self.axis_cleaned_TOD.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_cleaned_TOD)

        self.TODcleanedplot.setLayout(self.layout)

    def draw_cleaned_TOD(self, data = None):

        '''
        Same of draw_TOD but for the cleaned TOD
        '''
        
        self.axis_cleaned_TOD.set_axis_on()
        self.axis_cleaned_TOD.clear()
        try:           
            self.axis_cleaned_TOD.plot(data)
        except AttributeError or NameError or TypeError:
            pass
        self.axis_cleaned_TOD.set_title('Cleaned Data')
        self.matplotlibWidget_cleaned_TOD.canvas.draw()

class BeamTab(ParamMapTab):

    '''
    Layout for the tab used to show the calculated beams
    '''


    def __init__(self, parent=None, checkbox=None):

        super(QWidget, self).__init__(parent)

        self.beammaps = MapPlotsGroup(checkbox=checkbox, data=None)

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.beammaps, 0, 0)
        
<<<<<<< HEAD
        label_final = []

=======
            self.RAoffset.setText(str(self.offset_angle[0]))
            self.DECoffset.setText(str(self.offset_angle[1]))

        elif self.ctype == 'AZ and EL':
            self.AZoffset.setText(str(self.offset_angle[0]))
            self.ELoffset.setText(str(self.offset_angle[1]))
            
        elif self.ctype == 'CROSS-EL and EL':
            self.CROSSELoffset.setText(str(self.offset_angle[0]))
            self.ELxoffset.setText(str(self.offset_angle[1]))

    def load_func(self):


        '''
        Wrapper function to loaddata.py to read the DIRFILEs.
        If the paths are not correct a warning is generated. To reduce the time to 
        re-run the code everytime, a new DIRFILE is loaded a pickle object is created so it can be 
        loaded again when the plot button is pushed. The pickles object are deleted when
        the software is closed
        '''
        
        label_final = []
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
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
<<<<<<< HEAD
            detfile = os.stat(self.detpath.text()+'/'+self.detname.text())
=======
            os.stat(self.detpath.text()+'/'+self.detname.text())
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        except OSError:
            label = self.detpathlabel.text()
            label_final.append(label)
        if self.experiment.currentText().lower() == 'blast-tng':    
            try:
<<<<<<< HEAD
                coordfile = os.stat(self.coordpath.text())
=======
                os.stat(self.coordpath.text())
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            except OSError:
                label = self.coordpathlabel.text()
                label_final.append(label)
        elif self.experiment.currentText().lower() == 'blastpol':
            try:
<<<<<<< HEAD
                coordfile = os.stat(self.coordpath.text()+'/'+self.coord1.lower())
=======
                if self.coord1 == 'RA':
                    os.stat(self.coordpath.text()+'/'+self.coord1.lower())
                else:
                    os.stat(self.coordpath.text()+'/az')
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            except OSError:
                label = self.coord1.lower()+' coordinate'
                label_final.append(label)
            try:
<<<<<<< HEAD
                coordfile = os.stat(self.coordpath.text()+'/'+self.coord2.lower())
=======
                os.stat(self.coordpath.text()+'/'+self.coord2.lower())
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            except OSError:
                label = self.coord2.lower()+' coordinate'
                label_final.append(label)

        if np.size(label_final) != 0:
<<<<<<< HEAD
=======

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            self.warningbox = QMessageBox()
            self.warningbox.setIcon(QMessageBox.Warning)
            self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
=======
        self.setLayout(mainlayout)
        
class MapPlotsGroup(QWidget):
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs

    '''
    Generic layout to create a tabbed plot layout for maps
    in case only I is requested or also polarization maps
    are requested as output.
    This class is used for plotting both the maps and the beams 
    '''

    def __init__(self, data, checkbox, parent=None):

        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        self.data = data

        self.checkbox = checkbox

        self.tabvisible()

        self.tabs.addTab(self.tab1,"I Map")
        self.ImapTab()
        self.QmapTab()
        self.UmapTab()

        self.checkbox.toggled.connect(self.tabvisible)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def tabvisible(self):

        '''
        Function to update the visibility of the polarization maps.
        '''

        if self.checkbox.isChecked():
            self.Qsave = self.tabs.widget(1)
            self.tabs.removeTab(1)
            self.Usave = self.tabs.widget(1)
            self.tabs.removeTab(1)
        else:
<<<<<<< HEAD
<<<<<<< HEAD
            dataload = ld.data_value(self.detpath.text(), self.detname.text(), self.coordpath.text(), \
                                     self.coord1, self.coord2, self.dettype.text(), \
                                     self.coord1type.text(), self.coord2type.text(), \
                                     self.experiment.currentText())

            self.det_data, self.coord1_data, self.coord2_data = dataload.values()

<<<<<<< HEAD
            if self.DirConvCheckBox.isChecked:
                self.dirfile_conversion()
=======
        '''
        Create an empty plot for I map (or beam if the class is used in the beam tab)
        '''

        mainlayout = QGridLayout()
>>>>>>> 651e1e6... Commented files

=======
            
            os.makedirs(os.path.dirname('pickles_object/'), exist_ok=True)

            det_path_pickle = 'pickles_object/'+self.detname.text()
            coord1_path_pickle = 'pickles_object/'+self.coord1
            coord2_path_pickle = 'pickles_object/'+self.coord2

            try:               
                self.det_data = pickle.load(open(det_path_pickle, 'rb'))  
                self.coord1_data = pickle.load(open(coord1_path_pickle, 'rb'))
                self.coord2_data = pickle.load(open(coord2_path_pickle, 'rb'))

            except FileNotFoundError:          
                dataload = ld.data_value(self.detpath.text(), self.detname.text(), self.coordpath.text(), \
                                         self.coord1, self.coord2, self.dettype.text(), \
                                         self.coord1type.text(), self.coord2type.text(), \
                                         self.experiment.currentText())

                self.det_data, self.coord1_data, self.coord2_data = dataload.values()

                pickle.dump(self.det_data, open(det_path_pickle, 'wb'))  
                pickle.dump(self.coord1_data, open(coord1_path_pickle, 'wb'))
                pickle.dump(self.coord2_data, open(coord2_path_pickle, 'wb'))

                del dataload
                gc.collect()
            
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
            if self.experiment.currentText().lower() == 'blast-tng':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
                                                  self.endframe.text(), self.experiment.currentText(), \
                                                  roach_number = self.roachnumber.text(), \
                                                  roach_pps_path = self.coordpath.text())
            elif self.experiment.currentText().lower() == 'blastpol':
                zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
                                                  self.detframe.text(), self.coord1_data, \
                                                  self.coord2_data, self.acsfreq.text(), 
                                                  self.acsframe.text(), self.startframe.text(), \
<<<<<<< HEAD
                                                  self.endframe.text(), self.experiment.currentText())
<<<<<<< HEAD

<<<<<<< HEAD
            self.timemap, self.detslice, self.coord1slice, \
                                         self.coord2slice = zoomsyncdata.sync_data()
            print('tests', self.detslice)

    def clean_func(self):
=======
=======
            print(self.coord1_data)
>>>>>>> 8b07277... IQ -> power libraries
=======
                                                  self.endframe.text(), self.experiment.currentText(), offset)


>>>>>>> b96fc79... Added wcs projection to maps
            (self.timemap, self.detslice, self.coord1slice, \
             self.coord2slice) = zoomsyncdata.sync_data()

            if self.DirConvCheckBox.isChecked:
                self.dirfile_conversion()

            ### CONVERSION TO RADIANS FOR ALL THE ANGLES ###

            self.coord2slice = np.radians(self.coord2slice)

            if self.coord1.lower() == 'ra':
                self.coord1slice = np.radians(self.coord1slice*15.) #Conversion between hours to degree and then converted to radians
            elif self.coord1.lower() == 'cross-el':
                self.coord1slice = np.radians(self.coord1slice)*np.cos(self.coord2slice)
            else:
                self.coord1slice = np.radians(self.coord1slice)
            
            del self.det_data
            del self.coord1_data
            del self.coord2_data
            del zoomsyncdata
            gc.collect()

            self.clean_func()

    def clean_func(self):

        '''
        Function to compute the cleaned detector TOD
        '''

>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
        det_tod = tod.data_cleaned(self.detslice, self.detfreq.text(), self.highpassfreq.text())
        self.cleaned_data = det_tod.data_clean()

    def dirfile_conversion(self):

<<<<<<< HEAD
        det_conv = ld.convert_dirfile(self.det_data, float(self.adetconv.text()), \
                                      float(self.bdetconv.text()))
        coord1_conv = ld.convert_dirfile(self.coord1_data, float(self.acoord1conv.text()), \
                                         float(self.bcoord1conv.text()))
        coord2_conv = ld.convert_dirfile(self.coord2_data, float(self.acoord2conv.text()), \
                                         float(self.bcoord2conv.text()))

        self.det_data = det_conv.conversion()
        self.coord1_data = coord1_conv.conversion()
        self.coord2_data = coord2_conv.conversion()

    def beamLayout(self):

        self.createbeamplot()
=======
            try:
                self.tabs.insertTab(1, self.Qsave)
                self.tabs.insertTab(2, self.Usave)
            except:
                self.tabs.addTab(self.tab2,"Q Map")
                self.tabs.addTab(self.tab3,"U Map")
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs

    def ImapTab(self, button=None):

        mainlayout = QGridLayout()

        self.matplotlibWidget_Imap = MatplotlibWidget(self)
        self.axis_Imap = self.matplotlibWidget_Imap.figure.add_subplot(111)
        self.axis_Imap.set_axis_off()
        mainlayout.addWidget(self.matplotlibWidget_Imap)
        #self.map2d(self.data)
        # try:
        #     button.clicked.connect(partial(self.map2d,self.data))
        # except AttributeError:
        #     pass

        self.tab1.setLayout(mainlayout)

    def QmapTab(self):
        mainlayout = QGridLayout()

        self.matplotlibWidget_Qmap = MatplotlibWidget(self)
        self.axis_Qmap = self.matplotlibWidget_Qmap.figure.add_subplot(111)
        self.axis_Qmap.set_axis_off()
        mainlayout.addWidget(self.matplotlibWidget_Qmap)

        self.tab2.setLayout(mainlayout)

    def UmapTab(self):

        self.UmapGroup = QGroupBox("Detector Offset")
        mainlayout = QGridLayout()

        self.matplotlibWidget_Umap = MatplotlibWidget(self)
        self.axis_Umap = self.matplotlibWidget_Umap.figure.add_subplot(111)
        self.axis_Umap.set_axis_off()
        mainlayout.addWidget(self.matplotlibWidget_Umap)

        self.tab3.setLayout(mainlayout)

    def updateTab(self, data):
        if np.size(np.shape(data)) > 2:
            idx_list = ['I', 'Q', 'U']

            for i in range(len(idx_list)):
                self.map2d(data[i], idx_list[i])
        else:
            self.map2d(data)

    def map2d(self, data=None, idx='I'):
        
        if idx == 'I':
            axis = self.axis_Imap
        elif idx == 'Q':
            axis = self.axis_Qmap
        elif idx == 'U':
            axis = self.axis_Umap

        img = axis.images
        if np.size(img) > 0:
            cb = img[-1].colorbar
            cb.remove()

        axis.set_axis_on()
        axis.clear()
        axis.set_title('Maps')
        
        #max_index = np.where(np.abs(self.map_value) == np.amax((np.abs(self.map_value))))

        #levels = 5

        #interval = np.flip(np.linspace(0, 1, levels+1))

        #map_levels = self.map_value[max_index]*(1-interval)

        # extent = (np.amin(self.coord1slice), np.amax(self.coord1slice), \
        #           np.amin(self.coord2slice), np.amax(self.coord2slice))

        data_masked = np.ma.masked_where(data == 0, data)

        im = axis.imshow(data_masked, origin='lower', cmap=plt.cm.viridis)     
        plt.colorbar(im, ax=axis)

        # if self.ctype == 'RA and DEC':
        #     self.axis_map.set_xlabel('RA (deg)')
        #     self.axis_map.set_ylabel('Dec (deg)')
        # elif self.ctype == 'AZ and EL':
        #     self.axis_map.set_xlabel('Azimuth (deg)')
        #     self.axis_map.set_ylabel('Elevation (deg)')
        # elif self.ctype == 'CROSS-EL and EL':
        #     self.axis_map.set_xlabel('Cross Elevation (deg)')
        #     self.axis_map.set_ylabel('Elevation (deg)')

        self.matplotlibWidget_Imap.canvas.draw()

class MatplotlibWidget(QWidget):
=======
        '''
        Function to convert the DIRFILE data.
        '''

        det_conv = ld.convert_dirfile(self.detslice, float(self.adetconv.text()), \
                                      float(self.bdetconv.text()))
        coord1_conv = ld.convert_dirfile(self.coord1slice, float(self.acoord1conv.text()), \
                                         float(self.bcoord1conv.text()))
        coord2_conv = ld.convert_dirfile(self.coord2slice, float(self.acoord2conv.text()), \
                                         float(self.bcoord2conv.text()))

        det_conv.conversion()
        coord1_conv.conversion()
        coord2_conv.conversion()

        self.detslice = det_conv.data
        self.coord1slice = coord1_conv.data
        self.coord2slice = coord2_conv.data

    def mapvalues(self, data):

        '''
        Function to compute the maps
        '''

        self.ctype = self.coordchoice.currentText()

        self.crpix = np.array([int(float(self.crpix1.text())),\
                               int(float(self.crpix2.text()))])
        self.cdelt = np.array([float(self.cdelt1.text()),\
                               float(self.cdelt2.text())])
        self.crval = np.array([float(self.crval1.text()),\
                               float(self.crval2.text())])

        if self.convchoice.currentText().lower() == 'gaussian':
            self.convolution = True
            self.std = self.GaussianSTD.text()
        else:
            self.convolution = False
            self.std = 0

        self.maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
                            data, self.coord1slice, self.coord2slice, \
                            self.convolution, self.std, self.ICheckBox.isChecked())

        self.maps.wcs_proj()

        self.proj = self.maps.proj

        self.map_value = self.maps.map2d()

class TODTab(QWidget):

    '''
    Layout Class for the TOD tab
    '''

    def __init__(self, parent=None):

        super(QWidget, self).__init__(parent)

        self.c = ParamMapTab()

        self.createTODplot()
        self.createTODcleanedplot()

        mainlayout = QGridLayout()
        mainlayout.addWidget(self.TODplot, 0, 0)
        mainlayout.addWidget(self.TODcleanedplot, 0, 1)
        
        self.setLayout(mainlayout)

    def createTODplot(self, data = None):

        '''
        Function to create the TOD empty plot
        '''

        self.TODplot = QGroupBox("Detector TOD")
        TODlayout = QGridLayout()

        self.matplotlibWidget_TOD = MatplotlibWidget(self)
        self.axis_TOD = self.matplotlibWidget_TOD.figure.add_subplot(111)
        self.axis_TOD.set_axis_off()
        TODlayout.addWidget(self.matplotlibWidget_TOD)        

        self.TODplot.setLayout(TODlayout)

    def draw_TOD(self, data = None):

        '''
        Function to draw the TOD when the plot button is pushed.
        The plotted TOD is the one between the frame of interest
        '''
        
        self.axis_TOD.set_axis_on()
        self.axis_TOD.clear()
        try:
            self.axis_TOD.plot(data)
        except AttributeError:
            pass
        self.axis_TOD.set_title('detTOD')
        self.matplotlibWidget_TOD.canvas.draw()

    def createTODcleanedplot(self, data = None):

        '''
        Same of createTODPlot but for the cleanedTOD
        '''

        self.TODcleanedplot = QGroupBox("Detector Cleaned TOD")
        self.layout = QVBoxLayout()

        self.matplotlibWidget_cleaned_TOD = MatplotlibWidget(self)
        self.axis_cleaned_TOD = self.matplotlibWidget_cleaned_TOD.figure.add_subplot(111)
        self.axis_cleaned_TOD.set_axis_off()
        self.layout.addWidget(self.matplotlibWidget_cleaned_TOD)

        self.TODcleanedplot.setLayout(self.layout)

    def draw_cleaned_TOD(self, data = None):

        '''
        Same of draw_TOD but for the cleaned TOD
        '''
        
        self.axis_cleaned_TOD.set_axis_on()
        self.axis_cleaned_TOD.clear()
        try:           
            self.axis_cleaned_TOD.plot(data)
        except AttributeError or NameError or TypeError:
            pass
        self.axis_cleaned_TOD.set_title('Cleaned Data')
        self.matplotlibWidget_cleaned_TOD.canvas.draw()

class BeamTab(ParamMapTab):

    '''
    Layout for the tab used to show the calculated beams
    '''


    def __init__(self, parent=None, checkbox=None):

        super(QWidget, self).__init__(parent)

        c = ParamMapTab()

        beammaps = MapPlotsGroup(checkbox=c.ICheckBox)

        mainlayout = QGridLayout()
        mainlayout.addWidget(beammaps, 0, 0)
        
        self.setLayout(mainlayout)
        
class MapPlotsGroup(QWidget):

    '''
    Generic layout to create a tabbed plot layout for maps
    in case only I is requested or also polarization maps
    are requested as output.
    This class is used for plotting both the maps and the beams 
    '''

    def __init__(self, data, checkbox, projection=None, parent=None):

        super(QWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()

        
        self.data = data
        self.checkbox = checkbox
        self.projection = projection

        self.tabvisible()

        self.tabs.addTab(self.tab1,"I Map")
        self.ImapTab()
        self.QmapTab()
        self.UmapTab()

        self.checkbox.toggled.connect(self.tabvisible)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

    def tabvisible(self):

        '''
        Function to update the visibility of the polarization maps.
        '''

        if self.checkbox.isChecked():
            self.Qsave = self.tabs.widget(1)
            self.tabs.removeTab(1)
            self.Usave = self.tabs.widget(1)
            self.tabs.removeTab(1)
        else:
            try:
                self.tabs.insertTab(1, self.Qsave)
                self.tabs.insertTab(2, self.Usave)
            except:
                self.tabs.addTab(self.tab2,"Q Map")
                self.tabs.addTab(self.tab3,"U Map")

    def ImapTab(self, button=None):

        '''
        Create an empty plot for I map (or beam if the class is used in the beam tab)
        '''

        mainlayout = QGridLayout()

        self.matplotlibWidget_Imap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Imap)

        self.tab1.setLayout(mainlayout)

    def QmapTab(self):

        '''
        Same of ImapTab but the Q Stokes parameter
        '''

        mainlayout = QGridLayout()

        self.matplotlibWidget_Qmap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Qmap)

        self.tab2.setLayout(mainlayout)

    def UmapTab(self):
        
        '''
        Same of ImapTab but the U Stokes parameter
        '''

        self.UmapGroup = QGroupBox("Detector Offset")
        mainlayout = QGridLayout()

        self.matplotlibWidget_Umap = MatplotlibWidget(self)
        mainlayout.addWidget(self.matplotlibWidget_Umap)

        self.tab3.setLayout(mainlayout)

    def updateTab(self, data, projection = None):

        '''
        Function to updates the I, Q and U plots when the 
        button plot is pushed
        '''

        if np.size(np.shape(data)) > 2:
            idx_list = ['I', 'Q', 'U']

            for i in range(len(idx_list)):
                self.map2d(data[i], idx_list[i])
        else:
            self.map2d(data=data, projection=projection)

    def map2d(self, data=None, idx='I', projection=None):

        '''
        Function to generate the map plots (I,Q and U) 
        when the plot button is pushed
        '''
        
        #register_projection(projection)
        if idx == 'I':
            self.matplotlibWidget_Imap.figure.clear()
            self.axis_Imap = (self.matplotlibWidget_Imap.figure.add_subplot(111, \
                              projection=projection))
            axis = self.axis_Imap
        elif idx == 'Q':
            self.matplotlibWidget_Qmap.figure.clear()
            self.axis_Qmap = (self.matplotlibWidget_Qmap.figure.add_subplot(111, \
                              projection=projection))
            axis = self.axis_Qmap
        elif idx == 'U':
            self.matplotlibWidget_Umap.figure.clear()
            self.axis_Umap = (self.matplotlibWidget_Umap.figure.add_subplot(111, \
                              projection=projection))

            axis = self.axis_Umap

        img = axis.images
        if np.size(img) > 0:
            cb = img[-1].colorbar
            cb.remove()

        axis.set_axis_on()
        #axis.clear()
        axis.set_title('Maps')

        #print(axis.gca())
        
        #max_index = np.where(np.abs(self.map_value) == np.amax((np.abs(self.map_value))))

        #levels = 5

        #interval = np.flip(np.linspace(0, 1, levels+1))

        #map_levels = self.map_value[max_index]*(1-interval)

        # extent = (np.amin(self.coord1slice), np.amax(self.coord1slice), \
        #           np.amin(self.coord2slice), np.amax(self.coord2slice))

        data_masked = np.ma.masked_where(data == 0, data)

        im = axis.imshow(data_masked, origin='lower', cmap=plt.cm.viridis)     
        plt.colorbar(im, ax=axis)
        print('TEST')

        # if self.ctype == 'RA and DEC':
        #     self.axis_map.set_xlabel('RA (deg)')
        #     self.axis_map.set_ylabel('Dec (deg)')
        # elif self.ctype == 'AZ and EL':
        #     self.axis_map.set_xlabel('Azimuth (deg)')
        #     self.axis_map.set_ylabel('Elevation (deg)')
        # elif self.ctype == 'CROSS-EL and EL':
        #     self.axis_map.set_xlabel('Cross Elevation (deg)')
        #     self.axis_map.set_ylabel('Elevation (deg)')

        self.matplotlibWidget_Imap.canvas.draw()

class MatplotlibWidget(QWidget):

    '''
    Class to generate an empty matplotlib.pyplot object
    '''

<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> 651e1e6... Commented files
    def __init__(self, parent=None):
        super(MatplotlibWidget, self).__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layoutVertical = QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)
        self.layoutVertical.addWidget(self.toolbar)
<<<<<<< HEAD
<<<<<<< HEAD
=======
=======
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs

<<<<<<< HEAD
# class MainWindow(QTabWidget):
    
#     def __init__(self, parent = None):
#         super(MainWindow, self).__init__(parent)
#         self.tab1 = QWidget()
#         self.tab2 = QWidget()
#         self.tab3 = QWidget()
            
#         self.addTab(self.tab1,"Parameters and Maps")
#         self.addTab(self.tab2,"Detector TOD")
#         self.addTab(self.tab3, "Beam")
#         self.ParamMapLayout()
#         self.TODLayout()    
#         self.beamLayout()
#         self.setWindowTitle("Naive MapMaker")
        
#         self.show()

<<<<<<< HEAD
<<<<<<< HEAD

=======
#     def configuration_value(self):
#         text = self.experiment.currentText()
#         dir_path = os.getcwd()+'/config/'
        
#         filepath = dir_path+text.lower()+'.cfg'
#         model = configparser.ConfigParser()

#         model.read(filepath)
#         sections = model.sections()

#         for section in sections:
#             if section.lower() == 'experiment parameters':
#                 self.detfreq_config = float(model.get(section, 'DETFREQ').split('#')[0])
#                 self.acsfreq_config = float(model.get(section, 'ACSFREQ').split('#')[0])
#                 det_dir_conv = model.get(section,'DET_DIR_CONV').split('#')[0].strip()
#                 self.detconv_config = np.array(det_dir_conv.split(',')).astype(float)
#                 coor1_dir_conv = model.get(section,'COOR1_DIR_CONV').split('#')[0].strip()
#                 self.coord1conv_config = np.array(coor1_dir_conv.split(',')).astype(float)
#                 coor2_dir_conv = model.get(section,'COOR2_DIR_CONV').split('#')[0].strip()
#                 self.coord2conv_config = np.array(coor2_dir_conv.split(',')).astype(float)
#                 self.detframe_config = float(model.get(section, 'DET_SAMP_FRAME').split('#')[0])
#                 self.acsframe_config = float(model.get(section, 'ACS_SAMP_FRAME').split('#')[0])
#                 self.dettype_config = model.get(section,'DET_FILE_TYPE').split('#')[0].strip()
#                 self.coord1type_config = model.get(section,'COOR1_FILE_TYPE').split('#')[0].strip()
#                 self.coord2type_config = model.get(section,'COOR2_FILE_TYPE').split('#')[0].strip()

#         self.detfreq.setText(str(self.detfreq_config))
#         self.acsfreq.setText(str(self.acsfreq_config))
#         self.detframe.setText(str(self.detframe_config))
#         self.acsframe.setText(str(self.acsframe_config))
#         self.dettype.setText(str(self.dettype_config))
#         self.coord1type.setText(str(self.coord1type_config))
#         self.coord2type.setText(str(self.coord2type_config))
#         self.adetconv.setText(str(self.detconv_config[0]))
#         self.bdetconv.setText(str(self.detconv_config[1]))
#         self.acoord1conv.setText(str(self.coord1conv_config[0]))
#         self.bcoord1conv.setText(str(self.coord1conv_config[1]))
#         self.acoord2conv.setText(str(self.coord2conv_config[0]))
#         self.bcoord2conv.setText(str(self.coord2conv_config[1]))
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
=======

>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

#     def createMapPlotGroup(self):
#         self.MapPlotGroup = QTabWidget()

#         self.mapTab1 = QWidget()
#         self.mapTab2 = QWidget()
#         self.mapTab3 = QWidget()

#         self.layout = QVBoxLayout()  
#         self.matplotlibWidget_Map = MatplotlibWidget(self)
#         self.axis_map = self.matplotlibWidget_Map.figure.add_subplot(111)
#         self.axis_map.set_axis_off()
#         self.layout.addWidget(self.matplotlibWidget_Map)
#         self.plotbutton.clicked.connect(self.map2d)

#         self.MapPlotGroup.setLayout(self.layout)
           
#     def map2d(self, data=None):
#         data = [random.random() for i in range(25)]

#         self.ctype = self.coordchoice.currentText()

#         self.crpix = np.array([int(float(self.crpix1.text())),\
#                                int(float(self.crpix2.text()))])
#         self.cdelt = np.array([float(self.cdelt1.text()),\
#                                float(self.cdelt2.text())])
#         self.crval = np.array([float(self.crval1.text()),\
#                                float(self.crval2.text())])

#         if self.convchoice.currentText().lower() == 'gaussian':
#             self.convolution = True
#             self.std = self.GaussianSTD.text()
#         else:
#             self.convolution = False
#             self.std = 0

#         self.maps = mp.maps(self.ctype, self.crpix, self.cdelt, self.crval, \
#                        self.cleaned_data, self.coord1slice, self.coord2slice, \
#                        self.convolution, self.std)

#         self.maps.wcs_proj()

#         self.map_value = self.maps.map2d()
        
#         self.axis_map.set_axis_on()
#         self.axis_map.clear()
#         self.axis_map.set_title('Maps')
        
#         max_index = np.where(np.abs(self.map_value) == np.amax((np.abs(self.map_value))))

#         levels = 5

#         interval = np.flip(np.linspace(0, 1, levels+1))

#         map_levels = self.map_value[max_index]*(1-interval)

#         extent = (np.amin(self.coord1slice), np.amax(self.coord1slice), \
#                   np.amin(self.coord2slice), np.amax(self.coord2slice))
#         im = self.axis_map.imshow(self.map_value, extent = extent, origin='lower', cmap=plt.cm.viridis)
#         plt.colorbar(im)

#         if self.ctype == 'RA and DEC':
#             self.axis_map.set_xlabel('RA (deg)')
#             self.axis_map.set_ylabel('Dec (deg)')
#         elif self.ctype == 'AZ and EL':
#             self.axis_map.set_xlabel('Azimuth (deg)')
#             self.axis_map.set_ylabel('Elevation (deg)')
#         elif self.ctype == 'CROSS-EL and EL':
#             self.axis_map.set_xlabel('Cross Elevation (deg)')
#             self.axis_map.set_ylabel('Elevation (deg)')

#         self.matplotlibWidget_Map.canvas.draw()

#     def createOffsetGroup(self):

#         self.OffsetGroup = QGroupBox("Detector Offset")

#         self.RAoffsetlabel = QLabel('RA (deg)')
#         self.DECoffsetlabel = QLabel('Dec (deg)')
#         self.RAoffset = QLineEdit('')
#         self.DECoffset = QLineEdit('')

#         self.AZoffsetlabel = QLabel('Azimuth (deg)')
#         self.ELoffsetlabel = QLabel('Elevation (deg)')
#         self.AZoffset = QLineEdit('')
#         self.ELoffset = QLineEdit('')

#         self.CROSSELoffsetlabel = QLabel('Cross Elevation (deg)')
#         self.ELxoffsetlabel = QLabel('Elevation (deg)')
#         self.CROSSELoffset = QLineEdit('')
#         self.ELxoffset = QLineEdit('')

#         self.coordchoice.activated[str].connect(self.updateOffsetLabel)
#         self.plotbutton.clicked.connect(self.updateOffsetValue)

#         self.updateOffsetLabel()

#         self.layout = QGridLayout()
#         self.layout.addWidget(self.RAoffsetlabel, 0, 0)
#         self.layout.addWidget(self.RAoffset, 0, 1)
#         self.layout.addWidget(self.DECoffsetlabel, 1, 0)
#         self.layout.addWidget(self.DECoffset, 1, 1)
#         self.layout.addWidget(self.AZoffsetlabel, 0, 0)
#         self.layout.addWidget(self.AZoffset, 0, 1)
#         self.layout.addWidget(self.ELoffsetlabel, 1, 0)
#         self.layout.addWidget(self.ELoffset, 1, 1)
#         self.layout.addWidget(self.CROSSELoffsetlabel, 0, 0)
#         self.layout.addWidget(self.CROSSELoffset, 0, 1)
#         self.layout.addWidget(self.ELxoffsetlabel, 1, 0)
#         self.layout.addWidget(self.ELxoffset, 1, 1)

#         self.RAoffset.setEnabled(False)
#         self.DECoffset.setEnabled(False)

#         self.AZoffset.setEnabled(False)
#         self.ELoffset.setEnabled(False)

#         self.ELxoffset.setEnabled(False)
#         self.CROSSELoffset.setEnabled(False)
        
#         self.OffsetGroup.setLayout(self.layout)

#     def updateOffsetLabel(self):

#         self.ctype = self.coordchoice.currentText()

#         if self.ctype == 'RA and DEC':
#             self.RADECvisibile = True
#             self.AZELvisibile = False
#             self.CROSSEL_ELvisibile = False

#         elif self.ctype == 'AZ and EL':
#             self.RADECvisibile = False
#             self.AZELvisibile = True
#             self.CROSSEL_ELvisibile = False
            
#         elif self.ctype == 'CROSS-EL and EL':
#             self.RADECvisibile = False
#             self.AZELvisibile = False
#             self.CROSSEL_ELvisibile = True


#         self.RAoffset.setVisible(self.RADECvisibile)
#         self.DECoffset.setVisible(self.RADECvisibile)
#         self.RAoffsetlabel.setVisible(self.RADECvisibile)
#         self.DECoffsetlabel.setVisible(self.RADECvisibile)

#         self.AZoffset.setVisible(self.AZELvisibile)
#         self.ELoffset.setVisible(self.AZELvisibile)
#         self.AZoffsetlabel.setVisible(self.AZELvisibile)
#         self.ELoffsetlabel.setVisible(self.AZELvisibile)      

#         self.CROSSELoffset.setVisible(self.CROSSEL_ELvisibile)
#         self.ELxoffset.setVisible(self.CROSSEL_ELvisibile)
#         self.CROSSELoffsetlabel.setVisible(self.CROSSEL_ELvisibile)
#         self.ELxoffsetlabel.setVisible(self.CROSSEL_ELvisibile) 

#     def updateOffsetValue(self):
        
#         ctype_map = self.coordchoice.currentText()

#         offset = bm.computeoffset(self.map_value, float(self.crval1.text()), float(self.crval2.text()), ctype_map)

#         self.offset_angle = offset.offset(self.maps.proj)

#         if self.ctype == 'RA and DEC':
#             self.RAoffset = QLineEdit(str(self.offset_angle[0]))
#             self.DECoffset = QLineEdit(str(self.offset_angle[1]))

#         elif self.ctype == 'AZ and EL':
#             self.AZoffset = QLineEdit(str(self.offset_angle[0]))
#             self.ELoffset = QLineEdit(str(self.offset_angle[1]))
            
#         elif self.ctype == 'CROSS-EL and EL':
#             self.CROSSELoffset = QLineEdit(str(self.offset_angle[0]))
#             self.ELxoffset = QLineEdit(str(self.offset_angle[1]))

#     def TODLayout(self):

#         

<<<<<<< HEAD
<<<<<<< HEAD

=======
#     def load_func(self):
        
#         label_final = []

#         coord_type = self.coordchoice.currentText()
#         if coord_type == 'RA and DEC':
#             self.coord1 = str('RA')
#             self.coord2 = str('DEC')
#         elif coord_type == 'AZ and EL':
#             self.coord1 = str('AZ')
#             self.coord2 = str('EL')
#         elif coord_type == 'CROSS-EL and EL':
#             self.coord1 = str('CROSS-EL')
#             self.coord2 = str('EL')

#         try:
#             detfile = os.stat(self.detpath.text()+'/'+self.detname.text())
#         except OSError:
#             label = self.detpathlabel.text()
#             label_final.append(label)
#         if self.experiment.currentText().lower() == 'blast-tng':    
#             try:
#                 coordfile = os.stat(self.coordpath.text())
#             except OSError:
#                 label = self.coordpathlabel.text()
#                 label_final.append(label)
#         elif self.experiment.currentText().lower() == 'blastpol':
#             try:
#                 coordfile = os.stat(self.coordpath.text()+'/'+self.coord1.lower())
#             except OSError:
#                 label = self.coord1.lower()+' coordinate'
#                 label_final.append(label)
#             try:
#                 coordfile = os.stat(self.coordpath.text()+'/'+self.coord2.lower())
#             except OSError:
#                 label = self.coord2.lower()+' coordinate'
#                 label_final.append(label)

#         if np.size(label_final) != 0:
#             self.warningbox = QMessageBox()
#             self.warningbox.setIcon(QMessageBox.Warning)
#             self.warningbox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

#             self.warningbox.setWindowTitle('Warning')

#             msg = 'Incorrect Path(s): \n'
#             for i in range(len(label_final)): 
#                 msg += (str(label_final[i][:-1])) +'\n'
            
#             self.warningbox.setText(msg)        
        
#             self.warningbox.exec_()

#         else:
#             dataload = ld.data_value(self.detpath.text(), self.detname.text(), self.coordpath.text(), \
#                                      self.coord1, self.coord2, self.dettype.text(), \
#                                      self.coord1type.text(), self.coord2type.text(), \
#                                      self.experiment.currentText())

#             self.det_data, self.coord1_data, self.coord2_data = dataload.values()

#             if self.DirConvCheckBox.isChecked:
#                 self.dirfile_conversion()

#             if self.experiment.currentText().lower() == 'blast-tng':
#                 zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
#                                                   self.detframe.text(), self.coord1_data, \
#                                                   self.coord2_data, self.acsfreq.text(), 
#                                                   self.acsframe.text(), self.startframe.text(), \
#                                                   self.endframe.text(), self.experiment.currentText(), \
#                                                   roach_number = self.roachnumber.text(), \
#                                                   roach_pps_path = self.coord_path.text())
#             elif self.experiment.currentText().lower() == 'blastpol':
#                 zoomsyncdata = ld.frame_zoom_sync(self.det_data, self.detfreq.text(), \
#                                                   self.detframe.text(), self.coord1_data, \
#                                                   self.coord2_data, self.acsfreq.text(), 
#                                                   self.acsframe.text(), self.startframe.text(), \
#                                                   self.endframe.text(), self.experiment.currentText())

#             self.timemap, self.detslice, self.coord1slice, \
#                                          self.coord2slice = zoomsyncdata.sync_data()
#             print('tests', self.detslice)

#     def clean_func(self):
#         det_tod = tod.data_cleaned(self.detslice, self.detfreq.text(), self.highpassfreq.text())
#         self.cleaned_data = det_tod.data_clean()

#     def dirfile_conversion(self):

#         det_conv = ld.convert_dirfile(self.det_data, float(self.adetconv.text()), \
#                                       float(self.bdetconv.text()))
#         coord1_conv = ld.convert_dirfile(self.coord1_data, float(self.acoord1conv.text()), \
#                                          float(self.bcoord1conv.text()))
#         coord2_conv = ld.convert_dirfile(self.coord2_data, float(self.acoord2conv.text()), \
#                                          float(self.bcoord2conv.text()))

#         self.det_data = det_conv.conversion()
#         self.coord1_data = coord1_conv.conversion()
#         self.coord2_data = coord2_conv.conversion()
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
=======

>>>>>>> 6c0d8b1... Solved some errors in polarization maps (still some to be corrected)

#     def beamLayout(self):

#         self.createbeamplot()

#         mainlayout = QGridLayout()
#         mainlayout.addWidget(self.beamplot, 0, 0)

#         self.tab3.setLayout(mainlayout)

#     def createbeamplot(self, data = None):

#         self.beamplot = QGroupBox("Beam Map")
        
#         self.layout = QVBoxLayout()

#         self.matplotlibWidget_beam = MatplotlibWidget(self)
#         self.axis_beam = self.matplotlibWidget_beam.figure.add_subplot(111)
#         self.axis_beam.set_axis_off()
#         self.layout.addWidget(self.matplotlibWidget_beam)        
#         self.plotbutton.clicked.connect(partial(self.draw_beam, data))

#         self.beamplot.setLayout(self.layout)

#     def draw_beam(self, data = None):
#         self.axis_beam.set_axis_on()
#         self.axis_beam.clear()
#         try:
#             beam = bm.beam(self.map_value)
#             self.beam_data, self.beam_param, self.beam_error = beam.beam_fit()
#             self.axis_beam.imshow(self.beam_data)
#         except AttributeError:
#             pass
#         self.axis_beam.set_title('Beam')
#         self.matplotlibWidget_beam.canvas.draw()
=======
>>>>>>> cab17d8... update repo


<<<<<<< HEAD
>>>>>>> c2f9e18a58705b8f7b3979aa1ee2eb19c9939d72
=======
>>>>>>> fbc01b1... Gui updated with polarizarion maps tabs
