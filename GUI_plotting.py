import sys
import os
import time
from PyQt4 import Qt
from PyQt4 import QtGui, QtCore

PARAMS = []
PARAMS_values = []
PARAMS_total = []

current_directory = os.getcwd()

python_file = 'python 20_12_16_plotting.py'

class Ressonant_P09(Qt.QWidget):

    def __init__(self, parent=None):

        Qt.QWidget.__init__(self, parent)

        self.xmlvalues = {}

        self.xmlvalues['working directory'] = '%s'%current_directory
        self.xmlvalues['file to open'] = '071016_0108.txt'
        self.xmlvalues['first angle'] = '-10'
        self.xmlvalues['last angle'] = '10'
        self.xmlvalues['low energy'] = '150'
        self.xmlvalues['high_energy'] = '152'
        self.xmlvalues['times_delta_e'] ='5'
        self.xmlvalues['cut_file_output'] = 'output_file.dat'
        self.xmlvalues['window_size'] = '11'
        self.xmlvalues['order'] = '4'

        self.layout_name = self.setLayout(Qt.QGridLayout())

        self.Ressonant = Qt.QPushButton('PLOT')
        self.layout().addWidget(self.Ressonant, 0, 0)
        self.connect(self.Ressonant, Qt.SIGNAL('clicked()'), self.doRessonant)

        self.plot_image = Qt.QPushButton('PLOT all intensities')
        self.layout().addWidget(self.plot_image, 30, 0)
        self.connect(self.plot_image, Qt.SIGNAL('clicked()'), self.plot_all_intensities_and_filtering)

        self.plot_image_bis = Qt.QPushButton('PLOT image intensities')
        self.layout().addWidget(self.plot_image_bis, 30, 1)
        self.connect(self.plot_image_bis, Qt.SIGNAL('clicked()'), self.plot_image_all_intensities_and_cut)

        self.plot_image_3 = Qt.QPushButton('PLOT 3D')
        self.layout().addWidget(self.plot_image_3, 30, 2)
        self.connect(self.plot_image_3, Qt.SIGNAL('clicked()'), self.plot_3D)

        self.save_data_file = Qt.QPushButton('Save Data')
        self.layout().addWidget(self.save_data_file, 19, 2)
        self.connect(self.save_data_file, Qt.SIGNAL('clicked()'), self.save_data)

        self.update_data = Qt.QPushButton('Update')
        self.layout().addWidget(self.update_data, 7, 2)
        self.connect(self.update_data, Qt.SIGNAL('clicked()'), self.read_update_data)
        self.update_data.setEnabled(False)


        self.textEdit = Qt.QLineEdit()
        #self.layout().addWidget(self.textEdit, 0,3)
        self.layout().addWidget(self.textEdit, 0,3)

        self.search_path = Qt.QPushButton('Search File')
        self.layout().addWidget(self.search_path, 0,2)
        self.connect(self.search_path, Qt.SIGNAL('clicked()'), self.openFileDialog)
        #self.textEdit.setGeometry(QtCore.QRect(20,180,301,181))



        # POPULATE THE 'QUICKGRID'
        self.value_widgets = []
        last_diffplan_row = 1
        last_expcond_row = 1
        last_sample_row = 1




        # short list
        short_list = [

            'working directory',
            'file to open',
            'first angle',
            'last angle',
            'low energy',
            'high_energy',
            'times_delta_e',
            'cut_file_output',
            'window_size',
            'order'

        ]

        for k in short_list:

            print k

            v = self.xmlvalues[k]

            # PARAMS.append(k+"  "+v)
            PARAMS.append(k)
            PARAMS_values.append(v)

            #PARAMS_total.append(k + " " + v)

            w_label = Qt.QLabel(k)
            w_value = Qt.QLineEdit()
            w_value.setObjectName(k)
            w_value.setText(v)

            self.value_widgets.append(w_value)
            self.connect(w_value, Qt.SIGNAL('textChanged(QString)'), self.updateXMLValues)


            #if k.startswith('path'):
            self.layout().addWidget(w_label, last_expcond_row, 0)
            self.layout().addWidget(w_value, last_expcond_row, 1)
            last_expcond_row += 1

            # elif k.startswith('concentration'):
            #self.layout().addWidget(w_label, last_diffplan_row + 1, 0)
            #self.layout().addWidget(w_value, last_diffplan_row + 1, 1)
            #self.layout().addWidget(w_value, last_diffplan_row + 2, 1)
            #last_diffplan_row += 1

            #self.layout().addWidget(w_label, last_sample_row + 2, 0)
            #self.layout().addWidget(w_value, last_sample_row + 2, 1)
            #last_sample_row += 1

            #self.layout().addWidget(w_label, last_sample_row + 3, 0)
            #self.layout().addWidget(w_value, last_sample_row + 3, 1)
            #last_sample_row += 1


    def updateXMLValues(self, *args):

        w = self.sender()
        param = str(w.objectName())

        if param != 'dataset':
            value = args[0]
            self.xmlvalues[param] = value

        i = 0
        for item in PARAMS:
            # print value
            if item == param:
                PARAMS_values[i] = "%s" % str(value)

            i = i + 1


    def update_angle(self):
        pass


    def doRessonant(self):

        print PARAMS_values
        os.system('%s %s %s %s %s %s %s %s %s %s &'%(python_file, PARAMS_values[1], PARAMS_values[2], PARAMS_values[3], PARAMS_values[4], PARAMS_values[5], PARAMS_values[6], PARAMS_values[7],PARAMS_values[8],PARAMS_values[9]))

    def plot_all_intensities_and_filtering(self):

        graph_mode = 'all_intensities'
        os.system('%s %s %s %s %s %s %s %s %s %s all_intensities &'%(python_file, PARAMS_values[1], PARAMS_values[2], PARAMS_values[3], PARAMS_values[4], PARAMS_values[5], PARAMS_values[6], PARAMS_values[7],PARAMS_values[8],PARAMS_values[9]))

    def plot_image_all_intensities_and_cut(self):

        os.system('%s %s %s %s %s %s %s %s %s %s image_intensities &' % (python_file,
        PARAMS_values[1], PARAMS_values[2], PARAMS_values[3], PARAMS_values[4], PARAMS_values[5], PARAMS_values[6],
        PARAMS_values[7], PARAMS_values[8], PARAMS_values[9]))

    def plot_3D(self):
        os.system('%s %s %s %s %s %s %s %s %s %s 3_D &' % (python_file,
        PARAMS_values[1], PARAMS_values[2], PARAMS_values[3], PARAMS_values[4], PARAMS_values[5], PARAMS_values[6],
        PARAMS_values[7], PARAMS_values[8], PARAMS_values[9]))

    def save_data(self):
        os.system('%s %s %s %s %s %s %s %s %s %s save_data &' % (python_file,
        PARAMS_values[1], PARAMS_values[2], PARAMS_values[3], PARAMS_values[4], PARAMS_values[5], PARAMS_values[6],
        PARAMS_values[7], PARAMS_values[8], PARAMS_values[9]))

        print 'save data'


    def read_update_data(self):
        print 'read update data'
        self.update_angle()

    def openFileDialog(self):

        filename = QtGui.QFileDialog.getOpenFileName(self, "open file") #, "%s" % self.start_path)
        print filename
        self.textEdit.setText(filename)
        #open_file = self.xmlvalues['file to open']
        #w_value.setObjectName('file to open')
        #self.connect(w_value, Qt.SIGNAL('textChanged(QString)'), self.updateXMLValues)





if __name__ == '__main__':

    app = Qt.QApplication(sys.argv)
    w = Ressonant_P09()
    w.show()
    sys.exit(app.exec_())