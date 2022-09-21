import logging
import os.path
import sys

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QPushButton, QFileDialog, QSpinBox, QComboBox, QLabel
from PyQt5.uic import loadUi

from domino.plotting import state_to_ram
from domino.add_log_level import addLoggingLevel
from .calc_mappings import img_file_pattern, img_file_labels, debug_loc
from .calculate import Calculator
from .parameters import Parameters
from .params_mapping import getters

logger = logging.getLogger('custom')
logger.propagate = False
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(debug_loc)
fh.setLevel(logging.DEBUG)
# create formatter and add it to the handlers
formatter = logging.Formatter("%(message)s")
fh.setFormatter(formatter)
# add the handlers to logger
logger.addHandler(fh)
script_dir = os.path.dirname(__file__)
gui_xml = os.path.join(script_dir,  "untitled.ui")

class State:
    stopped = 0
    running = 1


class Worker(QObject):
    finished = pyqtSignal()
    image_ready = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.result = None

    def run(self):
        """Long-running task."""
        c = Calculator(self.params)
        c.calc()
        c.multirun_stats()
        self.result = c.experiments
        self.finished.emit()


class UserInterface(QtWidgets.QMainWindow):

    def __init__(self):
        super(UserInterface, self).__init__()
        loadUi(gui_xml, self)
        self.setFixedSize(self.width(), self.height())
        self.start_pushButton: QPushButton
        self.start_pushButton.clicked.connect(self.start)
        self.readca_btn.clicked.connect(self.getfile)
        self.readstr_btn.clicked.connect(self.getfile)
        self.debug_checkBox.stateChanged.connect(self.set_debug_btn)
        self.seedCheckBox.stateChanged.connect(lambda: self.seedLineEdit.setEnabled(self.seedCheckBox.isChecked()))

        self.spinBox_iter_step: QSpinBox
        self.spinBox_iter_step.valueChanged.connect(self.change_img)
        self.spinBox_exp_no.valueChanged.connect(self.change_img)

        self.comboBox_states: QComboBox
        self.comboBox_states.currentIndexChanged.connect(self.change_img)
        # self.comboBox_states.currentTextChanged.connect(self.change_img)
        self.plot_label: QLabel
        self.plot_label.setScaledContents(True)

        self.state = State.stopped
        self.num_iter = 0
        self.num_exper = 0

        self.experiments = None

        self.show()

    def set_debug_btn(self):
        self.readca_btn.setEnabled(self.debug_checkBox.isChecked())
        self.readstr_btn.setEnabled(self.debug_checkBox.isChecked())

    def getfile(self):
        fname = QFileDialog.getOpenFileName(self, 'Load array',
                                            '.')
        rel_file = os.path.relpath(fname[0])
        sender: QPushButton = self.sender()
        sender.setText(rel_file)

    def start(self):
        if self.state == State.stopped:

            params_records = (
                (self.m2RowsLineEdit, "mrows"),
                (self.n2ColsLineEdit, "ncols"),
                (self.p_init_CLineEdit, "p_init_c"),
                (self.sharingCheckBox, "sharing"),
                (self.computationTypeComboBox, "competition_type"),
                (self.p_state_mutLineEdit, "p_state_mut"),
                (self.p_strat_mutLineEdit, "p_strat_mut"),
                (self.p_0_neighLineEdit, "p_0_neigh"),
                (self.num_of_iterLineEdit, "num_of_iter"),
                (self.num_of_experLineEdit, "num_of_exper"),
                (self.seedCheckBox, "if_seed"),
                (self.seedLineEdit, "seed"),

                (self.dd_penalty_lineEdit, "dd_penalty"),
                (self.dc_penalty_lineEdit, "dc_penalty"),
                (self.dd_reward_lineEdit, "dd_reward"),
                (self.dc_reward_lineEdit, "dc_reward"),
                (self.cd_reward_lineEdit, "cd_reward"),
                (self.cc_penalty_lineEdit, "cc_penalty"),
                (self.special_penalty_checkbox, "if_special_penalty"),
                (self.special_penalty_LineEdit, "special_penalty"),

                (self.allCLineEdit, "all_c"),
                (self.allDLineEdit, "all_d"),
                (self.kDLineEdit, "k_d"),
                (self.kCLineEdit, "k_c"),
                (self.kDCLineEdit, "k_dc"),
                (self.kconst_LineEdit, "k_const"),
                (self.kvar1_LineEdit, "k_var_0"),
                (self.kvar2_LineEdit, "k_var_1"),
                (self.k_buttonGroup, "k_change"),

                (self.level_LineEdit, "synchronization"),
                (self.log_to_debug_checkBox, "log_to_debug"),
                (self.debug_checkBox, "load_init_files"),
                (self.readca_btn, "state_filename"),
                (self.readstr_btn, "strat_filename")

            )

            param_dict = {}

            for ui_object, param_object in params_records:
                ui_getter = getters[type(ui_object)]
                param_dict[param_object] = ui_getter(ui_object)

            params = Parameters(**param_dict).freeze()
            self.num_iter = params.num_of_iter
            self.num_exper = params.num_of_exper
            self.start_calc(params)
            self.state = State.running
            self.start_pushButton.setEnabled(False)

    def start_calc(self, params):
        # Step 2: Create a QThread object
        self.thread = QThread()
        # Step 3: Create a worker object
        self.worker = Worker(params)
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress.connect(self.reportProgress)
        # Step 6: Start the thread
        self.thread.start()
        self.thread.finished.connect(
            self.on_stop_calc
        )

    def reportProgress(self):
        iter_num = self.spinBox_iter_step.setValue(self.num_iter)
        exper_num = self.spinBox_exp_no.setValue(self.num_exper)
        self.change_img()

    def on_stop_calc(self):
        self.state = State.stopped
        self.experiments = self.worker.result

        self.start_pushButton.setEnabled(True)

        self.spinBox_iter_step.setEnabled(True)
        self.spinBox_iter_step.setMaximum(self.num_iter)
        self.spinBox_exp_no.setEnabled(True)
        self.spinBox_exp_no.setMaximum(self.num_exper)
        self.comboBox_states.setEnabled(True)
        self.change_img()

    def change_img(self):
        iter_num = self.spinBox_iter_step.value()
        exper_num = self.spinBox_exp_no.value()
        self.comboBox_states: QComboBox
        disp_state = self.comboBox_states.currentIndex()

        if 0 <= iter_num <= self.num_iter and 0 < exper_num <= self.num_exper and 0 <= disp_state < len(img_file_labels):
            # img_file = img_file_pattern % (exper_num, img_file_labels[disp_state], iter_num)
            # pixmap = QPixmap(img_file)

            experiment = self.experiments[exper_num-1]
            state = experiment.history[iter_num]
            label = img_file_labels[disp_state]
            img_data = state_to_ram(state, label)

            img = Image.open(img_data)
            img_qt = ImageQt(img)

            pixmap = QtGui.QPixmap.fromImage(img_qt)

            self.plot_label.setPixmap(pixmap)
            self.plot_label.show()


from os import environ


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


def main():
    suppress_qt_warnings()
    app = QtWidgets.QApplication(sys.argv)
    window = UserInterface()
    app.exec_()


if __name__ == "__main__":
    main()
