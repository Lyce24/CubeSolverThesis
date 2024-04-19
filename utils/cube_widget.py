from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIntValidator
from cube import Cube, Move
from PyQt5.QtWidgets import QApplication
from search import MAWAStar, MBS, MWAStar

# write the number in scientific notation => 1000 -> 1.0 x 10^3
def scientific_notation(num):
    return "{:.1e}".format(num)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Create the main window
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 960)
        
        self.actions_delay = 250
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # Vertical layout for the main window
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.gridLayout_7 = QtWidgets.QGridLayout()
        self.gridLayout_7.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout_7.setObjectName("gridLayout_7")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem2, 0, 3, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem3, 0, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem4, 0, 0, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem5, 2, 0, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem6, 2, 3, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_7.addItem(spacerItem7, 2, 2, 1, 1)


        """
                      |  2  5  8 |
                      |  1  4  7 |
                      |  0  3  6 |
             --------------------------------------------
             20 23 26 | 47 50 53 | 29 32 35 | 38 41 44
             19 22 25 | 46 49 52 | 28 31 34 | 37 40 43
             18 21 24 | 45 48 51 | 27 30 33 | 36 39 42
             --------------------------------------------           
                      | 11 14 17 |
                      | 10 13 16 |
                      | 9  12 15 |
        
        """
        
        ### Front Face - Orange
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.gridLayout_3.setSpacing(0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.subface_47 = self.setupsubfaces("orange", "cubie_47", self.gridLayout_3, 0, 0)
        self.subface_46 = self.setupsubfaces("orange", "cubie_46", self.gridLayout_3, 1, 0)
        self.subface_45 = self.setupsubfaces("orange", "cubie_45", self.gridLayout_3, 2, 0)
        self.subface_50 = self.setupsubfaces("orange", "cubie_50", self.gridLayout_3, 0, 1)
        self.subface_49 = self.setupsubfaces("orange", "cubie_49", self.gridLayout_3, 1, 1)
        self.subface_48 = self.setupsubfaces("orange", "cubie_48", self.gridLayout_3, 2, 1)
        self.subface_53 = self.setupsubfaces("orange", "cubie_53", self.gridLayout_3, 0, 2)
        self.subface_52 = self.setupsubfaces("orange", "cubie_52", self.gridLayout_3, 1, 2)
        self.subface_51 = self.setupsubfaces("orange", "cubie_51", self.gridLayout_3, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout_3, 1, 1, 1, 1)
        
        ### Up Face - White
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.subface_2 = self.setupsubfaces("white", "cubie_2", self.gridLayout, 0, 0)
        self.subface_1 = self.setupsubfaces("white", "cubie_1", self.gridLayout, 1, 0)
        self.subface_0 = self.setupsubfaces("white", "cubie_0", self.gridLayout, 2, 0)
        self.subface_5 = self.setupsubfaces("white", "cubie_5", self.gridLayout, 0, 1)
        self.subface_4 = self.setupsubfaces("white", "cubie_4", self.gridLayout, 1, 1)
        self.subface_3 = self.setupsubfaces("white", "cubie_3", self.gridLayout, 2, 1)
        self.subface_8 = self.setupsubfaces("white", "cubie_8", self.gridLayout, 0, 2)
        self.subface_7 = self.setupsubfaces("white", "cubie_7", self.gridLayout, 1, 2)
        self.subface_6 = self.setupsubfaces("white", "cubie_6", self.gridLayout, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout, 0, 1, 1, 1)
        
        ### Left Face - Blue
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setSpacing(0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.subface_20 = self.setupsubfaces("blue", "cubie_20", self.gridLayout_2, 0, 0)
        self.subface_19 = self.setupsubfaces("blue", "cubie_19", self.gridLayout_2, 1, 0)
        self.subface_18 = self.setupsubfaces("blue", "cubie_18", self.gridLayout_2, 2, 0)
        self.subface_23 = self.setupsubfaces("blue", "cubie_23", self.gridLayout_2, 0, 1)
        self.subface_22 = self.setupsubfaces("blue", "cubie_22", self.gridLayout_2, 1, 1)
        self.subface_21 = self.setupsubfaces("blue", "cubie_21", self.gridLayout_2, 2, 1)
        self.subface_26 = self.setupsubfaces("blue", "cubie_26", self.gridLayout_2, 0, 2)
        self.subface_25 = self.setupsubfaces("blue", "cubie_25", self.gridLayout_2, 1, 2)
        self.subface_24 = self.setupsubfaces("blue", "cubie_24", self.gridLayout_2, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        
        ### Back Face - Red
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setSpacing(0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        
        self.subface_38 = self.setupsubfaces("red", "cubie_38", self.gridLayout_5, 0, 0)
        self.subface_37 = self.setupsubfaces("red", "cubie_37", self.gridLayout_5, 1, 0)
        self.subface_36 = self.setupsubfaces("red", "cubie_36", self.gridLayout_5, 2, 0)
        self.subface_41 = self.setupsubfaces("red", "cubie_41", self.gridLayout_5, 0, 1)
        self.subface_40 = self.setupsubfaces("red", "cubie_40", self.gridLayout_5, 1, 1)
        self.subface_39 = self.setupsubfaces("red", "cubie_39", self.gridLayout_5, 2, 1)
        self.subface_44 = self.setupsubfaces("red", "cubie_44", self.gridLayout_5, 0, 2)
        self.subface_43 = self.setupsubfaces("red", "cubie_43", self.gridLayout_5, 1, 2)
        self.subface_42 = self.setupsubfaces("red", "cubie_42", self.gridLayout_5, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout_5, 1, 3, 1, 1)
        
        ### Right Face - Green
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setSpacing(0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.subface_29 = self.setupsubfaces("green", "cubie_29", self.gridLayout_4, 0, 0)
        self.subface_28 = self.setupsubfaces("green", "cubie_28", self.gridLayout_4, 1, 0)
        self.subface_27 = self.setupsubfaces("green", "cubie_27", self.gridLayout_4, 2, 0)
        self.subface_32 = self.setupsubfaces("green", "cubie_32", self.gridLayout_4, 0, 1)
        self.subface_31 = self.setupsubfaces("green", "cubie_31", self.gridLayout_4, 1, 1)
        self.subface_30 = self.setupsubfaces("green", "cubie_30", self.gridLayout_4, 2, 1)
        self.subface_35 = self.setupsubfaces("green", "cubie_35", self.gridLayout_4, 0, 2)
        self.subface_34 = self.setupsubfaces("green", "cubie_34", self.gridLayout_4, 1, 2)
        self.subface_33 = self.setupsubfaces("green", "cubie_33", self.gridLayout_4, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout_4, 1, 2, 1, 1)
        
        ### Down Face - Yellow
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setSpacing(0)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.subface_11 = self.setupsubfaces("yellow", "cubie_11", self.gridLayout_6, 0, 0)
        self.subface_10 = self.setupsubfaces("yellow", "cubie_10", self.gridLayout_6, 1, 0)
        self.subface_9 = self.setupsubfaces("yellow", "cubie_9", self.gridLayout_6, 2, 0)
        self.subface_14 = self.setupsubfaces("yellow", "cubie_14", self.gridLayout_6, 0, 1)
        self.subface_13 = self.setupsubfaces("yellow", "cubie_13", self.gridLayout_6, 1, 1)
        self.subface_12 = self.setupsubfaces("yellow", "cubie_12", self.gridLayout_6, 2, 1)
        self.subface_17 = self.setupsubfaces("yellow", "cubie_17", self.gridLayout_6, 0, 2)
        self.subface_16 = self.setupsubfaces("yellow", "cubie_16", self.gridLayout_6, 1, 2)
        self.subface_15 = self.setupsubfaces("yellow", "cubie_15", self.gridLayout_6, 2, 2)
        self.gridLayout_7.addLayout(self.gridLayout_6, 2, 1, 1, 1)
        
        
        self.horizontalLayout.addLayout(self.gridLayout_7)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem8)
        self.verticalLayout_11 = QtWidgets.QVBoxLayout()
        self.verticalLayout_11.setObjectName("verticalLayout_11")
        spacerItem9 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem9)
        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setObjectName("formLayout")
        
        
        self.btn_up = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_up.sizePolicy().hasHeightForWidth())
        self.btn_up.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_up.setFont(font)
        self.btn_up.setStyleSheet("background-color:white;\n"
"")
        self.btn_up.setObjectName("btn_up")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.btn_up)
        self.btn_up_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_up_reverse.sizePolicy().hasHeightForWidth())
        self.btn_up_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_up_reverse.setFont(font)
        self.btn_up_reverse.setStyleSheet("background-color:white;\n"
"")
        self.btn_up_reverse.setObjectName("btn_up_reverse")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.btn_up_reverse)
        self.btn_left = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_left.setFont(font)
        self.btn_left.setStyleSheet("background-color:blue;\n"
"")
        self.btn_left.setObjectName("btn_left")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.btn_left)
        self.btn_left_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_left_reverse.sizePolicy().hasHeightForWidth())
        self.btn_left_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_left_reverse.setFont(font)
        self.btn_left_reverse.setStyleSheet("background-color:blue;\n"
"")
        self.btn_left_reverse.setObjectName("btn_left_reverse")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.btn_left_reverse)
        self.btn_front = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_front.setFont(font)
        self.btn_front.setStyleSheet("background-color:orange;\n"
"")
        self.btn_front.setObjectName("btn_front")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.btn_front)
        self.btn_front_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_front_reverse.sizePolicy().hasHeightForWidth())
        self.btn_front_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_front_reverse.setFont(font)
        self.btn_front_reverse.setStyleSheet("background-color:orange;\n"
"")
        self.btn_front_reverse.setObjectName("btn_front_reverse")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.btn_front_reverse)
        self.btn_right = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_right.setFont(font)
        self.btn_right.setStyleSheet("background-color:green;\n"
"")
        self.btn_right.setObjectName("btn_right")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.btn_right)
        self.btn_right_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_right_reverse.sizePolicy().hasHeightForWidth())
        self.btn_right_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_right_reverse.setFont(font)
        self.btn_right_reverse.setStyleSheet("background-color:green;\n"
"")
        self.btn_right_reverse.setObjectName("btn_right_reverse")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.btn_right_reverse)
        self.btn_back = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_back.setFont(font)
        self.btn_back.setStyleSheet("background-color:red;\n"
"")
        self.btn_back.setObjectName("btn_back")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.btn_back)
        self.btn_back_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_back_reverse.sizePolicy().hasHeightForWidth())
        self.btn_back_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_back_reverse.setFont(font)
        self.btn_back_reverse.setStyleSheet("background-color:red;\n"
"")
        self.btn_back_reverse.setObjectName("btn_back_reverse")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.btn_back_reverse)
        self.btn_down = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_down.setFont(font)
        self.btn_down.setStyleSheet("background-color:yellow;\n"
"")
        self.btn_down.setObjectName("btn_down")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.btn_down)
        self.btn_down_reverse = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_down_reverse.sizePolicy().hasHeightForWidth())
        self.btn_down_reverse.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_down_reverse.setFont(font)
        self.btn_down_reverse.setStyleSheet("background-color:yellow;\n"
"")
        self.btn_down_reverse.setObjectName("btn_down_reverse")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.btn_down_reverse)
        
        ### Reset
        self.verticalLayout_11.addLayout(self.formLayout)
        self.horizontalLayout_26 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_26.setObjectName("horizontalLayout_26")
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_26.addItem(spacerItem10)
        self.btn_reset = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_reset.setFont(font)
        self.btn_reset.setStyleSheet("background-color:black;color:white;\n"
"")
        self.btn_reset.setObjectName("btn_reset")
        self.horizontalLayout_26.addWidget(self.btn_reset)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_26.addItem(spacerItem11)
        self.verticalLayout_11.addLayout(self.horizontalLayout_26)
        spacerItem12 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_11.addItem(spacerItem12)
        self.verticalLayout_11.setStretch(0, 1)
        self.verticalLayout_11.setStretch(1, 10)
        self.verticalLayout_11.setStretch(2, 2)
        self.verticalLayout_11.setStretch(3, 2)
        self.horizontalLayout.addLayout(self.verticalLayout_11)
        spacerItem13 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem13)
        self.verticalLayout.addLayout(self.horizontalLayout)
        
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem14 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem14)
        self.line_actions = QtWidgets.QLineEdit(self.centralwidget)
        self.line_actions.setEnabled(False)
        font = QtGui.QFont()
        self.line_actions.setStyleSheet("color:black;")
        font.setPointSize(14)
        self.line_actions.setFont(font)
        self.line_actions.setAlignment(QtCore.Qt.AlignCenter)
        self.line_actions.setObjectName("line_actions")
        self.horizontalLayout_2.addWidget(self.line_actions)
        
        spacerItem15 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem15)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 18)
        self.horizontalLayout_2.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        
        
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem30 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem30)
        self.sol = QtWidgets.QLineEdit(self.centralwidget)
        self.sol.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(14)
        
        # set color of the font to be black
        self.sol.setStyleSheet("color:black;")
        self.sol.setFont(font)
        self.sol.setAlignment(QtCore.Qt.AlignCenter)
        self.sol.setObjectName("sol")
        self.horizontalLayout_3.addWidget(self.sol)
        
        spacerItem31 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem31)
        self.horizontalLayout_3.setStretch(0, 1)
        self.horizontalLayout_3.setStretch(1, 18)
        self.horizontalLayout_3.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        spacerItem32 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem32)
        self.sol_status = QtWidgets.QLineEdit(self.centralwidget)
        self.sol_status.setEnabled(False)
        font = QtGui.QFont()
        font.setPointSize(14)
        
        # set color of the font to be black
        self.sol_status.setStyleSheet("color:black;")
        self.sol_status.setFont(font)
        self.sol_status.setAlignment(QtCore.Qt.AlignCenter)
        self.sol_status.setObjectName("sol_status")
        self.horizontalLayout_4.addWidget(self.sol_status)

        spacerItem33 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_4.addItem(spacerItem33)
        self.horizontalLayout_4.setStretch(0, 1)
        self.horizontalLayout_4.setStretch(1, 18)
        self.horizontalLayout_4.setStretch(2, 1)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
                
        spacerItem16 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem16)

        spacerItem28 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem28)
        
        """
        start here
        """
        
        # Create the main horizontal layout for the lower half
        self.lowerHorizontalLayout = QtWidgets.QHBoxLayout()

        # Left vertical layout which will take up 3/5 of the space
        self.scramble_section = QtWidgets.QVBoxLayout()
        self.scramble_buttons = QtWidgets.QVBoxLayout()
   
        self.randomize_option = QtWidgets.QHBoxLayout()        
        self.line_scramble_depth = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_depth.setFont(font)
        self.line_scramble_depth.setMaxLength(3)
        self.line_scramble_depth.setValidator(QIntValidator())
        self.line_scramble_depth.setMaximumWidth(150)
        self.line_scramble_depth.setObjectName("line_scramble_depth")
        self.randomize_option.addWidget(self.line_scramble_depth)    
        
        self.config_options = QtWidgets.QVBoxLayout()
        self.max_scramble_string = 150
        self.line_scramble_string_U = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_U.setFont(font)
        self.line_scramble_string_U.setMaxLength(9)
        self.line_scramble_string_U.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_U.setObjectName("line_scramble_string_U")
        self.config_options.addWidget(self.line_scramble_string_U)
        
        # repeat for D, L, R, B, F
        self.line_scramble_string_D = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_D.setFont(font)
        self.line_scramble_string_D.setMaxLength(9)
        self.line_scramble_string_D.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_D.setObjectName("line_scramble_string_D")
        self.config_options.addWidget(self.line_scramble_string_D)
        
        self.line_scramble_string_L = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_L.setFont(font)
        self.line_scramble_string_L.setMaxLength(9)
        self.line_scramble_string_L.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_L.setObjectName("line_scramble_string_L")
        self.config_options.addWidget(self.line_scramble_string_L)

        self.line_scramble_string_R = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_R.setFont(font)
        self.line_scramble_string_R.setMaxLength(9)
        self.line_scramble_string_R.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_R.setObjectName("line_scramble_string_R")
        self.config_options.addWidget(self.line_scramble_string_R)
        
        self.line_scramble_string_B = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_B.setFont(font)
        self.line_scramble_string_B.setMaxLength(9)
        self.line_scramble_string_B.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_B.setObjectName("line_scramble_string_B")
        self.config_options.addWidget(self.line_scramble_string_B)
        
        self.line_scramble_string_F = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.line_scramble_string_F.setFont(font)
        self.line_scramble_string_F.setMaxLength(9)
        self.line_scramble_string_F.setMaximumWidth(self.max_scramble_string)
        self.line_scramble_string_F.setObjectName("line_scramble_string_F")
        self.config_options.addWidget(self.line_scramble_string_F)
    
         
        self.btn_scramble = QtWidgets.QPushButton(self.centralwidget)
        self.btn_scramble.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_scramble.setFont(font)
        self.btn_scramble.setObjectName("btn_scramble")
        
        self.btn_config = QtWidgets.QPushButton(self.centralwidget)
        self.btn_config.setMinimumSize(QtCore.QSize(100, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_config.setFont(font)
        self.btn_config.setObjectName("btn_config")
        
        
        self.spacerItem100 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.scramble_section.addLayout(self.randomize_option, 1)       
        self.scramble_section.addItem(self.spacerItem100)
        self.scramble_section.addLayout(self.config_options, 1)
        self.spacerItem101 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.scramble_section.setStretch(1, 1)
        self.scramble_section.setStretch(2, 20)
        self.scramble_section.setStretch(3, 1)
        
        self.scramble_buttons.addWidget(self.btn_scramble)
        self.scramble_buttons.addItem(self.spacerItem101)
        self.scramble_buttons.addWidget(self.btn_config)
        
        self.scramble_buttons.setStretch(0, 1)
        self.scramble_buttons.setStretch(1, 20)
        self.scramble_buttons.setStretch(2, 1)
        
        self.spacerItem102 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.lowerHorizontalLayout.addItem(self.spacerItem102)
        # Add left layout with stretch factor 3
        self.lowerHorizontalLayout.addLayout(self.scramble_section)
        self.lowerHorizontalLayout.addLayout(self.scramble_buttons)
        self.spacerItem103 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.lowerHorizontalLayout.addItem(self.spacerItem103)
        
        self.rightVerticalLayout = QtWidgets.QVBoxLayout()        

        ## EBWA* options
        self.astarLayout = QtWidgets.QHBoxLayout()
        self.astar_option = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.astar_option.setFont(font)
        self.astar_option.setChecked(True)
        self.astar_option.setObjectName("astar_option")
        self.astarLayout.addWidget(self.astar_option)
        
        # Scalar factor for A* algorithm
        self.astar_scalar_factor = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        
        self.astar_scalar_factor.setFont(font)
        self.astar_scalar_factor.setMaxLength(5)
        self.astar_scalar_factor.setObjectName("astar_scalar_factor")
        self.astarLayout.addWidget(self.astar_scalar_factor)
        
        # Batch size for A* algorithm
        self.astar_batch_size = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.astar_batch_size.setFont(font)
        self.astar_batch_size.setMaxLength(5)
        self.astar_batch_size.setObjectName("astar_batch_size")
        self.astarLayout.addWidget(self.astar_batch_size)
        self.rightVerticalLayout.addLayout(self.astarLayout)
        
        ### EAWA* options
        self.eawastarLayout = QtWidgets.QHBoxLayout()
        self.eawastar_option = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eawastar_option.setFont(font)
        self.eawastar_option.setObjectName("eawastar_option")
        self.eawastarLayout.addWidget(self.eawastar_option)
        
        # Scalar factor for EAWA* algorithm
        self.eawastar_scalar_factor = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eawastar_scalar_factor.setFont(font)
        self.eawastar_scalar_factor.setMaxLength(5)
        self.eawastar_scalar_factor.setObjectName("eawastar_scalar_factor")
        self.eawastarLayout.addWidget(self.eawastar_scalar_factor)
        
        # Batch size for EAW A* algorithm
        self.eawastar_batch_size = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eawastar_batch_size.setFont(font)
        self.eawastar_batch_size.setMaxLength(5)
        self.eawastar_batch_size.setObjectName("eawastar_batch_size")
        self.eawastarLayout.addWidget(self.eawastar_batch_size)
        
        self.eawastar_time_limit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.eawastar_time_limit.setFont(font)
        self.eawastar_time_limit.setMaxLength(5)
        self.eawastar_time_limit.setObjectName("eawastar_time_limit")
        self.eawastarLayout.addWidget(self.eawastar_time_limit)
        self.rightVerticalLayout.addLayout(self.eawastarLayout)

        ### EBS options
        self.beam_searchLayout = QtWidgets.QHBoxLayout()
        self.beam_search_option = QtWidgets.QRadioButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.beam_search_option.setFont(font)
        self.beam_search_option.setObjectName("beam_search_option")
        self.beam_searchLayout.addWidget(self.beam_search_option)
                
        # Beam width for Beam Search algorithm
        self.beam_width_options = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.beam_width_options.setFont(font)
        self.beam_width_options.setMaxLength(5)
        self.beam_width_options.setObjectName("beam_width_options")
        self.beam_searchLayout.addWidget(self.beam_width_options)
        self.rightVerticalLayout.addLayout(self.beam_searchLayout)
        
        # make btn_solve LEFT ALIGNED
        self.btn_solve = QtWidgets.QPushButton(self.centralwidget)
        self.btn_solve.setMinimumSize(QtCore.QSize(150, 0))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.btn_solve.setFont(font)
        self.btn_solve.setObjectName("btn_solve")
        self.rightVerticalLayout.addWidget(self.btn_solve)
        
        self.spacerItem103 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)

        self.lowerHorizontalLayout.addLayout(self.rightVerticalLayout)
        self.lowerHorizontalLayout.addItem(self.spacerItem103)
        
        """
        
        [ ] [Scramble Section] [Scramble Button] [ ] [Algorithms] [ ]
        
        """
        self.lowerHorizontalLayout.setStretch(0, 1)
        self.lowerHorizontalLayout.setStretch(1, 6.5)
        self.lowerHorizontalLayout.setStretch(2, 4)
        self.lowerHorizontalLayout.setStretch(3, 0.5)
        self.lowerHorizontalLayout.setStretch(4, 8)
        self.lowerHorizontalLayout.setStretch(5, 1)
        

        # Add the lower horizontal layout to the main vertical layout
        self.verticalLayout.addLayout(self.lowerHorizontalLayout)
                
        self.verticalLayout_2.addLayout(self.verticalLayout)
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)



        self.VALUE_COLOR = ["white", "yellow", "blue", "green", "red", "orange"]
        
        self.color_to_face = {"W": "U", "Y": "D", "B": "L", "G": "R", "R": "B", "O": "F"}

        self.cubies = [ self.subface_0, self.subface_1, self.subface_2, self.subface_3, self.subface_4, self.subface_5, self.subface_6, self.subface_7, self.subface_8,
                        self.subface_9, self.subface_10, self.subface_11, self.subface_12, self.subface_13, self.subface_14, self.subface_15, self.subface_16, self.subface_17,
                        self.subface_18, self.subface_19, self.subface_20, self.subface_21, self.subface_22, self.subface_23, self.subface_24, self.subface_25, self.subface_26,
                        self.subface_27, self.subface_28, self.subface_29, self.subface_30, self.subface_31, self.subface_32, self.subface_33, self.subface_34, self.subface_35,
                        self.subface_36, self.subface_37, self.subface_38, self.subface_39, self.subface_40, self.subface_41, self.subface_42, self.subface_43, self.subface_44,
                        self.subface_45, self.subface_46, self.subface_47, self.subface_48, self.subface_49, self.subface_50, self.subface_51, self.subface_52, self.subface_53]
                
        self.cube = Cube()

        self.line_scramble_depth.setValidator(QIntValidator())
        self.btn_scramble.clicked.connect(self.randomize)
        self.btn_reset.clicked.connect(self.reset)
        self.btn_solve.clicked.connect(self.solve)
        self.btn_config.clicked.connect(self.from_color)

        self.manual_buttons = []
        self.btn_up.clicked.connect(lambda: self.move_list([Move.U1]))
        self.manual_buttons.append(self.btn_up)
        self.btn_up_reverse.clicked.connect(lambda: self.move_list([Move.U3]))
        self.manual_buttons.append(self.btn_up_reverse)
        self.btn_left.clicked.connect(lambda: self.move_list([Move.L1]))
        self.manual_buttons.append(self.btn_left)
        self.btn_left_reverse.clicked.connect(lambda: self.move_list([Move.L3]))
        self.manual_buttons.append(self.btn_left_reverse)
        self.btn_front.clicked.connect(lambda: self.move_list([Move.F1]))
        self.manual_buttons.append(self.btn_front)
        self.btn_front_reverse.clicked.connect(lambda: self.move_list([Move.F3]))
        self.manual_buttons.append(self.btn_front_reverse)
        self.btn_right.clicked.connect(lambda: self.move_list([Move.R1]))
        self.manual_buttons.append(self.btn_right)
        self.btn_right_reverse.clicked.connect(lambda: self.move_list([Move.R3]))
        self.manual_buttons.append(self.btn_right_reverse)
        self.btn_back.clicked.connect(lambda: self.move_list([Move.B1]))
        self.manual_buttons.append(self.btn_back)
        self.btn_back_reverse.clicked.connect(lambda: self.move_list([Move.B3]))
        self.manual_buttons.append(self.btn_back_reverse)
        self.btn_down.clicked.connect(lambda: self.move_list([Move.D1]))
        self.manual_buttons.append(self.btn_down)
        self.btn_down_reverse.clicked.connect(lambda: self.move_list([Move.D3]))
        self.manual_buttons.append(self.btn_down_reverse)
        # self.btn_settings.clicked.connect(self.open_settings)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    
        
    def setupsubfaces(self, color, subface_name, face, row, col):
        self.subface_name = QtWidgets.QLineEdit(self.centralwidget)
        self.subface_name.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.subface_name.sizePolicy().hasHeightForWidth())
        
        self.subface_name.setSizePolicy(sizePolicy)
        self.subface_name.setMinimumSize(QtCore.QSize(50, 50))
        self.subface_name.setMaximumSize(QtCore.QSize(50, 50))
        self.subface_name.setStyleSheet(f"background-color:{color};\n"
"")
        self.subface_name.setObjectName(f"{subface_name}")
        face.addWidget(self.subface_name, row, col, 1, 1)
        return self.subface_name
            
    def render(self):
        if self.i == len(self.states):
            self.timer.stop()
            self.btn_reset.setEnabled(True)
            self.btn_scramble.setEnabled(True)
            self.btn_solve.setEnabled(True)
            [button.setEnabled(True) for button in self.manual_buttons]
        else:
            for idx, value in enumerate(self.states[self.i]):
                self.cubies[idx].setStyleSheet(f"""background-color:{self.VALUE_COLOR[value]};\n""")
            self.i+=1

    def solve(self):
         # Update the action line to inform the user that solving has started
        self.sol.setText("Start solving...")
        QApplication.processEvents()  # Ensure the GUI updates to show the message
        
        self.btn_reset.setEnabled(False)
        self.btn_scramble.setEnabled(False)
        self.btn_solve.setEnabled(False)
        [button.setEnabled(False) for button in self.manual_buttons]
        
        astar_option = self.astar_option.isChecked()
        beam_search_option = self.beam_search_option.isChecked()
        eawastar_option = self.eawastar_option.isChecked()
        
        if not astar_option and not beam_search_option and not eawastar_option:
            raise Exception("No search algorithm selected.")
        
        if (astar_option and beam_search_option) or (astar_option and eawastar_option) or (beam_search_option and eawastar_option):
            raise Exception("More than one search algorithm selected.")
        
        if self.cube.is_solved():
            self.line_actions.clear()
            self.line_actions.setVisible(True)
            self.line_actions.setText("The cube is already solved.")
            self.sol.clear()
            self.sol_status.clear()
            self.btn_reset.setEnabled(True)
            self.btn_scramble.setEnabled(True)
            self.btn_solve.setEnabled(True)
            [button.setEnabled(True) for button in self.manual_buttons]
            return
        
        result = None   
        if astar_option:
            scalar_factor = self.astar_scalar_factor.text()
            batch_size = self.astar_batch_size.text()
            if scalar_factor == "":
                scalar_factor = 3.0
            else:
                scalar_factor = float(scalar_factor)
            
            if batch_size == "":
                batch_size = 1000
            else:
                batch_size = int(batch_size)
                
            self.astar_batch_size.clear()
            self.astar_scalar_factor.clear()
            mwas = MWAStar(self.cube, scalar_factor, batch_size)
            result = mwas.search()
            
        elif beam_search_option:
            beam_width = self.beam_width_options.text()
            if beam_width == "":
                beam_width = 1000
            else:
                beam_width = int(beam_width)
                
            self.beam_width_options.clear()
            
            mbs = MBS(self.cube, beam_width)
            result = mbs.search()
            
        elif eawastar_option:
            scalar_factor = self.eawastar_scalar_factor.text()
            batch_size = self.eawastar_batch_size.text()
            time_limit = self.eawastar_time_limit.text()
            
            if scalar_factor == "":
                scalar_factor = 3.0
            else:
                scalar_factor = float(scalar_factor)
            
            if batch_size == "":
                batch_size = 1000
            else:
                batch_size = int(batch_size)
                
            if time_limit == "":
                time_limit = 60
            else:
                time_limit = int(time_limit)
                
            self.eawastar_batch_size.clear()
            self.eawastar_scalar_factor.clear()
            self.eawastar_time_limit.clear()
            
            mwas = MAWAStar(self.cube, scalar_factor, batch_size, time_limit)
            result = mwas.search()
    
        if result["success"] and "error" in result:
            self.move_list(result["solutions"])
            self.sol.setText(f"Solution: {self.cube.move_to_string(result['solutions'])}")
            self.sol_status.setText(f"Solved in {result['length']} steps, {result['time_taken']:.2f}s, {scientific_notation(result['num_nodes'])} states explored, {result['num_nodes'] / result['time_taken']:.2f} states/s, error: {result['error']:.2f}.")
        
        elif result["success"] and "error" not in result:
            self.move_list(result["solutions"])
            self.sol.setText(f"Solution: {self.cube.move_to_string(result['solutions'])}")
            self.sol_status.setText(f"Solved in {result['length']} steps, {result['time_taken']:.2f}s, {scientific_notation(result['num_nodes'])} states explored, {result['num_nodes'] / result['time_taken']:.2f} states/s.")
        else:
            self.sol.setText("The cube couldn't be solved. You can increase the time, batch/beam size, or scalar factor.")
        self.btn_reset.setEnabled(True)
        self.btn_scramble.setEnabled(True)
        self.btn_solve.setEnabled(True)
        [button.setEnabled(True) for button in self.manual_buttons]

    def reset(self):
        self.line_actions.clear()
        self.sol.clear()
        self.sol_status.clear()
        self.line_scramble_depth.clear()
        self.line_scramble_string_U.clear()
        self.line_scramble_string_D.clear()
        self.line_scramble_string_L.clear()
        self.line_scramble_string_R.clear()
        self.line_scramble_string_B.clear()
        self.line_scramble_string_F.clear()
        self.astar_batch_size.clear()
        self.astar_scalar_factor.clear()
        self.eawastar_batch_size.clear()
        self.eawastar_scalar_factor.clear()
        self.eawastar_time_limit.clear()
        self.beam_width_options.clear()
        
        self.btn_solve.setEnabled(True)
        self.cube.reset()
                
        for idx, value in enumerate(self.cube.state):
            self.cubies[idx].setStyleSheet(f"""background-color:{self.VALUE_COLOR[value]};\n""")

        self.line_actions.setVisible(True)
        self.line_actions.setText("Reset.")
        
    def move_list(self, move_list):
        # render steps by steps
        self.states = []
        for move in move_list:
            self.cube.move(move)
            self.states.append(self.cube.state.copy())
            
        self.i = 0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.render)
        self.timer.start(self.actions_delay)
        
    def from_color(self):
        # clear the actions line and depth line sol sol_status
        self.line_actions.clear()
        self.sol.clear()
        self.sol_status.clear()
        self.line_scramble_depth.clear()
        
        U_face = self.line_scramble_string_U.text()
        D_face = self.line_scramble_string_D.text()
        L_face = self.line_scramble_string_L.text()
        R_face = self.line_scramble_string_R.text()
        B_face = self.line_scramble_string_B.text()
        F_face = self.line_scramble_string_F.text()
            
        scramble_string = U_face + D_face + L_face + R_face + B_face + F_face
        scramble_string = "".join([self.color_to_face[color] for color in scramble_string])

        if len(scramble_string) != 54:
            self.line_actions.setVisible(True)
            self.line_actions.setText("Please enter 9 colors for each face.")
            return
        
        validation_result = scramble_string
        if validation_result["success"] == False:
            self.line_actions.setVisible(True)
            self.line_actions.setText(validation_result["error"])
            return
                
        self.cube.from_string(scramble_string)
        for idx, value in enumerate(self.cube.state):
            self.cubies[idx].setStyleSheet(f"""background-color:{self.VALUE_COLOR[value]};\n""")
        self.line_actions.setVisible(True)
        self.line_actions.setText("Finished constructing cube from colors")
        
    def randomize(self):
        depth = self.line_scramble_depth.text()
        self.line_scramble_depth.clear()
            
        if depth == "":
            depth = 15
        else:
            depth = int(depth)
            
        move_str, move_list = self.cube.random_moves(depth)
        self.move_list(move_list)
                
        self.btn_reset.setEnabled(False)
        self.btn_scramble.setEnabled(False)
        self.btn_solve.setEnabled(False)
        [button.setEnabled(False) for button in self.manual_buttons]
        self.line_actions.setVisible(True)
        self.line_actions.setText("Scramble: " + move_str)
        self.sol.clear()
        self.sol_status.clear()
            
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Cube Solver"))
        self.btn_up.setText(_translate("MainWindow", "U"))
        self.btn_up_reverse.setText(_translate("MainWindow", "U\'"))
        self.btn_left.setText(_translate("MainWindow", "L"))
        self.btn_left_reverse.setText(_translate("MainWindow", "L\'"))
        self.btn_front.setText(_translate("MainWindow", "F"))
        self.btn_front_reverse.setText(_translate("MainWindow", "F\'"))
        self.btn_right.setText(_translate("MainWindow", "R"))
        self.btn_right_reverse.setText(_translate("MainWindow", "R\'"))
        self.btn_back.setText(_translate("MainWindow", "B"))
        self.btn_back_reverse.setText(_translate("MainWindow", "B\'"))
        self.btn_down.setText(_translate("MainWindow", "D"))
        self.btn_down_reverse.setText(_translate("MainWindow", "D\'"))
        self.btn_reset.setText(_translate("MainWindow", "RESET"))
        self.btn_solve.setText(_translate("MainWindow", "Solve with Selected Algorithm"))
        self.astar_option.setText(_translate("MainWindow", "EBWA* Search"))
        self.beam_search_option.setText(_translate("MainWindow", "EBS"))
        self.eawastar_option.setText(_translate("MainWindow", "EAWA* Search"))
        
        self.astar_scalar_factor.setPlaceholderText(_translate("MainWindow", "Scalar factor : 3.0"))
        self.astar_batch_size.setPlaceholderText(_translate("MainWindow", "Batch size : 1000"))
        self.eawastar_scalar_factor.setPlaceholderText(_translate("MainWindow", "Scalar factor : 3.0"))
        self.eawastar_batch_size.setPlaceholderText(_translate("MainWindow", "Batch size : 1000"))
        self.eawastar_time_limit.setPlaceholderText(_translate("MainWindow", "Time limit : 60s"))
        self.beam_width_options.setPlaceholderText(_translate("MainWindow", "Beam width : 1000"))
        
        
        self.btn_scramble.setText(_translate("MainWindow", "Randomly Scramble with Depth"))
        self.line_scramble_depth.setPlaceholderText(_translate("MainWindow", "Depth : 15"))
        self.line_scramble_string_U.setPlaceholderText(_translate("MainWindow", "Colors on U Face"))
        self.line_scramble_string_D.setPlaceholderText(_translate("MainWindow", "Colors on D Face"))
        self.line_scramble_string_L.setPlaceholderText(_translate("MainWindow", "Colors on L Face"))
        self.line_scramble_string_R.setPlaceholderText(_translate("MainWindow", "Colors on R Face"))
        self.line_scramble_string_B.setPlaceholderText(_translate("MainWindow", "Colors on B Face"))
        self.line_scramble_string_F.setPlaceholderText(_translate("MainWindow", "Colors on F Face"))
        self.btn_config.setText(_translate("MainWindow", "Construct Cube from Colors"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())