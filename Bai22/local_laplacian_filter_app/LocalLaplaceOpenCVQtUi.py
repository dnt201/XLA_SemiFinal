# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LocalLaplaceOpenCVQt.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_LocalLaplaceOpenCVQtClass(object):
    def setupUi(self, LocalLaplaceOpenCVQtClass):
        LocalLaplaceOpenCVQtClass.setObjectName("LocalLaplaceOpenCVQtClass")
        LocalLaplaceOpenCVQtClass.showFullScreen()

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(LocalLaplaceOpenCVQtClass.sizePolicy().hasHeightForWidth())
        LocalLaplaceOpenCVQtClass.setSizePolicy(sizePolicy)
        LocalLaplaceOpenCVQtClass.setStyleSheet("")
        self.centralWidget = QtWidgets.QWidget(LocalLaplaceOpenCVQtClass)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralWidget.sizePolicy().hasHeightForWidth())
        self.centralWidget.setSizePolicy(sizePolicy)
        self.centralWidget.setMinimumSize(QtCore.QSize(0, 0))
        self.centralWidget.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.centralWidget.setObjectName("centralWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralWidget)
        self.horizontalLayout.setContentsMargins(11, 11, 11, 11)
        self.horizontalLayout.setSpacing(6)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBoxParametry = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBoxParametry.setMinimumSize(QtCore.QSize(200, 0))
        self.groupBoxParametry.setMaximumSize(QtCore.QSize(200, 16777215))
        self.groupBoxParametry.setObjectName("groupBoxParametry")
        self.formLayout_5 = QtWidgets.QFormLayout(self.groupBoxParametry)
        self.formLayout_5.setContentsMargins(11, 11, 11, 11)
        self.formLayout_5.setSpacing(6)
        self.formLayout_5.setObjectName("formLayout_5")
        self.groupBoxAlpha = QtWidgets.QGroupBox(self.groupBoxParametry)
        self.groupBoxAlpha.setMinimumSize(QtCore.QSize(180, 149))
        self.groupBoxAlpha.setMaximumSize(QtCore.QSize(180, 149))
        self.groupBoxAlpha.setTitle("")
        self.groupBoxAlpha.setObjectName("groupBoxAlpha")
        self.formLayout = QtWidgets.QFormLayout(self.groupBoxAlpha)
        self.formLayout.setContentsMargins(11, 11, 11, 11)
        self.formLayout.setSpacing(6)
        self.formLayout.setObjectName("formLayout")
        self.labelAlpha = QtWidgets.QLabel(self.groupBoxAlpha)
        self.labelAlpha.setObjectName("labelAlpha")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelAlpha)
        self.lcdNumberAlpha = QtWidgets.QLCDNumber(self.groupBoxAlpha)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lcdNumberAlpha.setFont(font)
        self.lcdNumberAlpha.setStyleSheet("background-color:black;")
        self.lcdNumberAlpha.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcdNumberAlpha.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lcdNumberAlpha.setLineWidth(1)
        self.lcdNumberAlpha.setMidLineWidth(0)
        self.lcdNumberAlpha.setSmallDecimalPoint(True)
        self.lcdNumberAlpha.setDigitCount(3)
        self.lcdNumberAlpha.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.lcdNumberAlpha.setProperty("value", 0.0)
        self.lcdNumberAlpha.setProperty("intValue", 0)
        self.lcdNumberAlpha.setObjectName("lcdNumberAlpha")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lcdNumberAlpha)
        self.dialAlpha = QtWidgets.QDial(self.groupBoxAlpha)
        self.dialAlpha.setMouseTracking(True)
        self.dialAlpha.setObjectName("dialAlpha")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dialAlpha)
        self.formLayout_5.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.groupBoxAlpha)
        self.groupBoxBeta = QtWidgets.QGroupBox(self.groupBoxParametry)
        self.groupBoxBeta.setMinimumSize(QtCore.QSize(180, 149))
        self.groupBoxBeta.setMaximumSize(QtCore.QSize(180, 149))
        self.groupBoxBeta.setTitle("")
        self.groupBoxBeta.setObjectName("groupBoxBeta")
        self.formLayout_2 = QtWidgets.QFormLayout(self.groupBoxBeta)
        self.formLayout_2.setContentsMargins(11, 11, 11, 11)
        self.formLayout_2.setSpacing(6)
        self.formLayout_2.setObjectName("formLayout_2")
        self.labelBeta = QtWidgets.QLabel(self.groupBoxBeta)
        self.labelBeta.setObjectName("labelBeta")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelBeta)
        self.lcdNumberBeta = QtWidgets.QLCDNumber(self.groupBoxBeta)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lcdNumberBeta.setFont(font)
        self.lcdNumberBeta.setStyleSheet("background-color:black;")
        self.lcdNumberBeta.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcdNumberBeta.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lcdNumberBeta.setLineWidth(1)
        self.lcdNumberBeta.setMidLineWidth(0)
        self.lcdNumberBeta.setSmallDecimalPoint(True)
        self.lcdNumberBeta.setDigitCount(3)
        self.lcdNumberBeta.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.lcdNumberBeta.setProperty("value", 0.0)
        self.lcdNumberBeta.setProperty("intValue", 0)
        self.lcdNumberBeta.setObjectName("lcdNumberBeta")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lcdNumberBeta)
        self.dialBeta = QtWidgets.QDial(self.groupBoxBeta)
        self.dialBeta.setMouseTracking(True)
        self.dialBeta.setObjectName("dialBeta")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dialBeta)
        self.formLayout_5.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.groupBoxBeta)
        self.groupBoxSigmaR = QtWidgets.QGroupBox(self.groupBoxParametry)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxSigmaR.sizePolicy().hasHeightForWidth())
        self.groupBoxSigmaR.setSizePolicy(sizePolicy)
        self.groupBoxSigmaR.setMinimumSize(QtCore.QSize(180, 149))
        self.groupBoxSigmaR.setMaximumSize(QtCore.QSize(180, 149))
        self.groupBoxSigmaR.setTitle("")
        self.groupBoxSigmaR.setObjectName("groupBoxSigmaR")
        self.formLayout_3 = QtWidgets.QFormLayout(self.groupBoxSigmaR)
        self.formLayout_3.setContentsMargins(11, 11, 11, 11)
        self.formLayout_3.setSpacing(6)
        self.formLayout_3.setObjectName("formLayout_3")
        self.labelSigmaR = QtWidgets.QLabel(self.groupBoxSigmaR)
        self.labelSigmaR.setObjectName("labelSigmaR")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.labelSigmaR)
        self.lcdNumberSigmaR = QtWidgets.QLCDNumber(self.groupBoxSigmaR)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Black")
        font.setPointSize(11)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.lcdNumberSigmaR.setFont(font)
        self.lcdNumberSigmaR.setStyleSheet("background-color:black;")
        self.lcdNumberSigmaR.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.lcdNumberSigmaR.setFrameShadow(QtWidgets.QFrame.Plain)
        self.lcdNumberSigmaR.setLineWidth(1)
        self.lcdNumberSigmaR.setMidLineWidth(0)
        self.lcdNumberSigmaR.setSmallDecimalPoint(True)
        self.lcdNumberSigmaR.setDigitCount(3)
        self.lcdNumberSigmaR.setSegmentStyle(QtWidgets.QLCDNumber.Outline)
        self.lcdNumberSigmaR.setProperty("value", 0.0)
        self.lcdNumberSigmaR.setProperty("intValue", 0)
        self.lcdNumberSigmaR.setObjectName("lcdNumberSigmaR")
        self.formLayout_3.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lcdNumberSigmaR)
        self.dialSigmaR = QtWidgets.QDial(self.groupBoxSigmaR)
        self.dialSigmaR.setMouseTracking(True)
        self.dialSigmaR.setObjectName("dialSigmaR")
        self.formLayout_3.setWidget(1, QtWidgets.QFormLayout.SpanningRole, self.dialSigmaR)
        self.formLayout_5.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.groupBoxSigmaR)
        self.groupBoxGuziki = QtWidgets.QGroupBox(self.groupBoxParametry)
        self.groupBoxGuziki.setMinimumSize(QtCore.QSize(180, 58))
        self.groupBoxGuziki.setMaximumSize(QtCore.QSize(180, 58))
        self.groupBoxGuziki.setTitle("")
        self.groupBoxGuziki.setObjectName("groupBoxGuziki")
        self.formLayout_4 = QtWidgets.QFormLayout(self.groupBoxGuziki)
        self.formLayout_4.setContentsMargins(11, 11, 11, 11)
        self.formLayout_4.setSpacing(6)
        self.formLayout_4.setObjectName("formLayout_4")
        self.pushButtonDefault = QtWidgets.QPushButton(self.groupBoxGuziki)
        self.pushButtonDefault.setObjectName("pushButtonDefault")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.pushButtonDefault)
        self.pushButtonApply = QtWidgets.QPushButton(self.groupBoxGuziki)
        self.pushButtonApply.setObjectName("pushButtonApply")
        self.formLayout_4.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.pushButtonApply)
        self.formLayout_5.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.groupBoxGuziki)
        self.label = QtWidgets.QLabel(self.groupBoxParametry)
        self.label.setObjectName("label")
        self.formLayout_5.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label)
        self.logArea = QtWidgets.QPlainTextEdit(self.groupBoxParametry)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.logArea.sizePolicy().hasHeightForWidth())
        self.logArea.setSizePolicy(sizePolicy)
        self.logArea.setMinimumSize(QtCore.QSize(180, 0))
        self.logArea.setMaximumSize(QtCore.QSize(180, 16777215))
        self.logArea.setStyleSheet("color:white;background-color:black")
        self.logArea.setObjectName("logArea")
        self.formLayout_5.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.logArea)
        self.groupScale = QtWidgets.QGroupBox(self.groupBoxParametry)
        self.groupScale.setObjectName("groupScale")
        self.formLayout_5.setWidget(3, QtWidgets.QFormLayout.SpanningRole, self.groupScale)
        self.groupImageQuality = QtWidgets.QGroupBox(self.groupBoxParametry)
        self.groupImageQuality.setObjectName("groupImageQuality")
        self.formLayout_6 = QtWidgets.QFormLayout(self.groupImageQuality)
        self.formLayout_6.setContentsMargins(11, 11, 11, 11)
        self.formLayout_6.setSpacing(6)
        self.formLayout_6.setObjectName("formLayout_6")
        self.comboBoxImageQuality = QtWidgets.QComboBox(self.groupImageQuality)
        self.comboBoxImageQuality.setObjectName("comboBoxImageQuality")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.comboBoxImageQuality.addItem("")
        self.formLayout_6.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.comboBoxImageQuality)
        self.formLayout_5.setWidget(4, QtWidgets.QFormLayout.SpanningRole, self.groupImageQuality)     
        self.horizontalLayout.addWidget(self.groupBoxParametry)
        self.groupBoxObrazy = QtWidgets.QGroupBox(self.centralWidget)
        self.groupBoxObrazy.setObjectName("groupBoxObrazy")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBoxObrazy)
        self.gridLayout.setContentsMargins(11, 11, 11, 11)
        self.gridLayout.setSpacing(6)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBoxStatystyki = QtWidgets.QGroupBox(self.groupBoxObrazy)
        self.groupBoxStatystyki.setMaximumSize(QtCore.QSize(16777215, 320))
        self.groupBoxStatystyki.setObjectName("groupBoxStatystyki")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBoxStatystyki)
        self.gridLayout_2.setContentsMargins(11, 11, 11, 11)
        self.gridLayout_2.setSpacing(6)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tabWidgetQuality = QtWidgets.QTabWidget(self.groupBoxStatystyki)
        self.tabWidgetQuality.setAutoFillBackground(True)
        self.tabWidgetQuality.setStyleSheet("")
        self.tabWidgetQuality.setObjectName("tabWidgetQuality")
        self.tab = QtWidgets.QWidget()
        self.tab.setAutoFillBackground(True)
        self.tab.setObjectName("tab")
        self.labelTytulBrisque = QtWidgets.QLabel(self.tab)
        self.labelTytulBrisque.setGeometry(QtCore.QRect(10, 10, 101, 16))
        self.labelTytulBrisque.setObjectName("labelTytulBrisque")
        self.labelWartoscBrisque = QtWidgets.QLabel(self.tab)
        self.labelWartoscBrisque.setGeometry(QtCore.QRect(110, 10, 491, 16))
        self.labelWartoscBrisque.setText("")
        self.labelWartoscBrisque.setObjectName("labelWartoscBrisque")
        self.tabWidgetQuality.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setAutoFillBackground(True)
        self.tab_2.setObjectName("tab_2")
        self.labelTytulMSE = QtWidgets.QLabel(self.tab_2)
        self.labelTytulMSE.setGeometry(QtCore.QRect(10, 10, 101, 16))
        self.labelTytulMSE.setObjectName("labelTytulMSE")
        self.labelWartoscMSE = QtWidgets.QLabel(self.tab_2)
        self.labelWartoscMSE.setGeometry(QtCore.QRect(90, 10, 491, 16))
        self.labelWartoscMSE.setText("")
        self.labelWartoscMSE.setObjectName("labelWartoscMSE")
        self.tabWidgetQuality.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setAutoFillBackground(True)
        self.tab_3.setObjectName("tab_3")
        self.labelTytulPSNR = QtWidgets.QLabel(self.tab_3)
        self.labelTytulPSNR.setGeometry(QtCore.QRect(10, 10, 101, 16))
        self.labelTytulPSNR.setObjectName("labelTytulPSNR")
        self.labelWartoscPSNR = QtWidgets.QLabel(self.tab_3)
        self.labelWartoscPSNR.setGeometry(QtCore.QRect(100, 10, 491, 16))
        self.labelWartoscPSNR.setText("")
        self.labelWartoscPSNR.setObjectName("labelWartoscPSNR")
        self.tabWidgetQuality.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setAutoFillBackground(True)
        self.tab_4.setObjectName("tab_4")
        self.labelTytulSSIM = QtWidgets.QLabel(self.tab_4)
        self.labelTytulSSIM.setGeometry(QtCore.QRect(10, 10, 101, 16))
        self.labelTytulSSIM.setObjectName("labelTytulSSIM")
        self.labelWartoscSSIM = QtWidgets.QLabel(self.tab_4)
        self.labelWartoscSSIM.setGeometry(QtCore.QRect(100, 10, 491, 16))
        self.labelWartoscSSIM.setText("")
        self.labelWartoscSSIM.setObjectName("labelWartoscSSIM")
        self.tabWidgetQuality.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setAutoFillBackground(True)
        self.tab_5.setObjectName("tab_5")
        self.labelTytulGMSD = QtWidgets.QLabel(self.tab_5)
        self.labelTytulGMSD.setGeometry(QtCore.QRect(10, 10, 101, 16))
        self.labelTytulGMSD.setObjectName("labelTytulGMSD")
        self.labelWartoscGMSD = QtWidgets.QLabel(self.tab_5)
        self.labelWartoscGMSD.setGeometry(QtCore.QRect(100, 10, 491, 16))
        self.labelWartoscGMSD.setText("")
        self.labelWartoscGMSD.setObjectName("labelWartoscGMSD")
        self.tabWidgetQuality.addTab(self.tab_5, "")
        self.gridLayout_2.addWidget(self.tabWidgetQuality, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBoxStatystyki, 2, 0, 1, 2)
        self.groupBoxObrazWejsciowy = QtWidgets.QGroupBox(self.groupBoxObrazy)
        self.groupBoxObrazWejsciowy.setObjectName("groupBoxObrazWejsciowy")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBoxObrazWejsciowy)
        self.verticalLayout.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.labelObrazWejsciowy = QtWidgets.QLabel(self.groupBoxObrazWejsciowy)
        self.labelObrazWejsciowy.setMaximumSize(QtCore.QSize(800, 1100))
        self.labelObrazWejsciowy.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelObrazWejsciowy.setAutoFillBackground(False)
        self.labelObrazWejsciowy.setStyleSheet("")
        self.labelObrazWejsciowy.setText("")
        self.labelObrazWejsciowy.setAlignment(QtCore.Qt.AlignCenter)
        self.labelObrazWejsciowy.setObjectName("labelObrazWejsciowy")
        self.verticalLayout.addWidget(self.labelObrazWejsciowy)
        self.gridLayout.addWidget(self.groupBoxObrazWejsciowy, 0, 0, 1, 1)
        self.groupBoxObrazWyjsciowy = QtWidgets.QGroupBox(self.groupBoxObrazy)
        self.groupBoxObrazWyjsciowy.setObjectName("groupBoxObrazWyjsciowy")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBoxObrazWyjsciowy)
        self.verticalLayout_2.setContentsMargins(11, 11, 11, 11)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.labelObrazWyjsciowy = QtWidgets.QLabel(self.groupBoxObrazWyjsciowy)
        self.labelObrazWyjsciowy.setMaximumSize(QtCore.QSize(800, 1100))
        self.labelObrazWyjsciowy.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelObrazWyjsciowy.setAutoFillBackground(False)
        self.labelObrazWyjsciowy.setStyleSheet("")
        self.labelObrazWyjsciowy.setText("")
        self.labelObrazWyjsciowy.setAlignment(QtCore.Qt.AlignCenter)
        self.labelObrazWyjsciowy.setObjectName("labelObrazWyjsciowy")
        self.verticalLayout_2.addWidget(self.labelObrazWyjsciowy)
        self.gridLayout.addWidget(self.groupBoxObrazWyjsciowy, 0, 1, 1, 1)
        self.horizontalLayout.addWidget(self.groupBoxObrazy)
        LocalLaplaceOpenCVQtClass.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(LocalLaplaceOpenCVQtClass)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 890, 21))
        self.menuBar.setObjectName("menuBar")
        self.menuPlik = QtWidgets.QMenu(self.menuBar)
        self.menuPlik.setObjectName("menuPlik")
        self.menuOstatnioOtwietane = QtWidgets.QMenu(self.menuPlik)
        self.menuOstatnioOtwietane.setObjectName("menuOstatnioOtwietane")
        self.menuSesje = QtWidgets.QMenu(self.menuBar)
        self.menuSesje.setObjectName("menuSesje")
        LocalLaplaceOpenCVQtClass.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(LocalLaplaceOpenCVQtClass)
        self.mainToolBar.setObjectName("mainToolBar")
        LocalLaplaceOpenCVQtClass.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(LocalLaplaceOpenCVQtClass)
        self.statusBar.setObjectName("statusBar")
        LocalLaplaceOpenCVQtClass.setStatusBar(self.statusBar)
        self.actionOtworz = QtWidgets.QAction(LocalLaplaceOpenCVQtClass)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("Resources/Open_Icon_256.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionOtworz.setIcon(icon)
        self.actionOtworz.setObjectName("actionOtworz")
        self.actionWyjscie = QtWidgets.QAction(LocalLaplaceOpenCVQtClass)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("Resources/Windows_Turn_Off_Icon_256.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionWyjscie.setIcon(icon1)
        self.actionWyjscie.setObjectName("actionWyjscie")
        self.actionZapiszSesje = QtWidgets.QAction(LocalLaplaceOpenCVQtClass)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap("Resources/Save_Icon_256.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionZapiszSesje.setIcon(icon2)
        self.actionZapiszSesje.setObjectName("actionZapiszSesje")
        self.menuOstatnioOtwietane.addSeparator()
        self.menuPlik.addAction(self.actionOtworz)
        self.menuPlik.addAction(self.menuOstatnioOtwietane.menuAction())
        self.menuPlik.addAction(self.actionWyjscie)
        self.menuSesje.addAction(self.actionZapiszSesje)
        self.menuBar.addAction(self.menuPlik.menuAction())
        self.menuBar.addAction(self.menuSesje.menuAction())
        self.mainToolBar.addAction(self.actionOtworz)
        self.mainToolBar.addAction(self.actionZapiszSesje)
        self.mainToolBar.addAction(self.actionWyjscie)

        self.retranslateUi(LocalLaplaceOpenCVQtClass)
        self.tabWidgetQuality.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(LocalLaplaceOpenCVQtClass)

    def retranslateUi(self, LocalLaplaceOpenCVQtClass):
        _translate = QtCore.QCoreApplication.translate
        LocalLaplaceOpenCVQtClass.setWindowTitle(_translate("LocalLaplaceOpenCVQtClass", "LocalLaplaceOpenCVQt"))
        self.groupBoxParametry.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Control Panel"))
        self.labelAlpha.setText(_translate("LocalLaplaceOpenCVQtClass", "Parameter Alpha"))
        self.labelBeta.setText(_translate("LocalLaplaceOpenCVQtClass", "Parameter Beta"))
        self.labelSigmaR.setText(_translate("LocalLaplaceOpenCVQtClass", "Parameter SigmaR"))
        self.groupBoxGuziki.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Action")) # zmiana
        self.pushButtonDefault.setText(_translate("LocalLaplaceOpenCVQtClass", "Default"))
        self.pushButtonApply.setText(_translate("LocalLaplaceOpenCVQtClass", "Apply"))
        self.label.setText(_translate("LocalLaplaceOpenCVQtClass", "Results :"))
        self.groupScale.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Input image scalling"))
        self.groupImageQuality.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Image Quality Assessment"))
        self.comboBoxImageQuality.setItemText(0, _translate("LocalLaplaceOpenCVQtClass", "No quality assessment"))
        self.comboBoxImageQuality.setItemText(1, _translate("LocalLaplaceOpenCVQtClass", "All methods"))
        self.comboBoxImageQuality.setItemText(2, _translate("LocalLaplaceOpenCVQtClass", "BRISQUE"))
        self.comboBoxImageQuality.setItemText(3, _translate("LocalLaplaceOpenCVQtClass", "MSE"))
        self.comboBoxImageQuality.setItemText(4, _translate("LocalLaplaceOpenCVQtClass", "SNR"))
        self.comboBoxImageQuality.setItemText(5, _translate("LocalLaplaceOpenCVQtClass", "SSIS"))
        self.comboBoxImageQuality.setItemText(6, _translate("LocalLaplaceOpenCVQtClass", "GMSD"))
        self.groupBoxObrazy.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Results"))
        self.groupBoxStatystyki.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Quality assessment of output image"))
        self.labelTytulBrisque.setText(_translate("LocalLaplaceOpenCVQtClass", "BRISQUE value : "))
        self.tabWidgetQuality.setTabText(self.tabWidgetQuality.indexOf(self.tab), _translate("LocalLaplaceOpenCVQtClass", "BRISQUE"))
        self.labelTytulMSE.setText(_translate("LocalLaplaceOpenCVQtClass", "MSE value : "))
        self.tabWidgetQuality.setTabText(self.tabWidgetQuality.indexOf(self.tab_2), _translate("LocalLaplaceOpenCVQtClass", "MSE"))
        self.labelTytulPSNR.setText(_translate("LocalLaplaceOpenCVQtClass", "PSNR value : "))
        self.tabWidgetQuality.setTabText(self.tabWidgetQuality.indexOf(self.tab_3), _translate("LocalLaplaceOpenCVQtClass", "PSNR"))
        self.labelTytulSSIM.setText(_translate("LocalLaplaceOpenCVQtClass", "SSIM value : "))
        self.tabWidgetQuality.setTabText(self.tabWidgetQuality.indexOf(self.tab_4), _translate("LocalLaplaceOpenCVQtClass", "SSIM"))
        self.labelTytulGMSD.setText(_translate("LocalLaplaceOpenCVQtClass", "GMSD value : "))
        self.tabWidgetQuality.setTabText(self.tabWidgetQuality.indexOf(self.tab_5), _translate("LocalLaplaceOpenCVQtClass", "GMSD"))
        self.groupBoxObrazWejsciowy.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Input Image"))
        self.groupBoxObrazWyjsciowy.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Output Image"))
        self.menuPlik.setTitle(_translate("LocalLaplaceOpenCVQtClass", "File"))
        self.menuOstatnioOtwietane.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Recently opened"))
        self.menuSesje.setTitle(_translate("LocalLaplaceOpenCVQtClass", "Sessions"))
        self.actionOtworz.setText(_translate("LocalLaplaceOpenCVQtClass", "Open"))
        self.actionOtworz.setShortcut(_translate("LocalLaplaceOpenCVQtClass", "Ctrl+O"))
        self.actionWyjscie.setText(_translate("LocalLaplaceOpenCVQtClass", "Quit"))
        self.actionWyjscie.setShortcut(_translate("LocalLaplaceOpenCVQtClass", "Ctrl+X"))
        self.actionZapiszSesje.setText(_translate("LocalLaplaceOpenCVQtClass", "Save session"))
        self.actionZapiszSesje.setShortcut(_translate("LocalLaplaceOpenCVQtClass", "Ctrl+S"))
