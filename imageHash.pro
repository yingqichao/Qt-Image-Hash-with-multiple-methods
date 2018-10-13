#-------------------------------------------------
#
# Project created by QtCreator 2018-09-20T23:31:11
#
#-------------------------------------------------

QT       += core gui widgets

TARGET = imageHash
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
        main.cpp \
        imgprocessor.cpp\
        ClickedLabel.cpp\
        inputdlg.cpp\
        settingdlg.cpp\
        showwidget.cpp

HEADERS += \
        imgprocessor.h\
        ClickedLabel.h\
        inputdlg.h\
        settingdlg.h\
        showwidget.h

FORMS += \
        imgprocessor.ui

INCLUDEPATH+=D:/opencv/build/include

CONFIG(debug,debug|release) {
LIBS += -LC:/Users/shiny/Documents/imageHash/ -lopencv_world340d
} else {
LIBS += -LC:/Users/shiny/Documents/imageHash/ -lopencv_world340
}

RESOURCES += \
    imagehash.qrc
