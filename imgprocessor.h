#ifndef IMGPROCESSOR_H
#define IMGPROCESSOR_H
#include<QProgressDialog>
#include <QMainWindow>
#include <QImage>
#include <QLabel>
#include <QMenu>
#include <QMenuBar>
#include <QAction>
#include <QComboBox>
#include <QSpinBox>
#include <QToolBar>
#include <QFontComboBox>
#include <QToolButton>
#include <QTextCharFormat>
#include "showwidget.h"
#include <QMainWindow>
#include "ClickedLabel.h"
#include<QTextEdit>
#include<QDockWidget>
#include <QTime>
#include <QApplication>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>

#include <QTextCodec>


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

class ImgProcessor : public QMainWindow
{
    Q_OBJECT

public:
    ImgProcessor(QWidget *parent = 0);
    ~ImgProcessor();
    void createActions();                        	//创建动作
    void createMenus();                           	//创建菜单
    void createToolBars();                      	//创建工具栏
    void loadFile(QString filename);
private:
    int method = 0;//0:pHash,1:simpleHash,2:DCTHash
    int threshold = 1;
    QList<QDockWidget*> m_docks;
    QTextCodec *codec;
    //dockWidget
    QDockWidget *dock;
//    QDockWidget *dataset_dock;
    QDockWidget *hostImage_dock;
    QDockWidget *code_dock;
    QDockWidget *bottom_dock;
    QDockWidget *bottom_dock_right;
    //from original
    std::string  strSrcImageName;
    std::string  strSrcImageName2;
    std::string  temp_img_address;
//    int iAvg1 = 0, iAvg2 = 0, i = 0, j = 0, tmp = 0, tmp1 = 0, ind = 0, iDiffNum = 0;
//    int arr1[6400], arr2[6400];
//    static int MAX_IMAGE_SIZE;

    string mark(string input,Mat imgMat,bool wr);
    string otsu(string input,Mat imgMat,bool wr);
    string histogram(string input,Mat imgMat,bool wr);
    Mat matchers(string input, Mat img_1,bool wr);
    void showDock();
    void removeAllDock();

//    QString strText;
//    QTextEdit *edit;
//    QTextEdit *te;
//    QWidget *mywid1;
//    QWidget *mywid2;
    QLabel *mtd;
    QLabel *pName;
    QLabel *fName;
    QComboBox *te2;
    QTextEdit *te1;
//    QWidget *ctrWid;
//    QPushButton *btn_Play;
//    QPushButton *btn_select;
//    QPushButton *btn_selectFolder;
//    QLabel *label;
//    QLabel *imageSelectedLabel;
//    QLabel *imageSelectedLabel2;
//    QLabel *paint;
//    InputDlg *inputDlg;
//    QTextEdit *te1;
//    QTextEdit *te2;
    QString filename = "* Not Selected *";
    QString foldname = "* Not Selected *";
    list<ClickedLabel *> matches;
    list<ClickedLabel *> database_img;
    list<ClickedLabel *> test_img;
//    ClickedLabel *matches;
    ClickedLabel *Images;
    bool changed = true;
    //主图像
    QLabel *imageLabel;
    QTextEdit *text;
    //Menus,bars and actions
    QMenu *fileMenu;                           		//各项菜单栏
    QMenu *zoomMenu;
    QMenu *rotateMenu;
    QMenu *SettingMenu;
    QMenu *WorkSpaceMenu;
    QMenu *RunMenu;
    QMenu *WindowMenu;
    QPixmap img;
//    ShowWidget *showWidget;
    QAction *window;
    QAction *openFileAction;                     	//文件菜单项
    QAction *NewFileAction;
    QAction *runFolder;
    QAction *runHash;
    QAction *PrintTextAction;
    QAction *PrintImageAction;
    QAction *exitAction;
    QAction *copyAction;                          	//编辑菜单项
    QAction *cutAction;
    QAction *pasteAction;
    QAction *aboutAction;
    QAction *rotate90Action;                     	//旋转菜单项
    QAction *rotate180Action;
    QAction *rotate270Action;
    QAction *mirrorVerticalAction;              	//镜像菜单项
    QAction *mirrorHorizontalAction;
    QAction *settingModification;
    QAction *showWorkSpace;
    QAction *showImageSpace;
    QToolBar *fileTool;                          	//工具栏
    QToolBar *zoomTool;
    QToolBar *rotateTool;
    QToolBar *doToolBar;
    QToolBar *listToolBar;                          //排序工具栏
protected slots:
    //from original
    void generateHashCode();
    void hashTest();
    void onMethodChange(int);
    void onThresholdChange();
//    void hashTest_match();
    void onclick(int i);
    void onclickFromDB(int i);
    void onclickFromTest(int i);
    //new
    void ShowNewFile();
    void ShowOpenFile();
    void ShowSettingDialog();
//    void ShowPrintText();
//    void ShowPrintImage();
//    void ShowZoomIn();
//    void ShowZoomOut();
    void ShowRotate90();
    void ShowRotate180();
    void ShowRotate270();
    void ShowMirrorVertical();
    void ShowMirrorHorizontal();
//    void ShowFontComboBox(QString comboStr);
//    void ShowSizeSpinBox(QString spinValue);
//    void ShowBoldBtn();
//    void ShowItalicBtn();
//    void ShowUnderlineBtn();
//    void ShowColorBtn();
//    void ShowCurrentFormatChanged(const QTextCharFormat &fmt);
//    void ShowList(int);
//    void ShowAlignment(QAction *act);
//    void ShowCursorPositionChanged();
};

#endif // IMGPROCESSOR_H
