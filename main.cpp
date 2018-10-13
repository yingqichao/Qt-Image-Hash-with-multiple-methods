#include "imgprocessor.h"
#include <QApplication>
#include <QTextCodec>

#include <QSplashScreen>

int main(int argc, char *argv[])
{

//    // 以下部分解决中文乱码

//    QTextCodec *codec = QTextCodec::codecForName("utf8"); //GBK gbk


   QTextCodec::setCodecForLocale(QTextCodec::codecForName("GBK"));

//    QTextCodec::setCodecForCStrings(codec);

//    // 以上部分解决中文乱码

    QApplication a(argc, argv);
    QFont f("ZYSong18030",12);                        //设置显示的字体格式
    a.setFont(f);
    QPixmap pixmap(":/new/icon/111.png");				//(a)
    QSplashScreen splash(pixmap);			//(b)
    splash.show();
    ImgProcessor w;
    w.show();
    splash.finish(&w);

    return a.exec();
}
