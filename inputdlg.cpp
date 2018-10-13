#include "inputdlg.h"
#include <QInputDialog>
#include <QMessageBox>
InputDlg::InputDlg(QWidget* parent,std::string pth,int sim):QDialog(parent)
{
    setWindowTitle(QString::fromStdString(pth));
    pic = new QLabel;
    QImage *img=new QImage;
    if(! ( img->load(QString::fromStdString(pth)) ) ) //加载图像
    {
        QMessageBox::information(this,
                                 tr("打开图像失败"),
                                 tr("打开图像失败!"));
        delete img;
    }else{
        pic->setPixmap(QPixmap::fromImage(*img));
    }

    similarity = new QLabel;
    similarity->setText("Hash误差： "+ QString::number(sim));


    mainLayout =new QGridLayout(this);
    //第x行，第y列开始，占1行1列

    mainLayout->addWidget(pic,0,0,1,2);
    mainLayout->addWidget(similarity,1,0,1,2);

    mainLayout->setMargin(15);
    mainLayout->setSpacing(10);

}
