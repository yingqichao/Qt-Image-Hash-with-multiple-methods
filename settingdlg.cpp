#include "settingdlg.h"
#include <QInputDialog>
#include <QMessageBox>
using namespace std;

SettingDlg::SettingDlg(QWidget* parent):QDialog(parent)
{
    setWindowTitle("Setting");
    setFixedSize(400,240);
    selectWid = new QWidget(this);
    QHBoxLayout *HorizontalLayout = new QHBoxLayout(0);
    label = new QLabel;
    label->setText("Image Num:");

    te1 =new QTextEdit();
    te1->setFixedSize(120, 30);
    te1->setText(tr(""));
    HorizontalLayout->addWidget(label);
    HorizontalLayout->addWidget(te1);
    QHBoxLayout *HorizontalLayout2 = new QHBoxLayout(0);
    label2 = new QLabel;
    label2->setText("Method Selection:");

    te2 =new QComboBox();
    te2->setFixedSize(120, 30);
    te2->addItem(tr("Simplified Hash"));

    te2->addItem(tr("DCT Hash"));
    te2->addItem(tr("Perceptual Hash"));

    connect(te2,SIGNAL(activated(QString)),
            this,SLOT(ShowMethod(QString)));
    
    HorizontalLayout2->addWidget(label2);
    HorizontalLayout2->addWidget(te2);                                              //bind BtnIsEnable
    QVBoxLayout *VerticalLayout = new QVBoxLayout(0);
    VerticalLayout->addLayout(HorizontalLayout);
    VerticalLayout->addLayout(HorizontalLayout2);
    selectWid->setLayout(VerticalLayout);

}

void SettingDlg::ShowMethod(QString mtd)
{
    method = mtd;
}

string SettingDlg::getMethod()
{
   return method.toStdString();
}
