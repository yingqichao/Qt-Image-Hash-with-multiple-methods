#ifndef SETTINGDLG_H
#define SETTINGDLG_H


#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QDialog>
#include <QComboBox>
#include<QTextEdit>
using namespace std;

class SettingDlg : public QDialog
{
    Q_OBJECT
public:
    SettingDlg(QWidget* parent=0);
    string getMethod();

private:
    QWidget *selectWid;
    QLabel *label2;
    QComboBox *te2;
    QLabel *label;
    QTextEdit *te1;
    QString method = "DCT Hash";

public slots:
    void ShowMethod(QString);
};

#endif // INPUTDLG_H
