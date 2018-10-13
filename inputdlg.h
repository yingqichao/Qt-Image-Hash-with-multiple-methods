#ifndef INPUTDLG_H
#define INPUTDLG_H

#include <QLabel>
#include <QPushButton>
#include <QGridLayout>
#include <QDialog>
class InputDlg : public QDialog
{
    Q_OBJECT
public:
    InputDlg(QWidget* parent=0,std::string pth = "",int sim = 0);

private:
    QLabel *pic;
    QLabel *similarity;
    QGridLayout *mainLayout;
};

#endif // INPUTDLG_H
