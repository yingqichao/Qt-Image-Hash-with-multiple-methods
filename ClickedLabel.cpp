#include <QLabel>
#include <QWidget>
#include "ClickedLabel.h"

ClickedLabel::ClickedLabel(QWidget * parent) : QLabel(parent)
{ }
void ClickedLabel::mouseReleaseEvent(QMouseEvent * ev)
{
    Q_UNUSED(ev)
    emit clicked();
}

void  ClickedLabel::setRoute(string str){
    route = str;
}
