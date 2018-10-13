
    #ifndef CLICKEDLABEL_H_  
    #define CLICKEDLABEL_H_  
    #include <QLabel>  
    #include <QWidget>
    using namespace std;
    class ClickedLabel : public QLabel  
    {  
        Q_OBJECT
          public:
          ClickedLabel(QWidget * parent = 0);
          string route = "None.";
          void setRoute(string str);
          private:
          protected:
          virtual void mouseReleaseEvent(QMouseEvent * ev);
          signals:
          void clicked(void);
    };  
    #endif /* CLICKEDLABEL_H_ */  

