//#include "widget.h"
#include "processing.h"
#include <QApplication>
#include <QFile>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    //qApp->setStyle("Fusion");// Ensure Windows default menubar style is overridden.

    QFile f(":/themes/kodak_yellow.qss");
    if (f.open(QFile::ReadOnly | QFile::Text))
        qApp->setStyleSheet(QString::fromUtf8(f.readAll()));

    a.setWindowIcon(QIcon(":/icons/DigitalSuper8_logo2.png"));

    Processing w;
    //Processing w;

    w.show();

    return a.exec();
}
