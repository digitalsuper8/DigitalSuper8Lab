#ifndef SettingsDialog_H
#define SettingsDialog_H//

//#include "mythread.h"
#include <QDialog>
//new include
//#include "widget.h"

namespace Ui {
class SettingsDialog;
}

class SettingsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit SettingsDialog(QWidget *parent = 0);
    ~SettingsDialog();

private:
    Ui::SettingsDialog *ui;
    int RollSet;
    int ASASet;
    int exposure;

signals:
    //My created signal
    void ExposureChanged(int);
    void RollChanged(int,int,int,int,int,int,int);
    void ASAChanged(int,float);
    void Bitdepth_SettingChanged(int);
    void StatusUpdate(QString, bool);


public slots:

private slots:
    void on_radioButton_clicked();
    void on_radioButton_2_clicked();
    void on_radioButton_3_clicked();
    void on_radioButton_4_clicked();
    void on_radioButton_5_clicked();
    void on_radioButton_6_clicked();
    void on_radioButton_7_clicked();
    void on_radioButton_9_clicked();
    void on_radioButton_8_clicked();
    void on_radioButton_10_clicked();
    void on_pushButton_3_clicked();
    void on_horizontalSlider_valueChanged(int value);
   // void on_pushButton_clicked();
    void on_ASA150_clicked();
};

#endif // SettingsDialog_H
