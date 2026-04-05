#include "settingsdialog.h"
#include "ui_settingsdialog.h"

SettingsDialog::SettingsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::SettingsDialog)
{
    ui->setupUi(this);
    RollSet = 1;
    ASASet = 200;

 }

SettingsDialog::~SettingsDialog()
{
    delete ui;
}


void SettingsDialog::on_radioButton_clicked() //18fps VGA (Default - shutter 1/43)
{
   emit RollChanged(1, 16000, 4, 0, 0, 640, 480);
   ui->horizontalSlider->setValue(16000);//shutter opens at 12ms, closes at 36ms, delay is 20ms so sensor shutter read out starts at 20+17=37ms
   emit Bitdepth_SettingChanged(16);
   return;
}

void SettingsDialog::on_radioButton_2_clicked() //18fps QHD
{
   emit RollChanged(2, 7500, 2, 288, 216, 720, 540); // y offset original 214, used to be 960 x 544 (offset x 168, y 400at 17500ms
   ui->horizontalSlider->setValue(7500);
   emit Bitdepth_SettingChanged(16);
   return;
}

void SettingsDialog::on_radioButton_3_clicked() //24fps VGA (shutter 1/59)
{
    emit RollChanged(3, 16000, 4, 0, 0, 640, 480);// shutter open 9ms; close 27ms; delay is 20ms so 7ms exposure
    ui->horizontalSlider->setValue(16000);
    emit Bitdepth_SettingChanged(16);
    return;
}

void SettingsDialog::on_radioButton_4_clicked() //6fps - 1fps 720p (shutter 1/43)
{
    emit RollChanged(4, 4000, 2, 168, 126, 960, 720);
    ui->horizontalSlider->setValue(4000);
    emit Bitdepth_SettingChanged(16);
    return;
}

void SettingsDialog::on_radioButton_5_clicked() //<1fps fullHD
{
    emit RollChanged(5, 15000, 1, 336, 432, 1920, 1080);
    ui->horizontalSlider->setValue(15000);
    emit Bitdepth_SettingChanged(8);
    return;
}

void SettingsDialog::on_radioButton_6_clicked() // Single Frame full HD, but true Single Frames get captured in separate directory (number 6)
{
    emit RollChanged(6, 15000, 1, 336, 432, 1920, 1080);
    ui->horizontalSlider->setValue(15000);
    emit Bitdepth_SettingChanged(16);
    return;
}

void SettingsDialog::on_radioButton_7_clicked() // Single Frame full sensor
{
    emit RollChanged(7, 15000, 1, 0, 0, 2592, 1944);
    ui->horizontalSlider->setValue(15000);
    emit Bitdepth_SettingChanged(16);
    return;
}

void SettingsDialog::on_radioButton_9_clicked()
{
    emit ASAChanged(100, 0.5);
}

void SettingsDialog::on_radioButton_8_clicked()
{
    emit ASAChanged(200, 6.1);
}

void SettingsDialog::on_radioButton_10_clicked()
{
    emit ASAChanged(400, 12.0);
}

void SettingsDialog::on_pushButton_3_clicked()//exit
{
       
}

void SettingsDialog::on_horizontalSlider_valueChanged(int value)
{
    emit ExposureChanged(value);

}

void SettingsDialog::on_ASA150_clicked()
{
    emit ASAChanged(150, 3.0);
}
