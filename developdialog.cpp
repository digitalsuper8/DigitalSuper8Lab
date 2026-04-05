#include "developdialog.h"
#include "ui_developdialog.h"

#include <QDebug>
#include <QMetaObject>
#include <opencv2/videoio.hpp>
#include "developthread.h"

using namespace cv;

DevelopDialog::DevelopDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DevelopDialog)
{
    ui->setupUi(this);

    // Defaults
    fps = 18;
    cod = 1;            // RAW (IYUV) by default
    currentframe = false;

    // OK button already wired in .ui to accept()
    connect(this, &QDialog::accepted, this, &DevelopDialog::onDialogAccepted);
}

DevelopDialog::~DevelopDialog()
{
    delete ui;
}

void DevelopDialog::setDevelopContext(DevelopThread* devThread, int currentFrame, int lastFrame)
{
    dev = devThread;
    currentFrameIndex = currentFrame;
    lastFrameIndex    = lastFrame;
}

void DevelopDialog::on_spinBox_valueChanged(int arg1)
{
    fps = arg1;
}

void DevelopDialog::on_pushButton_clicked()
{
    // legacy signals removed; accept() will trigger onDialogAccepted()
}

void DevelopDialog::on_CurrentFrameBox_clicked()
{
    currentframe = !currentframe;
    qDebug() << "Develop from Current Frame =" << currentframe;
}

void DevelopDialog::on_MPEGcheckBox_clicked(bool checked)
{
    if (checked) {
        cod = 2; // MPEG (MP4V)
        qDebug() << "Codec set to MP4V (MPEG)";
    }
}

void DevelopDialog::on_RAWcheckBox_clicked(bool checked)
{
    if (checked) {
        cod = 1; // IYUV (AVI)
        qDebug() << "Codec set to IYUV (RAW-like)";
    }
}

void DevelopDialog::onDialogAccepted()
{
    if (!dev) {
        qWarning() << "[DevelopDialog] No DevelopThread set. Call setDevelopContext() before exec().";
        return;
    }

    const int  uiFps       = ui->spinBox->value();
    const bool fromCurrent = ui->CurrentFrameBox->isChecked();

    int fourcc      = (cod == 2) ? cv::VideoWriter::fourcc('M','P','4','V')
                            : cv::VideoWriter::fourcc('I','Y','U','V');
    const char* extC = (cod == 2) ? "mp4" : "avi";

    int firstIndex = fromCurrent ? currentFrameIndex : 1;
    if (firstIndex < 1) firstIndex = 1;
    int lastIndex  = (lastFrameIndex > 0) ? lastFrameIndex : firstIndex;

    // Make locals; DO NOT capture 'this'
    DevelopThread* worker = dev;                 // stable pointer owned by Processing
    const int      fpsVal = uiFps;
    const int      fourccVal = fourcc;
    const QString  ext = QString::fromLatin1(extC);
    const int      firstVal = firstIndex;
    const int      lastVal  = lastIndex;

    QMetaObject::invokeMethod(
        worker,
        [worker, fpsVal, fourccVal, ext, firstVal, lastVal]() {
            worker->fps         = fpsVal;
            worker->codec       = fourccVal;
            std::snprintf(worker->file_type, sizeof(worker->file_type),
                          "%s", ext.toLatin1().constData());
            worker->first_frame = firstVal;
            worker->frames      = lastVal;

            worker->startRenderToFile();
        },
        Qt::QueuedConnection
        );

    emit StatusUpdate(QStringLiteral("Starting development to file..."));
}

