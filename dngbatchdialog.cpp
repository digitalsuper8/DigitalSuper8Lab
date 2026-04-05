#include "dngbatchdialog.h"
#include "ui_dngbatchdialog.h"
#include "dngbatchconverter_tiny.h"

#include <QFileDialog>
#include <QDateTime>
#include <QMessageBox>

DngBatchDialog::DngBatchDialog(QWidget *parent)
    : QDialog(parent)
    , ui(new Ui::DngBatchDialog)
{
    ui->setupUi(this);

    // Sensible defaults per your paths
    ui->editSrc->setText("C:/Users/patri/Videos/XiImages");
    ui->editDst->setText("C:/Users/patri/Videos/DNG");
    ui->spinRoll->setValue(1);
    ui->editGlob->setText("XiCapture*.pgm");
    ui->comboCFA->addItems({"RGGB","GRBG","GBRG","BGGR"});
    ui->comboCFA->setCurrentText("RGGB");
    ui->spinBlack->setRange(0, 65535); ui->spinBlack->setValue(122);
    ui->spinWhite->setRange(1, 65535); ui->spinWhite->setValue(4095);

    ui->progressBar->setValue(0);

    connect(ui->btnBrowseSrc, &QPushButton::clicked, this, &DngBatchDialog::onBrowseSrc);
    connect(ui->btnBrowseDst, &QPushButton::clicked, this, &DngBatchDialog::onBrowseDst);
    connect(ui->btnStart, &QPushButton::clicked, this, &DngBatchDialog::onStart);
    connect(ui->btnStop, &QPushButton::clicked, this, &DngBatchDialog::onStop);

    // Prepare the worker thread (will create converter on demand)
    workerThread.setObjectName("DngBatchWorker");
}

DngBatchDialog::~DngBatchDialog(){
    onStop();
    workerThread.quit();
    workerThread.wait();
    delete ui;
}

void DngBatchDialog::ensureWorker() {
    if (conv) return;
    conv = new DngBatchConverterTiny();
    conv->moveToThread(&workerThread);

    // Wire signals from worker to UI
    connect(conv, &DngBatchConverterTiny::progress, this, &DngBatchDialog::onProgress);
    connect(conv, &DngBatchConverterTiny::fileConverted, this, &DngBatchDialog::onFileConverted);
    connect(conv, &DngBatchConverterTiny::rollFinished, this, &DngBatchDialog::onRollFinished);
    connect(conv, &DngBatchConverterTiny::errorOccured, this, &DngBatchDialog::onError);

    // Delete converter when thread finishes
    connect(&workerThread, &QThread::finished, conv, &QObject::deleteLater);

    if (!workerThread.isRunning()) workerThread.start();
}

void DngBatchDialog::onBrowseSrc() {
    const QString dir = QFileDialog::getExistingDirectory(this, "Select source root (contains images<roll>)", ui->editSrc->text());
    if (!dir.isEmpty()) ui->editSrc->setText(dir);
}

void DngBatchDialog::onBrowseDst() {
    const QString dir = QFileDialog::getExistingDirectory(this, "Select destination root for DNG", ui->editDst->text());
    if (!dir.isEmpty()) ui->editDst->setText(dir);
}

void DngBatchDialog::setRunning(bool running) {
    isRunning = running;
    ui->btnStart->setEnabled(!running);
    ui->btnStop->setEnabled(running);
    ui->groupSettings->setEnabled(!running);
}

void DngBatchDialog::onStart() {
    ensureWorker();

    // Apply settings to converter (across threads via queued invocations)
    const QString src = ui->editSrc->text();
    const QString dst = ui->editDst->text();

    QMetaObject::invokeMethod(conv, [=]{ conv->setWorkPath(src); });
    QMetaObject::invokeMethod(conv, [=]{ conv->setUsbPath(dst); });
    QMetaObject::invokeMethod(conv, [=]{ conv->setSrcGlob(ui->editGlob->text()); });
    QMetaObject::invokeMethod(conv, [=]{ conv->setCfaPattern(ui->comboCFA->currentText()); });
    QMetaObject::invokeMethod(conv, [=]{ conv->setBlackLevel(ui->spinBlack->value()); });
    QMetaObject::invokeMethod(conv, [=]{ conv->setWhiteLevel(ui->spinWhite->value()); });

    ui->log->clear();
    ui->progressBar->setValue(0);
    totalCount = 0; // will be set when progress starts

    setRunning(true);

    const int roll = ui->spinRoll->value();
    // Kick the job
    QMetaObject::invokeMethod(conv, [=]{ conv->convertRoll(roll); });
}

void DngBatchDialog::onStop() {
   // if (!conv) return;
   // QMetaObject::invokeMethod(conv, &DngBatchConverterTiny::cancel);

    if (!conv) return;
    conv->cancel();  // direct call, no queued signal
}

void DngBatchDialog::onProgress(int roll, int index, int total) {
    if (totalCount == 0) totalCount = total;
    if (totalCount > 0) {
        int pct = qBound(0, int((double(index+1) / double(totalCount)) * 100.0), 100);
        ui->progressBar->setValue(pct);
    }
  //  ui->statusbar->showMessage(QString("Roll %1: %2/%3").arg(roll).arg(index+1).arg(totalCount));
}

void DngBatchDialog::onFileConverted(int roll, int index, const QString &src, const QString &dst) {
    ui->log->append(QString("✔ [%1:%2] %3 → %4").arg(roll).arg(index).arg(src, dst));
}

void DngBatchDialog::onRollFinished(int roll, int converted, int skipped, int failed) {
    ui->log->append(QString("Roll %1 done: %2 ok, %3 failed, %4 skipped.")
                        .arg(roll).arg(converted).arg(failed).arg(skipped));
    ui->progressBar->setValue(100);
    setRunning(false);
}

void DngBatchDialog::onError(const QString &msg) {
    ui->log->append(QString("! %1").arg(msg));
}
