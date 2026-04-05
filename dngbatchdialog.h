#pragma once


#include <QDialog>
#include <QThread>
#include <QPointer>
#include <QTimer>


class DngBatchConverterTiny; // forward


QT_BEGIN_NAMESPACE
namespace Ui { class DngBatchDialog; }
QT_END_NAMESPACE


class DngBatchDialog : public QDialog {
    Q_OBJECT
public:
    explicit DngBatchDialog(QWidget *parent = nullptr);
    ~DngBatchDialog() override;


private slots:
    void onBrowseSrc();
    void onBrowseDst();
    void onStart();
    void onStop();


    // Slots to consume converter signals
    void onProgress(int roll, int index, int total);
    void onFileConverted(int roll, int index, const QString &src, const QString &dst);
    void onRollFinished(int roll, int converted, int skipped, int failed);
    void onError(const QString &msg);


private:
    void setRunning(bool running);
    void ensureWorker();


    Ui::DngBatchDialog *ui;


    QThread workerThread; // dedicated worker thread
    DngBatchConverterTiny *conv{nullptr}; // lives in workerThread


    bool isRunning{false};
    int totalCount{0};
};
