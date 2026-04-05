#ifndef DEVELOPDIALOG_H
#define DEVELOPDIALOG_H

#include <QDialog>

// Forward declare to avoid include in header
class DevelopThread;

namespace Ui {
class DevelopDialog;
}

class DevelopDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DevelopDialog(QWidget *parent = 0);
    ~DevelopDialog();

    // NEW: supply context before showing the dialog
    // devThread: the DevelopThread that will do the rendering
    // currentFrame: the frame where the UI cursor/slider currently is (0- or 1-based, match your app)
    // lastFrame: the maximum available frame index for this roll/sequence
    void setDevelopContext(DevelopThread* devThread, int currentFrame, int lastFrame);

signals:
    // (Kept for backward compatibility; not required anymore for starting the render)
    void DevelopSettings_Accepted(int fps, int codecId);
    void Start_from_currentframe(bool fromCurrent);
    void StatusUpdate(QString);

private slots:
    void on_spinBox_valueChanged(int arg1);
    void on_pushButton_clicked();      // (still emits your old signals if something listens)
    void on_CurrentFrameBox_clicked();
    void on_MPEGcheckBox_clicked(bool checked);
    void on_RAWcheckBox_clicked(bool checked);

    // NEW: called when the dialog is accepted (OK button)
    void onDialogAccepted();

private:
    Ui::DevelopDialog *ui;

    // Local state mirrored from UI
    int fps;          // 18 or 24
    int cod;          // 1=RAW (IYUV), 2=MPEG (MP4V)
    bool currentframe;

    // NEW: rendering context
    DevelopThread* dev = nullptr;
    int currentFrameIndex = 0;  // provided by caller
    int lastFrameIndex    = 0;  // provided by caller
};

#endif // DEVELOPDIALOG_H
