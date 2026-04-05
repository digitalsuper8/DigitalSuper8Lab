// developthread.h
#ifndef DEVELOPTHREAD_H
#define DEVELOPTHREAD_H

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <atomic>
#include <opencv2/core.hpp>
#include <QPointer>

class DevelopThread : public QObject
{
    Q_OBJECT
public:
    explicit DevelopThread(QObject* parent = nullptr);
    ~DevelopThread();

    // Public state (kept from your original API)
    int width = 0;
    int height = 0;
    int Roll = 0;
    int frames = 0;             // inclusive last index
    int fps = 24;               // render-to-file fps
    int codec = 0;
    int first_frame = 1;

    float Red = 1.f, Blue = 1.f, Green = 1.f;
    float Sat = 100.f, Contrast = 100.f, Bright = 0.f;

    cv::Mat imgBGR;
    int bitdepth = 16;
    bool HDR = false, filmlook = true, gammacorrect = false;
    char work_path[256] = {};
    char file_type[256] = "mp4";
    bool ImageDeveloped = false;

    double CurveRed = 8.0, CurveGreen = 8.0, CurveBlue = 8.0;
    ushort lut16[4096] = {};
    ushort lutSCurveRed[4096] = {}, lutSCurveGreen[4096] = {}, lutSCurveBlue[4096] = {};
    cv::Mat lut8_array, lutS_array;
    bool processingBusy_ = false;
    bool renderingBusy = false;


signals:
    void sCurvesUpdated(QVector<quint16> r, QVector<quint16> g, QVector<quint16> b);
    void FrameDeveloped(int);
    void StatusUpdate(QString, bool);
    void playbackStarted();
    void playbackPaused();
    void playbackStopped();
    void renderFinished();          // emitted when render loop ends (or cancels)

public slots:
    // lifecycle/init in worker thread
    void init();                    // creates timer in the worker thread

    // playback controls (queued from UI thread)
    void setFps(double fps);
    void setRange(int first, int last);
    void setLooping(bool enabled);
    void play();
    void pause();
    void stop();
    void seek(int index);

    // render-to-file (queued from dialog/UI thread)
    void startRenderToFile();
    void cancelRender();

    // processing
    void onDevelopFrame(int i);
    void calc_LutCurve();

private slots:
    void onTick();                  // playback timer tick

private:
    // playback state
    QPointer<QTimer> timer_;
    QElapsedTimer wallClock_;
    double playbackFps_ = 24.0;
    int rangeFirst_ = 1;
    int rangeLast_  = 1;
    int current_    = 1;
    bool looping_   = true;

    // render cancel flag
    std::atomic<bool> cancelRequested_{false};

    // helpers
    void startPlaybackTimer();
    void stopPlaybackTimer();
};

#endif // DEVELOPTHREAD_H
