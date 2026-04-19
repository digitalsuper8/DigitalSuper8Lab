// developthread.h
#ifndef DEVELOPTHREAD_H
#define DEVELOPTHREAD_H
#include "super8devparams.h"

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <QString>
#include <atomic>
#include <opencv2/core.hpp>
#include <QPointer>
#include <QMutex>
#include <chrono>

struct LeakBurstState
{
    bool active = false;
    int  phaseFrame = 0;
    int  rampUp = 5;
    int  hold  = 5;
    int  rampDown = 5;

    int  side = 0;
    float bandCenter = 0.f;
    float bandWidth  = 0.f;
    float sigmaBloom = 0.f;
    float strengthMul = 1.f;

    std::chrono::steady_clock::time_point nextAllowed;
};

class DevelopThread : public QObject
{
    Q_OBJECT
public:
    explicit DevelopThread(QObject* parent = nullptr);

    ~DevelopThread();

    // Public state
    int width = 0;
    int height = 0;
    int Roll = 0;
    int frames = 0;             // inclusive last index
    int fps = 24;               // render-to-file fps
    int codec = 0;
    int first_frame = 1;

    // Audio / mux options
    bool addAudio = false;
    bool muxAudioIntoVideo = false;
    QString ffmpegProgram = QStringLiteral("ffmpeg");

    QString lastRenderedVideoPath;
    QString lastRenderedWavPath;
    QString lastMuxedVideoPath;

    float Red = 1.f, Blue = 1.f, Green = 1.f;
    float Sat = 100.f, Contrast = 100.f, Bright = 0.f;

    cv::Mat imgBGR;
    int bitdepth = 16;
    bool HDR = false, filmlook = true, gammacorrect = false;
    bool useLog = false;
    char work_path[256] = {};
    char file_type[256] = "mp4";
    bool ImageDeveloped = false;

    double CurveRed = 8.0, CurveGreen = 8.0, CurveBlue = 8.0;
    ushort lut16[4096] = {};
    ushort lutSCurveRed[4096] = {}, lutSCurveGreen[4096] = {}, lutSCurveBlue[4096] = {};
    bool processingBusy_ = false;
    bool renderingBusy = false;
    float ExposureEV = 0.f;

signals:
    void sCurvesUpdated(QVector<quint16> r, QVector<quint16> g, QVector<quint16> b);
    void FrameDeveloped(int);
    void StatusUpdate(QString, bool);
    void playbackStarted();
    void playbackPaused();
    void playbackStopped();
    void renderFinished();

public slots:
    void setSuper8DevParams(const Super8DevParams &p);
    void setLogMode(int mode);
    void setToneCurveMode(int mode);
    void setGradingMode(int mode);
    void setSCurvePivot(int value);
    void init();

    void setFps(double fps);
    void setRange(int first, int last);
    void setLooping(bool enabled);
    void play();
    void pause();
    void stop();
    void seek(int index);

    void startRenderToFile();
    void cancelRender();

    void onDevelopFrame(int i);
    void calc_LutCurve();

private slots:
    void onTick();

private:
    cv::RNG m_rng;
    int m_currentFrameIndex = 0;
    LeakBurstState m_leakBurst;

    QPointer<QTimer> timer_;
    QElapsedTimer wallClock_;
    double playbackFps_ = 24.0;
    int rangeFirst_ = 1;
    int rangeLast_  = 1;
    int current_    = 1;
    bool looping_   = true;

    std::atomic<bool> cancelRequested_{false};

    void startPlaybackTimer();
    void stopPlaybackTimer();

    void applyLeakMask(cv::Mat &imgBgr, int frameIndex, float A);

    Super8DevParams m_super8;
    QMutex m_super8Mutex;

    void applySuper8Grain(cv::Mat &imgBgr);
    void applySuper8LightLeak(cv::Mat &imgBgr);

    void applyScratches(cv::Mat &imgBgr, int frameIndex);
    void drawSoftDust(cv::Mat& f32bgr, int cx, int cy, float radiusPx, float delta, float maxVal);
    void applyDust(cv::Mat& imgBgr, int frameIndex);
};

#endif // DEVELOPTHREAD_H
