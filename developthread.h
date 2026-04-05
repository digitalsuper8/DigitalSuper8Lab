// developthread.h
#ifndef DEVELOPTHREAD_H
#define DEVELOPTHREAD_H
#include "super8devparams.h"

#include <QObject>
#include <QTimer>
#include <QElapsedTimer>
#include <atomic>
#include <opencv2/core.hpp>
#include <QPointer>
#include <QMutex>
#include <chrono>

struct LeakBurstState
{
    bool active = false;
    int  phaseFrame = 0;          // counts frames since burst start
    int  rampUp = 5;
    int  hold  = 5;
    int  rampDown = 5;

    int  side = 0;                // 0=left, 1=right
    float bandCenter = 0.f;       // 0..1
    float bandWidth  = 0.f;       // 0..1
    float sigmaBloom = 0.f;       // blur amount
    float strengthMul = 1.f;      // random strength multiplier

    std::chrono::steady_clock::time_point nextAllowed;
};

class DevelopThread : public QObject
{
    Q_OBJECT
public:
    explicit DevelopThread(QObject* parent = nullptr);

    enum class LogMode {
        Linear = 0,   // geen log, gewoon ACEScg-linear
        Log16  = 1,    // jouw bestaande LOG16 LUT
        ArriLogC3 = 2,
        ACEScct = 3    // <-- nieuwe log-mode
        // later: SLog3, LogC, BMD, etc.
    };

    enum class ToneCurveMode {
        None        = 0,
        SCurves     = 1,  // jouw filmlook S-curves
        FilmicHable = 2,
        Reinhard    = 3,
        ACESLike    = 4   // jouw bestaande applyACES
    };
    enum class GradingMode
    {
        ACES = 0,
        Log  = 1
    };


    ~DevelopThread();

    // Public state (kept from your original API)
    int baseCurveMode = 0;
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
    bool useLog = false;   // NEW: enable / disable S-Log-style LUT
    char work_path[256] = {};
    char file_type[256] = "mp4";
    bool ImageDeveloped = false;

    double CurveRed = 8.0, CurveGreen = 8.0, CurveBlue = 8.0;
    ushort lut16[4096] = {};
    ushort lutSCurveRed[4096] = {}, lutSCurveGreen[4096] = {}, lutSCurveBlue[4096] = {};
    cv::Mat lut8_array, lutS_array;
    bool processingBusy_ = false;
    bool renderingBusy = false;
    float ExposureEV = 0.f;   // 0 = no change, +1 = +1 stop, etc.


signals:
    void sCurvesUpdated(QVector<quint16> r, QVector<quint16> g, QVector<quint16> b);
    void FrameDeveloped(int);
    void StatusUpdate(QString, bool);
    void playbackStarted();
    void playbackPaused();
    void playbackStopped();
    void renderFinished();          // emitted when render loop ends (or cancels)

public slots:
    void setSuper8DevParams(const Super8DevParams &p);
    void setLogMode(int mode);
    void setToneCurveMode(int mode);
    void setGradingMode(int mode);
    void setSCurvePivot(int value);  // 0..100 from UI
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
    cv::RNG m_rng;
 int m_currentFrameIndex = 0;
    float m_prevLeakSlider = 0.0f;
    LeakBurstState m_leakBurst;

    LogMode       m_logMode       = LogMode::Linear;
    ToneCurveMode m_toneCurveMode = ToneCurveMode::SCurves;
    GradingMode m_gradingMode = GradingMode::ACES;
    float m_sCurvePivot = -1.0f;
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

    void applyLeakMask(cv::Mat &imgBgr, int frameIndex, float A);

    Super8DevParams m_super8;
    QMutex m_super8Mutex;

    void applySuper8Grain(cv::Mat &imgBgr);
    void applySuper8LightLeak(cv::Mat &imgBgr);
    void applySuper8LightLeakVertical(cv::Mat &imgBgr, int frameIndex, float sliderAmount01);
    void applyScratches(cv::Mat &imgBgr, int frameIndex);
    void drawSoftDust(cv::Mat& f32bgr, int cx, int cy, float radiusPx, float delta, float maxVal);
    void applyDust(cv::Mat& imgBgr, int frameIndex);
};

#endif // DEVELOPTHREAD_H
