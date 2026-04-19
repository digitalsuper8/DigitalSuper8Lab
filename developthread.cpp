// developthread.cpp
#include "developthread.h"
#include <QtCore>
#include <QDebug>
#include <QMetaObject>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <random>
#include <algorithm> // std::clamp
#include <cmath>     // std::exp, std::ceil

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/photo.hpp>

#include "processing.h"
#include "ds8audioprocessor.h"
#include "ds8videomuxer.h"

#include <QFileInfo>
#include <QDir>

using namespace cv;
using namespace std;

namespace
{

//new helpers for wide gamut version:

namespace
{

static inline float clamp01f(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

static inline cv::Vec3f clamp01v(const cv::Vec3f& v)
{
    return cv::Vec3f(clamp01f(v[0]), clamp01f(v[1]), clamp01f(v[2]));
}

// ------------------------------------------------------------
// Exact float benadering van jouw huidige LOG16 LUT-formule
// Input: linear 0..1 where 1.0 == 4095
// Output: log encoded 0..1
// ------------------------------------------------------------
static inline float log16FloatExact(float x)
{
    x = std::max(0.0f, x);

    const double code = x * 4095.0; // same 12-bit code domain as before

    // protect against invalid log argument below black
    if (code <= 32.0) {
        return 0.0f;
    }

    const double y =
        (((3757.0 - 32.0) * 0.432699) *
         std::log10(((code - 32.0) / (470.0 - 32.0)) + 0.037584))
        + (0.616596 * 4095.0);

    return clamp01f(static_cast<float>(y / 4095.0));
}

// ------------------------------------------------------------
// Float version of your S-curve LUT logic
// Input/output both 0..1
// curveStrength == CurveRed / CurveGreen / CurveBlue
// ------------------------------------------------------------
static inline float sCurveFloat(float x, float curveStrength)
{
    x = clamp01f(x);

    const double in12  = x * 4095.0;
    const double pivot = 2450.0;

    const double y =
        (1.0 / (1.0 + std::exp(-(curveStrength / 4095.0) * (in12 - pivot)))) * 4095.0;

    return clamp01f(static_cast<float>(y / 4095.0));
}

// ------------------------------------------------------------
// Approx sRGB/709 linear -> Rec.2020 linear
// OpenCV image is BGR, matrix math below is done in RGB order
// ------------------------------------------------------------
static inline cv::Vec3f rgb709ToRgb2020(const cv::Vec3f& rgb)
{
    // from linear Rec.709/sRGB to linear Rec.2020
    const float m[3][3] = {
        {0.6274040f, 0.3292820f, 0.0433136f},
        {0.0690970f, 0.9195400f, 0.0113612f},
        {0.0163916f, 0.0880132f, 0.8955950f}
    };

    return cv::Vec3f(
        m[0][0]*rgb[0] + m[0][1]*rgb[1] + m[0][2]*rgb[2],
        m[1][0]*rgb[0] + m[1][1]*rgb[1] + m[1][2]*rgb[2],
        m[2][0]*rgb[0] + m[2][1]*rgb[1] + m[2][2]*rgb[2]
        );
}

// ------------------------------------------------------------
// Rec.2020 linear -> Rec.709/sRGB linear
// ------------------------------------------------------------
static inline cv::Vec3f rgb2020ToRgb709(const cv::Vec3f& rgb)
{
    const float m[3][3] = {
        { 1.6604960f, -0.5876560f, -0.0728400f},
        {-0.1245470f,  1.1328950f, -0.0083480f},
        {-0.0181540f, -0.1005970f,  1.1187510f}
    };

    return cv::Vec3f(
        m[0][0]*rgb[0] + m[0][1]*rgb[1] + m[0][2]*rgb[2],
        m[1][0]*rgb[0] + m[1][1]*rgb[1] + m[1][2]*rgb[2],
        m[2][0]*rgb[0] + m[2][1]*rgb[1] + m[2][2]*rgb[2]
        );
}

// ------------------------------------------------------------
// Apply per-pixel Rec.709 -> Rec.2020 on a BGR float image
// ------------------------------------------------------------
static inline void bgr709ToBgr2020(cv::Mat& img)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            const cv::Vec3f bgr = row[x];
            const cv::Vec3f rgb(bgr[2], bgr[1], bgr[0]);
            const cv::Vec3f rgb2020 = rgb709ToRgb2020(rgb);
            row[x] = cv::Vec3f(rgb2020[2], rgb2020[1], rgb2020[0]);
        }
    }
}

static inline void bgr2020ToBgr709(cv::Mat& img)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            const cv::Vec3f bgr = row[x];
            const cv::Vec3f rgb(bgr[2], bgr[1], bgr[0]);
            const cv::Vec3f rgb709 = rgb2020ToRgb709(rgb);
            row[x] = cv::Vec3f(rgb709[2], rgb709[1], rgb709[0]);
        }
    }
}

// ------------------------------------------------------------
// Wide-gamut white balance in linear light, camera-ish domain
// Keep your current ratios exactly: R=1.00, G=0.75, B=1.12
// ------------------------------------------------------------
static inline void applyWhiteBalanceFloatLinearBGR(cv::Mat& img,
                                                   float wbR,
                                                   float wbG,
                                                   float wbB)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x]; // BGR
            p[0] *= wbB;
            p[1] *= wbG;
            p[2] *= wbR;
        }
    }
}

// ------------------------------------------------------------
// Apply LOG16 in float per channel
// ------------------------------------------------------------
static inline void applyLog16Float(cv::Mat& img)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x];
            p[0] = log16FloatExact(p[0]);
            p[1] = log16FloatExact(p[1]);
            p[2] = log16FloatExact(p[2]);
        }
    }
}

// ------------------------------------------------------------
// Apply your channel S-curves in float
// ------------------------------------------------------------
static inline void applySCurveFloat(cv::Mat& img,
                                    float curveBlue,
                                    float curveGreen,
                                    float curveRed)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x];
            p[0] = sCurveFloat(p[0], curveBlue);
            p[1] = sCurveFloat(p[1], curveGreen);
            p[2] = sCurveFloat(p[2], curveRed);
        }
    }
}

// ------------------------------------------------------------
// Contrast / brightness in log domain
// Contrast 100 => neutral
// Bright 0 => neutral
// ------------------------------------------------------------
static inline void applyContrastBrightnessFloat(cv::Mat& img,
                                                float contrastPercent,
                                                float brightUiValue)
{
    CV_Assert(img.type() == CV_32FC3);

    const float c = contrastPercent / 100.0f;
    const float b = brightUiValue / 255.0f;   // keep old feel roughly

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x];
            for (int k = 0; k < 3; ++k) {
                p[k] = ((p[k] - 0.5f) * c) + 0.5f + b;
            }
        }
    }
}

// ------------------------------------------------------------
// Saturation without HSV, in log RGB space
// Sat 100 => neutral
// ------------------------------------------------------------
static inline void applySaturationFloat(cv::Mat& img, float satPercent)
{
    CV_Assert(img.type() == CV_32FC3);

    const float s = satPercent / 100.0f;

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x]; // BGR

            const float luma =
                0.0722f * p[0] +
                0.7152f * p[1] +
                0.2126f * p[2];

            p[0] = luma + (p[0] - luma) * s;
            p[1] = luma + (p[1] - luma) * s;
            p[2] = luma + (p[2] - luma) * s;
        }
    }
}

// ------------------------------------------------------------
// Final RGB channel gains
// ------------------------------------------------------------
static inline void applyRgbTrimFloat(cv::Mat& img,
                                     float blueGain,
                                     float greenGain,
                                     float redGain)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y) {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x) {
            cv::Vec3f& p = row[x];
            p[0] *= blueGain;
            p[1] *= greenGain;
            p[2] *= redGain;
        }
    }
}

static inline void clampImage01(cv::Mat& img)
{
    cv::min(img, 1.0f, img);
    cv::max(img, 0.0f, img);
}

} // namespace

inline void applyWhiteBalance16(cv::Mat& img, float wbR, float wbG, float wbB)
{
    CV_Assert(img.type() == CV_16UC3);

    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec<unsigned short, 3>* row = img.ptr<cv::Vec<unsigned short, 3>>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec<unsigned short, 3>& p = row[x]; // BGR

            const float b = p[0] * wbB;
            const float g = p[1] * wbG;
            const float r = p[2] * wbR;

            p[0] = static_cast<unsigned short>(std::min(4095.0f, std::max(0.0f, b)));
            p[1] = static_cast<unsigned short>(std::min(4095.0f, std::max(0.0f, g)));
            p[2] = static_cast<unsigned short>(std::min(4095.0f, std::max(0.0f, r)));
        }
    }
}

inline float toLogC3(float x)
{
    const float cut = 0.010591f;
    const float a = 0.247190f;
    const float b = 0.385537f;
    const float c = 0.598206f;
    const float d = 5.555556f;
    const float e = d * a;         // 1.373994
    const float f = c - e * cut;   // 0.583510

    if (x < 0.0f) x = 0.0f;

    if (x >= cut)
        return a * std::log10(x + b) + c;
    else
        return e * x + f;
}

static float plannedLeakEnvelope(int frameIndex, const Super8DevParams& p)
{
    for (int n : p.leakFrames) {
        if (n <= 0) continue;

        int d = frameIndex - n;

        // ramp up: F-5..F-1
        if (d >= -5 && d <= -1) return float(d + 6) / 5.0f;        // 0.2..1.0

        // full: F..F+4
        if (d >= 0 && d <= 4) return 1.0f;

        // ramp down: F+5..F+9
        if (d >= 5 && d <= 9) return 1.0f - float(d - 4) / 5.0f;  // 0.8..0.0
    }
    return 0.0f;
}
}

DevelopThread::DevelopThread(QObject* parent) : QObject(parent)
{
    m_rng = cv::RNG((uint64)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    m_leakBurst.nextAllowed = std::chrono::steady_clock::now();
    Roll = 1;
    renderingBusy = false;
    ImageDeveloped = false;
    sprintf(work_path, "C:/Users/patri/Videos/XiImages");
    sprintf(file_type, "mp4");
    filmlook = true;
    gammacorrect = false;
    CurveRed = 8.0;
    CurveGreen = 8.0;
    CurveBlue = 8.0;
    codec = VideoWriter::fourcc('I', 'Y', 'U', 'V');
    //codec = VideoWriter::fourcc('M', 'P', '4', 'V');

    for (int i = 0; i < 4096; i++)
    {
        lut16[i] = saturate_cast<ushort>(((3757.0-32.0)*0.432699)*log10((((double)i-(32.0))/(470.0-32.0))+0.037584) + (0.616596*4095));
    }
    calc_LutCurve();

    // Playback defaults
    playbackFps_ = 18.0;
    rangeFirst_ = 0;
    rangeLast_  = 0;
    current_    = 0;
    looping_    = true;
}

DevelopThread::~DevelopThread()
{
    // Ensure timer is stopped and deleted in the correct thread
    if (timer_) {
        QMetaObject::invokeMethod(timer_, "stop", Qt::BlockingQueuedConnection);
    }
}

void DevelopThread::init()
{
    if (timer_) return;

    timer_ = new QTimer(this);
    timer_->setTimerType(Qt::PreciseTimer);
    connect(timer_, &QTimer::timeout, this, &DevelopThread::onTick);
}

void DevelopThread::setFps(double fps)
{
    playbackFps_ = (fps > 0.0) ? fps : 24.0;
    this->fps = int(std::round(playbackFps_));

    if (timer_ && timer_->isActive()) {
        timer_->start(qMax(1, int(std::round(1000.0 / playbackFps_))));
    }
}

void DevelopThread::setRange(int first, int last)
{
    rangeFirst_ = std::max(1, first);
    rangeLast_  = std::max(rangeFirst_, last);
    current_    = std::clamp(current_, rangeFirst_, rangeLast_);
}

void DevelopThread::setLooping(bool enabled)
{
    looping_ = enabled;
}

void DevelopThread::play()
{
    if (!timer_) init();
    if (renderingBusy) return;

    startPlaybackTimer();
    emit playbackStarted();
}

void DevelopThread::pause()
{
    stopPlaybackTimer();
    emit playbackPaused();
}

void DevelopThread::stop()
{
    stopPlaybackTimer();
    current_ = rangeFirst_;
    emit playbackStopped();
}

void DevelopThread::seek(int index)
{
    current_ = std::clamp(index, rangeFirst_, rangeLast_);
}

void DevelopThread::startRenderToFile()
{
    qDebug() << "[AUDIO TEST] patched startRenderToFile entered. addAudio =" << addAudio
             << "muxAudioIntoVideo =" << muxAudioIntoVideo
             << "work_path =" << work_path
             << "roll =" << Roll
             << "first_frame =" << first_frame
             << "frames =" << frames;

    renderingBusy = true;
    cancelRequested_.store(false, std::memory_order_relaxed);

    if ((height == 720) || (height == 480)) {
        height = 1080;
        width  = 1440;
    }
    const cv::Size S(width, height);

    char vid_name[256];
    std::snprintf(vid_name, sizeof(vid_name),
                  "C:/Users/patri/Videos/XiCapture_old/DigitalSuper8_output/Super8mpg%02d.%s",
                  Roll, file_type);

    const QString videoPath = QString::fromLocal8Bit(vid_name);

    cv::VideoWriter w;
    if (!w.open(vid_name, codec, fps, S, true)) {
        emit StatusUpdate("Could not open the output video for write.", true);
        renderingBusy = false;
        return;
    }

    emit StatusUpdate("Developing film, please wait.", false);

    cv::Mat HDimage;
    bool cancelled = false;

    for (int i = first_frame; i <= frames; ++i) {
        if (cancelRequested_.load(std::memory_order_relaxed)) {
            emit StatusUpdate("Render cancelled.", true);
            cancelled = true;
            break;
        }

        onDevelopFrame(i);
        if (!ImageDeveloped)
            continue;

        // Always feed 8-bit BGR to VideoWriter
        cv::Mat frameOut;
        if (imgBGR.depth() == CV_16U) {
            imgBGR.convertTo(frameOut, CV_8UC3, 1.0 / 257.0);
        } else {
            frameOut = imgBGR;
        }

        if (height == 1080) {
            cv::resize(frameOut, HDimage, S, 0.0, 0.0, cv::INTER_CUBIC);
            w << HDimage;
        } else {
            w << frameOut;
        }
    }

    w.release();

    if (cancelled) {
        renderingBusy = false;
        emit renderFinished();
        return;
    }

    lastRenderedVideoPath = videoPath;
    lastRenderedWavPath.clear();
    lastMuxedVideoPath.clear();

    if (addAudio)
    {
        emit StatusUpdate("Building stretched WAV from PCM files...", false);

        QFileInfo vfi(videoPath);
        const QString baseName = vfi.completeBaseName();
        const QString outDir   = vfi.absolutePath();

        const QString wavPath =
            QStringLiteral("%1/%2_audio.wav").arg(outDir, baseName);

        const auto wavResult = DS8AudioProcessor::buildStretchedWavForRoll(
            QString::fromLocal8Bit(work_path),
            Roll,
            first_frame,
            frames,
            fps,
            wavPath
            );

        if (!wavResult.ok) {
            emit StatusUpdate(QStringLiteral("Video ready, but WAV failed: %1").arg(wavResult.error), true);
        } else {
            lastRenderedWavPath = wavResult.wavPath;

            emit StatusUpdate(QStringLiteral("WAV created: %1")
                                  .arg(QFileInfo(wavResult.wavPath).fileName()),
                              false);

            if (muxAudioIntoVideo)
            {
                emit StatusUpdate("Muxing audio into rendered video...", false);

                const QString muxedVideoPath =
                    QStringLiteral("%1/%2_with_audio.%3")
                        .arg(outDir, baseName, vfi.suffix());

                const auto muxResult = DS8VideoMuxer::muxAudioIntoVideo(
                    videoPath,
                    wavResult.wavPath,
                    muxedVideoPath,
                    ffmpegProgram
                    );

                if (!muxResult.ok) {
                    emit StatusUpdate(QStringLiteral("WAV created, but mux failed: %1").arg(muxResult.error), true);
                    qDebug().noquote() << "[MUX] Failed. stderr was:\n" << muxResult.stdErr;
                    qDebug().noquote() << "[MUX] Failed. stdout was:\n" << muxResult.stdOut;
                    emit StatusUpdate(QStringLiteral("WAV created, but mux failed. See Application Output.\n%1")
                                          .arg(muxResult.error),
                                      true);
                } else {
                    lastMuxedVideoPath = muxResult.outputPath;
                    emit StatusUpdate(QStringLiteral("Video + audio ready: %1")
                                          .arg(QFileInfo(muxResult.outputPath).fileName()),
                                      false);
                }
            }
        }
    }

    emit StatusUpdate("Status OK, waiting...", false);
    emit renderFinished();
    renderingBusy = false;
}


void DevelopThread::cancelRender()
{
    cancelRequested_.store(true);
}

void DevelopThread::startPlaybackTimer()
{
    if (!timer_) init();
    if (!timer_) return;

    wallClock_.restart();
    timer_->start(qMax(1, int(std::round(1000.0 / playbackFps_))));
}

void DevelopThread::stopPlaybackTimer()
{
    if (timer_) {
        timer_->stop();
    }
}

void DevelopThread::onTick()
{
    if (processingBusy_ || renderingBusy) return;

    processingBusy_ = true;

    onDevelopFrame(current_);

    if (current_ >= rangeLast_) {
        if (looping_) current_ = rangeFirst_;
        else          stop();
    } else {
        ++current_;
    }
    processingBusy_ = false;
}


// ----------------- Your NEW code below -----------------

void DevelopThread::onDevelopFrame(int i)
{
    m_currentFrameIndex = i;
    ImageDeveloped = false;

    char in_name[256];
    std::snprintf(in_name, sizeof(in_name),
                  "%s/images%d/XiCapture%03d.pgm", work_path, Roll, i);

    cv::Mat imgBAY = cv::imread(in_name, cv::IMREAD_ANYDEPTH);
    if (imgBAY.empty()) {
        qDebug() << "FROM function Develop Frame: No data in image";
        return;
    }

    // ------------------------------------------------------------
    // Unify 8-bit and 16-bit RAW into one 12-bit-style path
    // ------------------------------------------------------------
    cv::Mat imgBAYwork;
    if (imgBAY.depth() == CV_16U) {
        imgBAYwork = imgBAY;
        qDebug() << "Native 16-bit Bayer input";
    } else {
        imgBAY.convertTo(imgBAYwork, CV_16UC1, 16.0); // 8-bit -> ~12-bit domain
        qDebug() << "Promoted 8-bit Bayer input to 16-bit/12-bit domain";
    }

    // ------------------------------------------------------------
    // Demosaic
    // ------------------------------------------------------------
    cv::cvtColor(imgBAYwork, imgBGR, cv::COLOR_BayerGB2BGR, 3);

    qDebug() << "successfully demosaiced, bit depth =" << imgBGR.depth()
             << "number of channels:" << imgBGR.channels();

    // ------------------------------------------------------------
    // Move as early as possible into float linear 0..1
    // ------------------------------------------------------------
    imgBGR.convertTo(imgBGR, CV_32FC3, 1.0 / 4095.0, 0.0);

    // ------------------------------------------------------------
    // Exposure in linear light
    // ------------------------------------------------------------
    if (std::abs(ExposureEV) > 0.0001f) {
        const float exposureMul = std::pow(2.0f, ExposureEV);
        imgBGR *= exposureMul;
    }

    // ------------------------------------------------------------
    // White balance in linear camera-ish domain
    // Keep your current ratios exactly
    // ------------------------------------------------------------
    applyWhiteBalanceFloatLinearBGR(imgBGR, 1.00f, 0.75f, 1.12f);

    // Prevent negatives, but allow highlight headroom > 1 before log if desired
    cv::max(imgBGR, 0.0f, imgBGR);

    // ------------------------------------------------------------
    // Move into a larger working space: linear Rec.2020
    // ------------------------------------------------------------
    bgr709ToBgr2020(imgBGR);
    cv::max(imgBGR, 0.0f, imgBGR);

    // ------------------------------------------------------------
    // Your LOG16, but now in float
    // ------------------------------------------------------------
    applyLog16Float(imgBGR);

    // ------------------------------------------------------------
    // Optional extra S-curve, also in float
    // ------------------------------------------------------------
    if (filmlook) {
        applySCurveFloat(imgBGR,
                         static_cast<float>(CurveBlue),
                         static_cast<float>(CurveGreen),
                         static_cast<float>(CurveRed));
    }

    // ------------------------------------------------------------
    // Float grading in log space, no HSV
    // ------------------------------------------------------------
    applyContrastBrightnessFloat(imgBGR,
                                 static_cast<float>(Contrast),
                                 static_cast<float>(Bright));

    applySaturationFloat(imgBGR,
                         static_cast<float>(Sat));

    applyRgbTrimFloat(imgBGR,
                      static_cast<float>(Blue),
                      static_cast<float>(Green),
                      static_cast<float>(Red));

    clampImage01(imgBGR);

    // ------------------------------------------------------------
    // Back to display/output space
    // ------------------------------------------------------------
    bgr2020ToBgr709(imgBGR);
    clampImage01(imgBGR);

    // ------------------------------------------------------------
    // Final output to 8-bit preview/video
    // ------------------------------------------------------------
    imgBGR.convertTo(imgBGR, CV_8UC3, 255.0, 0.0);

    if (gammacorrect) {
        imgBGR = correctGamma(imgBGR, (1.0 / 2.2));
    }

    cv::flip(imgBGR, imgBGR, -1);

    // ------------------------------------------------------------
    // Keep your current Super8 post effects
    // ------------------------------------------------------------
    applySuper8LightLeak(imgBGR);
    applySuper8Grain(imgBGR);
    applyScratches(imgBGR, i);
    applyDust(imgBGR, i);

    ImageDeveloped = true;
    emit FrameDeveloped(i);
}

void DevelopThread::calc_LutCurve()
{
    for (int i = 0; i < 4096; i++)
    {
        lutSCurveRed[i] =
            cv::saturate_cast<ushort>(
                (1 / (1 + exp((double)(-(CurveRed / 4095.0) * ((i - 2450) - 0))))) * 4095.0
                );

        lutSCurveGreen[i] =
            cv::saturate_cast<ushort>(
                (1 / (1 + exp((double)(-(CurveGreen / 4095.0) * ((i - 2450) - 0))))) * 4095.0
                );

        lutSCurveBlue[i] =
            cv::saturate_cast<ushort>(
                (1 / (1 + exp((double)(-(CurveBlue / 4095.0) * ((i - 2450) - 0))))) * 4095.0
                );
    }

    // Keep current UI preview support
    QVector<quint16> r(4096), g(4096), b(4096);
    for (int i = 0; i < 4096; ++i) {
        r[i] = lutSCurveRed[i];
        g[i] = lutSCurveGreen[i];
        b[i] = lutSCurveBlue[i];
    }
    emit sCurvesUpdated(r, g, b);
}

void DevelopThread::setGradingMode(int)
{
}

void DevelopThread::setLogMode(int)
{
}

void DevelopThread::setToneCurveMode(int)
{
}

void DevelopThread::setSCurvePivot(int)
{
    calc_LutCurve();
}

void DevelopThread::setSuper8DevParams(const Super8DevParams &p)
{
    QMutexLocker lock(&m_super8Mutex);
    m_super8 = p;
}

static inline float clamp02(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }


void DevelopThread::applySuper8Grain(cv::Mat &imgBgr)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }

    if (!p.enabled) return;
    float amount = clamp02(p.grain);      // 0..1 from slider

    if (amount <= 0.0001f) return;

    // Support both 8-bit preview and 16-bit internal if you ever reuse this elsewhere
    const int type = imgBgr.type();
    const float maxVal =
        (type == CV_8UC3)  ? 255.0f :
            (type == CV_16UC3) ? 65535.0f : 0.0f;

    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    // More film-like: multiplicative grain, resolution-aware size,
    // a little luminance dependence (slightly more visible in mids/shadows).
    const int w = f.cols;
    const int h = f.rows;

    // ---- Grain character tuning ----
    // Scale noise size a bit with resolution
    const float resScale = std::sqrt((w * h) / (640.0f * 480.0f)); // 1.0 around VGA
    int coarseW = std::max(1, int(std::round(0.65f * resScale)));
    int coarseH = std::max(1, int(std::round(0.65f * resScale)));

    // Base amplitude
    // amount=1 -> still subtle; tune up/down here if needed
    const float sigmaLumaBase   = 0.065f * amount;
    const float sigmaChromaBase = 0.020f * amount;

    // Random fields (coarse)
    cv::Mat noiseL(h / coarseH + 2, w / coarseW + 2, CV_32F);
    cv::randn(noiseL, 0.0f, 1.0f);

    cv::Mat noiseC1(h / coarseH + 2, w / coarseW + 2, CV_32F);
    cv::Mat noiseC2(h / coarseH + 2, w / coarseW + 2, CV_32F);
    cv::randn(noiseC1, 0.0f, 1.0f);
    cv::randn(noiseC2, 0.0f, 1.0f);

    // Resize to image size (coarse grain blobs)
    cv::resize(noiseL,  noiseL,  f.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(noiseC1, noiseC1, f.size(), 0, 0, cv::INTER_CUBIC);
    cv::resize(noiseC2, noiseC2, f.size(), 0, 0, cv::INTER_CUBIC);

    // Optional tiny blur so it doesn’t feel digital
    cv::GaussianBlur(noiseL,  noiseL,  cv::Size(0,0), 0.35);
    cv::GaussianBlur(noiseC1, noiseC1, cv::Size(0,0), 0.35);
    cv::GaussianBlur(noiseC2, noiseC2, cv::Size(0,0), 0.35);

    // Compute luma for masking grain strength a bit
    std::vector<cv::Mat> ch(3);
    cv::split(f, ch); // BGR
    cv::Mat luma = 0.114f * ch[0] + 0.587f * ch[1] + 0.299f * ch[2];
    luma /= maxVal; // 0..1-ish

    // More visible in mids/shadows, less in highlights
    cv::Mat lumaMask;
    {
        // mask = 0.65 + 0.55*(1-luma), clamped
        lumaMask = 0.65f + 0.55f * (1.0f - luma);
        cv::min(lumaMask, 1.20f, lumaMask);
        cv::max(lumaMask, 0.45f, lumaMask);
    }

    // Multiplicative luminance grain:
    // pixel *= (1 + noise * sigma * mask)
    cv::Mat gain = 1.0f + noiseL.mul(lumaMask) * sigmaLumaBase;
    cv::min(gain, 1.35f, gain);
    cv::max(gain, 0.65f, gain);

    ch[0] = ch[0].mul(gain);
    ch[1] = ch[1].mul(gain);
    ch[2] = ch[2].mul(gain);

    // Tiny chroma grain (subtle, different per channel)
    ch[0] += noiseC1.mul(lumaMask) * (sigmaChromaBase * maxVal);
    ch[2] += noiseC2.mul(lumaMask) * (sigmaChromaBase * maxVal);
    ch[1] += (0.5f * (noiseC1 + noiseC2)).mul(lumaMask) * (0.6f * sigmaChromaBase * maxVal);

    cv::merge(ch, f);

    // Clamp and convert back
    cv::min(f, maxVal, f);
    cv::max(f, 0.0f, f);

    f.convertTo(imgBgr, type);
}

// ----------------------------------------------------------
// Utility
// ----------------------------------------------------------
static inline float leakEnvelope(int frameIndex, int startFrame, int length)
{
    if (length <= 0) return 0.0f;
    const int t = frameIndex - startFrame;
    if (t < 0 || t >= length) return 0.0f;

    const float x = float(t) / float(std::max(1, length - 1)); // 0..1

    // Fast rise, slower tail
    if (x < 0.18f) {
        return x / 0.18f; // 0..1
    } else {
        const float d = (x - 0.18f) / 0.82f;
        return std::max(0.0f, 1.0f - d); // 1..0
    }
}

void DevelopThread::applySuper8LightLeak(cv::Mat &imgBgr)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }

    if (!p.enabled) return;

    const float s = clamp02(p.lightLeak);   // 0..1
    if (s <= 0.0001f) return;

    const int inType = imgBgr.type();
    const float maxVal =
        (inType == CV_8UC3)  ? 255.0f :
            (inType == CV_16UC3) ? 65535.0f : 0.0f;
    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    const int w = f.cols;
    const int h = f.rows;
    const int frameIndex = std::max(0, m_currentFrameIndex);

    // ------------------------------------------------------------
    // RANDOM TIMING, LESS OFTEN:
    // We test for a burst start every 4 frames.
    // Chance is tuned to roughly ~1 burst per ~2 sec average at 18 fps,
    // but randomized in time.
    // No extra state variables needed.
    // ------------------------------------------------------------
    const int startStep = 4;         // candidate start every 4 frames
    const int lookback  = 64;        // search recent possible starts
    const int fpsHint   = 18;        // expected playback fps feel

    auto hash32 = [](uint32_t x) -> uint32_t {
        x ^= x >> 16;
        x *= 0x7feb352dU;
        x ^= x >> 15;
        x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    };

    bool burstActive = false;
    int  burstStart = 0;
    int  burstDuration = 0;
    int  mode = 0;          // 0=left, 1=right, 2=center
    float strengthJitter = 1.0f;
    uint32_t burstSeed = 0;

    // Find most recent active burst start in the recent past
    const int startScan = std::max(0, frameIndex - lookback);
    for (int cand = frameIndex - (frameIndex % startStep); cand >= startScan; cand -= startStep)
    {
        uint32_t h0 = hash32(uint32_t(cand) * 9781u + 0xA53u);

        // About 1 in 9 candidate blocks start a burst:
        // 18 fps / 4 = 4.5 checks per sec => 1 burst about every ~2 sec average
        const bool startsHere = ((h0 % 9u) == 0u);
        if (!startsHere)
            continue;

        burstSeed = hash32(uint32_t(cand) * 1237u + 0xBEEF1234u);

        // Random duration: ~9..22 frames
        burstDuration = 6 + int(burstSeed % 10u);

        if (frameIndex < cand || frameIndex >= cand + burstDuration)
            continue;

        burstActive = true;
        burstStart = cand;

        mode = int((burstSeed >> 8) % 3u); // 0=left,1=right,2=center

        // modest random strength variation
        strengthJitter = 0.80f + 0.40f * float((burstSeed >> 12) & 1023u) / 1023.0f;
        break;
    }

    if (!burstActive)
        return;

    // ------------------------------------------------------------
    // Burst envelope
    // ------------------------------------------------------------
    const int localFrame = frameIndex - burstStart;
    const float u = (burstDuration > 1)
                        ? float(localFrame) / float(burstDuration - 1)
                        : 0.0f;

    // Smooth in/out
    const float env = std::sin(3.14159265f * u);

    // subtle breathing
    const float breathe = 0.94f + 0.10f * std::sin(6.2831853f * u);

    // Slider now strongly controls overall intensity too
    const float A = std::clamp((0.28f + 1.55f * s) * strengthJitter * env * breathe,
                               0.0f, 2.2f);

    cv::RNG rng(burstSeed);

    // ------------------------------------------------------------
    // Build 1D horizontal leak profile
    // Slider strongly affects WIDTH/SPREAD now.
    // At max slider, leak can fill almost/all of the frame.
    // ------------------------------------------------------------
    cv::Mat leak1D(1, w, CV_32F, cv::Scalar(0));

    if (mode == 0 || mode == 1)
    {
        const bool fromLeft = (mode == 0);

        // Wider with higher slider:
        // at s=0 small edge leak, at s=1 it can spread almost fully.
        const float edgeDecay  = (0.06f + 0.52f * s) * rng.uniform(0.92f, 1.08f);
        const float bandCenter = rng.uniform(0.04f, 0.14f + 0.55f * s);
        const float bandWidth  = (0.05f + 0.55f * s) * rng.uniform(0.90f, 1.10f);

        for (int x = 0; x < w; ++x)
        {
            float xn = float(x) / float(std::max(1, w - 1)); // 0..1
            if (!fromLeft) xn = 1.0f - xn;

            const float edge = std::exp(-xn / std::max(0.0001f, edgeDecay));
            const float band = std::exp(-0.5f * ((xn - bandCenter) * (xn - bandCenter)) /
                                        std::max(0.0001f, bandWidth * bandWidth));

            // At high slider, allow a much broader wash over the whole frame
            const float broad = std::exp(-0.5f * (xn * xn) /
                                         std::max(0.0001f, (0.18f + 0.95f * s) * (0.18f + 0.95f * s)));

            leak1D.at<float>(0, x) = 0.82f * edge + 0.72f * band + 0.60f * s * broad;
        }
    }
    else
    {
        // CENTER leak
        const float center = rng.uniform(0.34f, 0.66f);

        // At max slider, center leak can fill almost the full frame
        const float width1 = (0.07f + 0.28f * s) * rng.uniform(0.94f, 1.08f);
        const float width2 = (0.14f + 0.85f * s) * rng.uniform(0.94f, 1.08f);

        for (int x = 0; x < w; ++x)
        {
            const float xn = float(x) / float(std::max(1, w - 1));

            const float core =
                std::exp(-0.5f * ((xn - center) * (xn - center)) /
                         std::max(0.0001f, width1 * width1));

            const float glow =
                std::exp(-0.5f * ((xn - center) * (xn - center)) /
                         std::max(0.0001f, width2 * width2));

            // Extra very broad component so max slider can wash the whole image
            const float wash =
                std::exp(-0.5f * ((xn - center) * (xn - center)) /
                         std::max(0.0001f, (0.30f + 1.05f * s) * (0.30f + 1.05f * s)));

            leak1D.at<float>(0, x) = 0.95f * core + 0.55f * glow + 0.70f * s * wash;
        }
    }

    double minv = 0.0, maxv1 = 1.0;
    cv::minMaxLoc(leak1D, &minv, &maxv1);
    if (maxv1 > 1e-6)
        leak1D /= float(maxv1);

    cv::Mat mask;
    cv::repeat(leak1D, h, 1, mask);

    // Keep it smooth and even: no visible horizontal banding
    cv::GaussianBlur(mask, mask, cv::Size(0,0), 8.0, 2.5);

    // ------------------------------------------------------------
    // Warm orange-yellow tint
    // Slightly stronger than before
    // BGR order
    // ------------------------------------------------------------
    const float addB = 0.04f * A * maxVal;
    const float addG = 0.40f * A * maxVal;
    const float addR = 0.95f * A * maxVal;

    std::vector<cv::Mat> ch(3);
    cv::split(f, ch);

    ch[0] += mask * addB; // B
    ch[1] += mask * addG; // G
    ch[2] += mask * addR; // R

    cv::merge(ch, f);

    // Global flare lift:
    // with high slider + strong burst this can wash the whole frame
    if (A > 0.10f) {
        const float flareLift = 1.0f + (0.03f + 0.26f * s) * (A - 0.10f);
        f *= flareLift;
    }

    cv::min(f, maxVal, f);
    cv::max(f, 0.0f, f);
    f.convertTo(imgBgr, inType);
}

void DevelopThread::applyLeakMask(cv::Mat &imgBgr, int frameIndex, float A)
{
    Q_UNUSED(frameIndex);

    if (A <= 0.0001f) return;

    const int inType = imgBgr.type();
    const float maxVal =
        (inType == CV_8UC3)  ? 255.0f :
            (inType == CV_16UC3) ? 65535.0f : 0.0f;
    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    const int w = f.cols;
    const int h = f.rows;

    // Soft band from chosen edge
    cv::Mat leak(1, w, CV_32F);
    const bool fromLeft = (m_rng.uniform(0, 2) == 0);

    for (int x = 0; x < w; ++x)
    {
        float xn = float(x) / float(std::max(1, w - 1)); // 0..1
        if (!fromLeft) xn = 1.0f - xn;

        // Exponential falloff from the edge + mild center hump
        const float edge = std::exp(-xn / 0.12f);
        const float hump = std::exp(-0.5f * (xn - 0.08f) * (xn - 0.08f) / (0.10f * 0.10f));
        leak.at<float>(0, x) = 0.7f * edge + 0.5f * hump;
    }

    // Normalize
    double minv = 0.0, maxv = 1.0;
    cv::minMaxLoc(leak, &minv, &maxv);
    if (maxv > 1e-6) leak /= float(maxv);

    // Expand vertically
    cv::Mat mask;
    cv::repeat(leak, h, 1, mask);

    // Subtle vertical modulation
    cv::Mat vNoise(h, 1, CV_32F);
    cv::randn(vNoise, 0.0f, 1.0f);
    cv::GaussianBlur(vNoise, vNoise, cv::Size(1, 0), 5.0);

    cv::normalize(vNoise, vNoise, 0.85, 1.15, cv::NORM_MINMAX);
    cv::Mat vScale;
    cv::repeat(vNoise, 1, w, vScale);
    mask = mask.mul(vScale);

    cv::GaussianBlur(mask, mask, cv::Size(0,0), 2.0);

    // Warm leak in BGR
    const float addB = 0.06f * A * maxVal;
    const float addG = 0.24f * A * maxVal;
    const float addR = 0.60f * A * maxVal;

    std::vector<cv::Mat> ch(3);
    cv::split(f, ch);

    ch[0] += mask * addB;
    ch[1] += mask * addG;
    ch[2] += mask * addR;

    cv::merge(ch, f);

    cv::min(f, maxVal, f);
    cv::max(f, 0.0f, f);
    f.convertTo(imgBgr, inType);
}

void DevelopThread::applyScratches(cv::Mat &imgBgr, int frameIndex)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }

    if (!p.enabled) return;

    float s = clamp02(p.scratches); // 0..1 from slider
    if (s <= 0.0001f) return;

    const int inType = imgBgr.type();
    const float maxVal =
        (inType == CV_8UC3)  ? 255.0f :
            (inType == CV_16UC3) ? 65535.0f : 0.0f;

    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    const int w = f.cols;
    const int h = f.rows;

    // ---- deterministic RNG per short block so scratches persist across 2–3 frames ----
    const int blockSize = 1;
    const int phase     = 0;

    cv::RNG rng(uint64_t(frameIndex) * 1103515245u + 12345u);

    // ---- scratch count: let slider mainly control HOW MANY ----
    int count = int((w / 420.0f) * (0.20f + 3.20f * s));
    count = std::clamp(count, 0, std::max(1, w / 42));

    // slightly more frequent extra burst at higher settings
    if (rng.uniform(0, 100) < int(12 * s)) {
        count += rng.uniform(1, 4);
    }

    // Masks for dark scratches + optional blue tint scratches
    cv::Mat maskBlack(h, w, CV_32F, cv::Scalar(0));
    cv::Mat maskBlue (h, w, CV_32F, cv::Scalar(0));

    // Resolution-aware scratch thickness
    const float resScale = std::sqrt((w * h) / (640.0f * 480.0f));

    for (int i = 0; i < count; ++i)
    {
        // Mostly near vertical, but allow slight slant
        const int x0 = rng.uniform(0, w);
        const int slant = rng.uniform(-2, 3);
        const int xEnd  = std::clamp(x0 + slant, 0, w - 1);

        // Thickness: keep max thinner than before
        int thickness = 1
                        + ((rng.uniform(0, 100) < 12) ? 1 : 0);   // rarer 2px scratches
        thickness = std::max(1, (int)std::round(thickness * (0.70f + 0.35f * resScale)));
        thickness = std::min(thickness, 1); // hard cap: keep scratches thin

        // Strength per scratch
        const float strength = std::clamp(0.15f + 0.85f * rng.uniform(0.0f, 1.0f), 0.0f, 1.0f);
        float k = std::clamp(s * strength, 0.0f, 1.0f);

        // ---- slight breathing: small intensity wobble across the 2–3 persistent frames ----
        // Make it scratch-specific so all scratches don't pulse together.
        const float wobblePhase = rng.uniform(0.0f, 6.2831853f);
        const float t = (blockSize > 1) ? (float)phase / (float)(blockSize - 1) : 0.0f;

        // 3–8% breathing feels "film", not "effect"
        const float breatheAmp = 0.03f + 0.05f * s; // slider increases the feel slightly
        const float breathe = 1.0f + breatheAmp * std::sin(6.2831853f * t + wobblePhase);

        k = std::clamp(k * breathe, 0.0f, 1.0f);

        // Occasionally a bluish scratch (rare)
        const bool bluish = (rng.uniform(0, 100) < 50);

        // --- variable continuous length (no dashes) ---
        const float minLenFrac = 0.40f; // 40% of frame height
        const float maxLenFrac = 1.00f; // up to full height

        float lenFrac = minLenFrac + (maxLenFrac - minLenFrac) * rng.uniform(0.0f, 1.0f);

        // Bias: low slider -> more short scratches; high slider -> more long scratches
        lenFrac = std::clamp(lenFrac * (0.85f + 0.9f * s), minLenFrac, 1.0f);

        int lenPx = std::max(8, (int)std::round(lenFrac * (h - 1)));

        int y0 = rng.uniform(0, std::max(1, h - lenPx));
        int y1 = std::clamp(y0 + lenPx, 0, h - 1);

        // Draw one continuous segment
        cv::line(maskBlack, cv::Point(x0, y0), cv::Point(xEnd, y1), cv::Scalar(k), thickness, cv::LINE_AA);
        if (bluish)
            cv::line(maskBlue,  cv::Point(x0, y0), cv::Point(xEnd, y1), cv::Scalar(k), thickness, cv::LINE_AA);

        // Optional: soften ends a bit so cutoffs don't look "digital"
        const bool softenEnds = true;
        if (softenEnds)
        {
            const float capK = k * 0.7f;
            const int r = std::max(1, thickness);

            cv::circle(maskBlack, cv::Point(x0,  y0), r, cv::Scalar(capK), cv::FILLED, cv::LINE_AA);
            cv::circle(maskBlack, cv::Point(xEnd, y1), r, cv::Scalar(capK), cv::FILLED, cv::LINE_AA);

            if (bluish)
            {
                cv::circle(maskBlue, cv::Point(x0,  y0), r, cv::Scalar(capK), cv::FILLED, cv::LINE_AA);
                cv::circle(maskBlue, cv::Point(xEnd, y1), r, cv::Scalar(capK), cv::FILLED, cv::LINE_AA);
            }
        }
    }

    // Slight blur so scratches feel like they're "in the emulsion" not razor-sharp
    {
        const int ksz = (w >= 1600) ? 5 : 3;
        cv::GaussianBlur(maskBlack, maskBlack, cv::Size(ksz, ksz), 0.0, 0.0, cv::BORDER_REPLICATE);
        cv::GaussianBlur(maskBlue,  maskBlue,  cv::Size(ksz, ksz), 0.0, 0.0, cv::BORDER_REPLICATE);
    }

    // Apply: black scratches mostly darken; bluish scratches also tint slightly blue.
    const float blackGain = 0.85f;  // overall darkness

    // More blue-green / cyan emulsion-damage look
    const float cyanTint  = 0.7f;  // total tint strength
    const float cyanB     = 1.00f;  // strong blue
    const float cyanG     = 0.72f;  // much more green than before
    const float cyanR     = 0.03f;  // keep red very low

    std::vector<cv::Mat> ch(3);
    cv::split(f, ch);

    // Darken (all channels): c *= (1 - mask * blackGain)
    cv::Mat darkFactor = 1.0f - (maskBlack * blackGain);
    cv::min(darkFactor, 1.0f, darkFactor);
    cv::max(darkFactor, 0.10f, darkFactor);

    ch[0] = ch[0].mul(darkFactor);
    ch[1] = ch[1].mul(darkFactor);
    ch[2] = ch[2].mul(darkFactor);

    // Blue-green / cyan tint for emulsion-like scratches
    if (cv::countNonZero(maskBlue > 0.0005f) > 0)
    {
        cv::Mat bAdd = maskBlue * (cyanTint * cyanB * maxVal);
        cv::Mat gAdd = maskBlue * (cyanTint * cyanG * maxVal);
        cv::Mat rAdd = maskBlue * (cyanTint * cyanR * maxVal);

        ch[0] += bAdd; // B
        ch[1] += gAdd; // G
        ch[2] += rAdd; // R
    }

    cv::merge(ch, f);

    // Clamp and convert back
    cv::min(f, maxVal, f);
    cv::max(f, 0.0f, f);

    f.convertTo(imgBgr, inType);
}

void DevelopThread::drawSoftDust(cv::Mat& f32bgr, int cx, int cy, float radiusPx, float delta, float maxVal)
{
    // This version makes dust less "perfect circle":
    // - elliptic gaussian (different sigmaX/sigmaY)
    // - random rotation
    // - slight "lumpiness" modulation
    //
    // NOTE: delta can be negative (dark) or positive (bright halo)

    // We need a deterministic pseudo-random but we don't have rng here.
    // So derive a tiny hash from position to vary shape consistently.
    auto hash01 = [](int x, int y, int k) -> float {
        uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u + uint32_t(k) * 2246822519u;
        h = (h ^ (h >> 13)) * 1274126177u;
        h ^= (h >> 16);
        return (h & 0x00FFFFFF) / float(0x01000000); // [0,1)
    };

    const float uA = hash01(cx, cy, 1);
    const float uB = hash01(cx, cy, 2);
    const float uC = hash01(cx, cy, 3);

    // Ellipse axes
    float sx = std::max(0.16f, radiusPx * (0.48f + 0.62f * uA));
    float sy = std::max(0.16f, radiusPx * (0.48f + 0.62f * uB));

    // Rotation angle [0..2pi)
    const float ang = 6.28318530718f * uC;
    const float ca = std::cos(ang);
    const float sa = std::sin(ang);

    // Kernel radius based on max axis
    const float rMax = std::max(sx, sy);
    const int rad = int(std::ceil(rMax * 3.0f));

    for (int dy = -rad; dy <= rad; ++dy) {
        const int y = cy + dy;
        if ((unsigned)y >= (unsigned)f32bgr.rows) continue;

        for (int dx = -rad; dx <= rad; ++dx) {
            const int x = cx + dx;
            if ((unsigned)x >= (unsigned)f32bgr.cols) continue;

            // rotate (dx,dy) into ellipse space
            const float rx = ca * dx + sa * dy;
            const float ry = -sa * dx + ca * dy;

            const float ex = (rx * rx) / (2.0f * sx * sx);
            const float ey = (ry * ry) / (2.0f * sy * sy);

            float w = std::exp(-(ex + ey));

            // "Lumpiness": modulate weight a bit so it isn't perfectly smooth
            // Keep subtle so it still looks like soft dust, not noise.
            const float n = hash01(x, y, 9);              // stable per-pixel
            w *= (0.85f + 0.30f * n);

            cv::Vec3f& p = f32bgr.at<cv::Vec3f>(y, x);

            // If delta is negative (dark dust), don't "ink" to black.
            // Scale the darkening by local brightness so it behaves more like attenuation.
            // If delta is positive (bright pinholes/halo), keep unchanged.
            float d = delta * w;
            if (d < 0.0f) {
                const float lum = (p[0] + p[1] + p[2]) / (3.0f * maxVal);   // 0..1
                const float k   = 0.25f + 0.75f * lum;                     // 0.25..1.0
                d *= k;
            }

            p[0] = std::clamp(p[0] + d, 0.0f, maxVal);
            p[1] = std::clamp(p[1] + d, 0.0f, maxVal);
            p[2] = std::clamp(p[2] + d, 0.0f, maxVal);

        }
    }
}

static inline void drawDustFiber(cv::Mat& f32bgr, int x0, int y0, int len, float angleRad, float delta, float maxVal)
{
    const float ca = std::cos(angleRad);
    const float sa = std::sin(angleRad);

    // Draw a short soft line (len 2..10 px)
    for (int i = 0; i < len; ++i) {
        int x = int(std::round(x0 + ca * i));
        int y = int(std::round(y0 + sa * i));
        if ((unsigned)x >= (unsigned)f32bgr.cols || (unsigned)y >= (unsigned)f32bgr.rows) continue;

        // small soft stamp per segment
        cv::Vec3f& p = f32bgr.at<cv::Vec3f>(y, x);
        p[0] = std::clamp(p[0] + delta, 0.0f, maxVal);
        p[1] = std::clamp(p[1] + delta, 0.0f, maxVal);
        p[2] = std::clamp(p[2] + delta, 0.0f, maxVal);
    }
}

void DevelopThread::applyDust(cv::Mat& imgBgr, int frameIndex)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }
    if (!p.enabled) return;

    const float amount = clamp02(p.dust); // 0..1 from slider
    if (amount <= 0.0001f) return;

    const int inType = imgBgr.type();
    const float maxVal =
        (inType == CV_8UC3)  ? 255.0f :
            (inType == CV_16UC3) ? 65535.0f : 0.0f;

    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    const int w = f.cols;
    const int h = f.rows;

    // ---- RNG (deterministic per frame) ----
    std::mt19937 rng(uint32_t(frameIndex * 11027u + 12345u));

    std::uniform_int_distribution<int> ix(0, w - 1);
    std::uniform_int_distribution<int> iy(0, h - 1);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> angle01(0.0f, 6.28318530718f);

    // Radius grows with slider, but not absurdly
    std::uniform_real_distribution<float> radSmall(0.55f, 1.10f + 1.70f * amount);
    std::uniform_real_distribution<float> radLarge(1.00f, 2.10f + 3.60f * amount);

    // ---- Density model ----
    // More specs as slider increases
    int count = int((double(w) * double(h) / 90000.0) * double(0.20f + 4.40f * amount));
    count = std::clamp(count, 0, 5000);

    // 50/50 bright vs dark
    const float brightProb = 0.50f;

    // Similar strength both ways
    const float darkDeltaBase   = -0.26f * amount * maxVal;
    const float brightDeltaBase =  0.22f * amount * maxVal;

    std::uniform_real_distribution<float> darkScale(0.75f, 1.15f);
    std::uniform_real_distribution<float> brightScale(0.75f, 1.10f);

    // More fibers/hairs when slider goes up
    int fiberCount = 0;
    if (amount > 0.03f) {
        fiberCount = int((w * h / 220000.0) * (0.25f + 4.50f * amount));
        fiberCount = std::clamp(fiberCount, 0, 120);
    }

    // Specks / blobs / blotches
    for (int i = 0; i < count; ++i)
    {
        const int x = ix(rng);
        const int y = iy(rng);

        const bool bright = (u01(rng) < brightProb);

        float delta = bright
                          ? (brightDeltaBase * brightScale(rng))
                          : (darkDeltaBase   * darkScale(rng));

        const float t = u01(rng);

        if (t < 0.58f)
        {
            // small speck
            const float r = radSmall(rng);
            drawSoftDust(f, x, y, r, delta, maxVal);
        }
        else if (t < 0.86f)
        {
            // irregular blotch made of a few overlapping blobs
            const int n = 2 + int(u01(rng) * 3.0f); // 2..4 sub-blobs
            const float baseR = radLarge(rng);

            for (int k = 0; k < n; ++k)
            {
                const float a = angle01(rng);
                const float d = (0.15f + 0.85f * u01(rng)) * baseR;

                const int bx = int(std::round(x + std::cos(a) * d));
                const int by = int(std::round(y + std::sin(a) * d));

                const float rr = baseR * (0.50f + 0.55f * u01(rng));
                const float dd = delta * (0.80f + 0.30f * u01(rng));

                drawSoftDust(f, bx, by, rr, dd, maxVal);
            }
        }
        else
        {
            // medium blob
            const float r = radLarge(rng);
            drawSoftDust(f, x, y, r, delta, maxVal);
        }
    }

    // Fibers / hairs
    if (fiberCount > 0)
    {
        std::uniform_int_distribution<int> lenDist(4, int(10 + 34 * amount));
        std::uniform_real_distribution<float> angDist(0.0f, 6.28318530718f);

        for (int i = 0; i < fiberCount; ++i)
        {
            const int x0 = ix(rng);
            const int y0 = iy(rng);
            const int len = lenDist(rng);
            const float ang = angDist(rng);

            const bool bright = (u01(rng) < brightProb);
            const float delta = bright
                                    ? (brightDeltaBase * 0.50f * brightScale(rng))
                                    : (darkDeltaBase   * 0.90f * darkScale(rng));

            drawDustFiber(f, x0, y0, len, ang, delta, maxVal);
        }
    }

    cv::min(f, maxVal, f);
    cv::max(f, 0.0f, f);
    f.convertTo(imgBgr, inType);
}
