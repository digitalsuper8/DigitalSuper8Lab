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

inline void applyLogC3(const cv::Mat& lin, cv::Mat& out)
{
    CV_Assert(lin.type() == CV_32FC3);
    out.create(lin.size(), CV_32FC3);

    lin.forEach<cv::Vec3f>([&](const cv::Vec3f& p, const int pos[]){
        cv::Vec3f v;
        for (int k=0;k<3;k++)
            v[k] = toLogC3(p[k]);
        out.at<cv::Vec3f>(pos[0], pos[1]) = v;
    });
}



// Eenvoudige white-balance helper: per kanaal vermenigvuldigen
static inline void applyWhiteBalance(cv::Mat &img, float wbR, float wbG, float wbB)
{
    CV_Assert(img.type() == CV_32FC3);
    img.forEach<cv::Vec3f>([&](cv::Vec3f &p, const int*){
        p[2] *= wbR;  // BGR volgorde: [0]=B, [1]=G, [2]=R
        p[1] *= wbG;
        p[0] *= wbB;
    });
}
// Approx. sRGB → ACEScg matrix (voor RGB volgorde)
const cv::Matx33f kSRGB_to_ACEScg(
    0.613097f, 0.339523f, 0.047380f,
    0.070193f, 0.916354f, 0.013453f,
    0.020615f, 0.109569f, 0.869815f
    );

inline void applySRGBtoACEScg(cv::Mat &img)
{
    CV_Assert(img.type() == CV_32FC3);

    img.forEach<cv::Vec3f>([](cv::Vec3f &p, const int*)
                           {
                               // p is BGR, maak er RGB van
                               cv::Vec3f rgb(p[2], p[1], p[0]);

                               // apply sRGB→ACEScg in RGB volgorde
                               cv::Vec3f aces = kSRGB_to_ACEScg * rgb;

                               // terugschrijven als BGR
                               p[0] = aces[2]; // B
                               p[1] = aces[1]; // G
                               p[2] = aces[0]; // R
                           });
}
// Very simple ACEScct-like log encoding: linear ACEScg -> log-ish ACEScct
static float simpleACEScctChannel(float x)
{
    // clamp to small positive to avoid log(0)
    x = std::max(x, 1e-6f);

    // a gentle log2 curve, scaled into ~0..1 range
    float y = (std::log2(x + 0.02f) + 9.0f) / 17.0f;
    //float y = 0;
    return std::min(std::max(y, 0.0f), 1.0f);
}

static void applyACEScct(const cv::Mat &acesCG, cv::Mat &acesCCT)
{
    CV_Assert(acesCG.type() == CV_32FC3);

    acesCCT.create(acesCG.size(), CV_32FC3);

    for (int y = 0; y < acesCG.rows; ++y)
    {
        const cv::Vec3f* src = acesCG.ptr<cv::Vec3f>(y);
        cv::Vec3f* dst       = acesCCT.ptr<cv::Vec3f>(y);

        for (int x = 0; x < acesCG.cols; ++x)
        {
            dst[x][0] = simpleACEScctChannel(src[x][0]); // B
            dst[x][1] = simpleACEScctChannel(src[x][1]); // G
            dst[x][2] = simpleACEScctChannel(src[x][2]); // R
        }
    }
}

inline float clamp01(float x)
{
    return std::max(0.0f, std::min(1.0f, x));
}

// ------------------------------------------------------------------
// Filmic curve (Hable-style)
// Input:  scene-linear [0, +inf[, verwachten ~[0, 1] normaal
// Output: display-ish [0, 1]
// ------------------------------------------------------------------
inline float filmicHable(float x)
{
    // small toe offset to avoid crushing very dark low values
    x = std::max(x - 0.004f, 0.0f);

    const float A = 0.22f;
    const float B = 0.30f;
    const float C = 0.10f;
    const float D = 0.20f;
    const float E = 0.01f;
    const float F = 0.30f;

    const float num = x * (A * x + C * B) + D * E;
    const float den = x * (A * x + B)     + D * F;

    float y = num / den - E / F;
    return clamp01(y);
}

// ------------------------------------------------------------------
// Reinhard tone map with simple gamma
// Input:  scene-linear [0, +inf[
// Output: display-ish [0, 1]
// ------------------------------------------------------------------
inline float reinhardGamma(float x, float gamma = 2.2f)
{
    x = std::max(x, 0.0f);
    // classic Reinhard: x / (1 + x)
    float y = x / (1.0f + x);

    // optional gamma for some extra contrast
    y = std::pow(y, 1.0f / gamma);
    return clamp01(y);
}

// Apply filmic Hable to a 32f RGB image in-place
void applyFilmicHable(cv::Mat &img)
{
    CV_Assert(img.type() == CV_32FC3);

    img.forEach<cv::Vec3f>([](cv::Vec3f &p, const int*) {
        for (int c = 0; c < 3; ++c)
            p[c] = filmicHable(p[c]);
    });
}

// Apply Reinhard+gamma to a 32f RGB image in-place
void applyReinhardGamma(cv::Mat &img, float gamma = 2.2f)
{
    CV_Assert(img.type() == CV_32FC3);

    img.forEach<cv::Vec3f>([gamma](cv::Vec3f &p, const int*) {
        for (int c = 0; c < 3; ++c)
            p[c] = reinhardGamma(p[c], gamma);
    });
}



// ACES AP1 → sRGB/Rec.709 matrix
static const cv::Matx33f ACES_AP1_to_sRGB(
    1.70505f, -0.62179f, -0.08326f,
    -0.13026f,  1.14080f, -0.01055f,
    -0.02400f, -0.12897f,  1.15297f
    );

inline void applySCurvesFloat(cv::Mat& img,
                              const ushort* lutR,
                              const ushort* lutG,
                              const ushort* lutB)
{
    CV_Assert(img.type() == CV_32FC3);

    const float inv4095 = 1.0f / 4095.0f;

    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec3f c = row[x];   // BGR

            // Convert [0,1] → [0,4095] index
            int iB = static_cast<int>(c[0] * 4095.0f + 0.5f); // B
            int iG = static_cast<int>(c[1] * 4095.0f + 0.5f); // G
            int iR = static_cast<int>(c[2] * 4095.0f + 0.5f); // R

            if (iB < 0) iB = 0; else if (iB > 4095) iB = 4095;
            if (iG < 0) iG = 0; else if (iG > 4095) iG = 4095;
            if (iR < 0) iR = 0; else if (iR > 4095) iR = 4095;

            // Correct channel → LUT mapping (BGR image, RGB LUTs)
            c[0] = static_cast<float>(lutB[iB]) * inv4095;  // B channel
            c[1] = static_cast<float>(lutG[iG]) * inv4095;  // G channel
            c[2] = static_cast<float>(lutR[iR]) * inv4095;  // R channel

            row[x] = c;
        }
    }
}


inline float ACES_RRT(float v)
{
    // RRT sweeteners – gentle shoulder + toe
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    return (v * (a * v + b)) / (v * (c * v + d) + e);
}

inline cv::Vec3f ACES_RRT_RGB(const cv::Vec3f& rgb)
{
    return cv::Vec3f(
        ACES_RRT(rgb[0]),
        ACES_RRT(rgb[1]),
        ACES_RRT(rgb[2])
        );
}

inline void applyACES(cv::Mat& img)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec3f c = row[x];

            // RRT
            c = ACES_RRT_RGB(c);

            // AP1 → sRGB
            c = ACES_AP1_to_sRGB * c;

            // clamp
            c[0] = std::min(std::max(c[0], 0.0f), 1.0f);
            c[1] = std::min(std::max(c[1], 0.0f), 1.0f);
            c[2] = std::min(std::max(c[2], 0.0f), 1.0f);

            row[x] = c;
        }
    }
}


// Apply colormatrix
inline void applyColorMatrix(cv::Mat& img, const cv::Matx33f& M)
{
    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            row[x] = M * row[x];   // matrix multiply
        }
    }
}
// Apply exposure

inline void applyExposure(cv::Mat& img, float ev)
{
    CV_Assert(img.type() == CV_32FC3);
    if (std::abs(ev) < 1e-4f) return;

    float gain = std::pow(2.0f, ev);   // +1 EV = ×2, -1 EV = ×0.5
    img *= gain;
}

// Clamp float image to [0, 1]
inline void clamp01(cv::Mat& img)
{
    CV_Assert(img.type() == CV_32FC3);
    cv::max(img, 0.0f, img);
    cv::min(img, 1.0f, img);
}

// Apply global contrast & brightness in linear space
// Contrast: 100 = neutral, 200 = x2, 50 = x0.5
// Bright:   roughly "offset" with 255 meaning +1.0 in linear space
inline void applyBrightnessContrast(cv::Mat& img, float contrast, float bright)
{
    CV_Assert(img.type() == CV_32FC3);

    const float c = contrast / 100.0f;
    const float b = bright   / 255.0f;

    cv::Mat tmp;
    // alpha = c, beta = b  (applies to all channels equally)
    img.convertTo(tmp, CV_32FC3, c, b);
    img = tmp;
}

// Apply saturation in linear space using luma/chroma model
// Sat: 100 = neutral, 0 = grayscale, 200 = double saturation
inline void applySaturation(cv::Mat& img, float satPercent)
{
    CV_Assert(img.type() == CV_32FC3);

    const float s = satPercent / 100.0f;
    if (std::abs(s - 1.0f) < 1e-3f)
        return;

    // Luma coefficients, but on BGR order (OpenCV default)
    // Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec3f cpx = row[x];
            const float Y =
                0.0722f * cpx[0] +  // B
                0.7152f * cpx[1] +  // G
                0.2126f * cpx[2];   // R

            cv::Vec3f grey(Y, Y, Y);
            row[x] = grey + (cpx - grey) * s;
        }
    }
}


// Apply per-channel gains in BGR order
inline void applyChannelGains(cv::Mat& img, float blue, float green, float red)
{
    CV_Assert(img.type() == CV_32FC3);
    std::vector<cv::Mat> ch;
    cv::split(img, ch);         // B, G, R
    ch[0] *= blue;
    ch[1] *= green;
    ch[2] *= red;
    cv::merge(ch, img);
}

// Simple highlight compression / HDR-ish shoulder
inline void applySimpleHDR(cv::Mat& img, bool enabled)
{
    if (!enabled)
        return;

    CV_Assert(img.type() == CV_32FC3);

    for (int y = 0; y < img.rows; ++y)
    {
        cv::Vec3f* row = img.ptr<cv::Vec3f>(y);
        for (int x = 0; x < img.cols; ++x)
        {
            cv::Vec3f& c = row[x];
            for (int k = 0; k < 3; ++k)
            {
                float v = c[k];
                // Soft shoulder: keeps blacks nearly unchanged, compresses highlights
                c[k] = v / (v + 0.5f);
            }
        }
    }
}

// -- Helpers out of ACES spec --
//static inline float saturate(float x) { return std::max(0.f, std::min(1.f, x)); }

static inline float saturate(float v)
{
    return std::max(0.0f, std::min(1.0f, v));
}

// Heel eenvoudige "filmic shoulder" + gamma 2.2 naar Rec709
static inline float filmicCurve(float x)
{
    // clamp input
    x = saturate(x);

    // simpele Reinhard-achtige compressie
    float y = x / (x + 0.155f);

    // gamma voor Rec.709-ish display
    y = std::pow(y, 1.0f / 2.2f);

    return saturate(y);
}

static void applyRRT_ODT_Rec709(const cv::Mat &src, cv::Mat &dst)
{
    CV_Assert(src.type() == CV_32FC3);

    dst.create(src.size(), CV_32FC3);

    for (int y = 0; y < src.rows; ++y)
    {
        const cv::Vec3f* s = src.ptr<cv::Vec3f>(y);
        cv::Vec3f*       d = dst.ptr<cv::Vec3f>(y);

        for (int x = 0; x < src.cols; ++x)
        {
            cv::Vec3f p = s[x];

            d[x][0] = filmicCurve(p[0]); // B
            d[x][1] = filmicCurve(p[1]); // G
            d[x][2] = filmicCurve(p[2]); // R
        }
    }
}

} // anonymous namespace

static float plannedLeakEnvelope(int frameIndex, int F)
{
    int d = frameIndex - F;

    // ramp up: F-5..F-1
    if (d >= -5 && d <= -1) return float(d + 6) / 5.0f;        // 0.2..1.0

    // full: F..F+4
    if (d >= 0 && d <= 4) return 1.0f;

    // ramp down: F+5..F+9
    if (d >= 5 && d <= 9) return 1.0f - float(d - 4) / 5.0f;  // 0.8..0.0

    return 0.0f;
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

    // new code for 8 bit development:
    Mat lut8(1, 256.0, CV_8UC1);
    Mat lutS(1, 256.0, CV_8UC1);
    uchar * ptr = lut8.ptr();
    uchar * ptrS = lutS.ptr();
    for( int i = 0; i < 256; i++ ){
        ptr[i]  = (((234.0-2.0)*0.432699)*log10(((i-(2.0))/(31.0-2.0))+0.037584) + (0.616596*255));
        ptrS[i] = ((1/(1+ exp((double)(-(8.0/255.0)*((i-153)-0))))) * 255.0);
    }
    lut8_array = lut8.clone();
    lutS_array = lutS.clone();

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
        timer_->deleteLater();
        timer_ = nullptr;
    }
}

void DevelopThread::init()
{
    // This runs in the worker thread (connected from QThread::started)
    timer_ = new QTimer(); // no parent: ensure it lives in worker thread
    timer_->setTimerType(Qt::PreciseTimer);
    QObject::connect(timer_, &QTimer::timeout, this, &DevelopThread::onTick);
}



// ---------- New playback control slots ----------

void DevelopThread::setFps(double fps)         { playbackFps_ = (fps > 0.0 ? fps : 24.0); }
void DevelopThread::setRange(int f, int l)     { rangeFirst_ = std::max(1, f); rangeLast_ = std::max(rangeFirst_, l); current_ = std::clamp(current_, rangeFirst_, rangeLast_); }
void DevelopThread::setLooping(bool e)         { looping_ = e; }


void DevelopThread::play()
{
    // ensure timer exists and start from worker thread
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
    // optional immediate refresh:
  //  onDevelopFrame(current_);
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
    cancelRequested_.store(true, std::memory_order_relaxed);
    renderingBusy = false;
}


// ---------- Playback internals ----------

void DevelopThread::startPlaybackTimer()
{
    const int intervalMs = std::max(1, qRound(1000.0 / playbackFps_));
    if (!timer_) return;
    QMetaObject::invokeMethod(timer_, [this, intervalMs]{
        wallClock_.restart();
        timer_->start(intervalMs);
    }, Qt::QueuedConnection);
}

void DevelopThread::stopPlaybackTimer()
{
    if (!timer_) return;
    QMetaObject::invokeMethod(timer_, [this]{ timer_->stop(); }, Qt::QueuedConnection);
}

void DevelopThread::onTick()
{
    if (processingBusy_) return;
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

    try
    {
        ImageDeveloped = false;

        char in_name[256];
        std::snprintf(in_name, sizeof(in_name),
                      "%s/images%d/XiCapture%03d.pgm", work_path, Roll, i);

        cv::Mat imgBAY = cv::imread(in_name, cv::IMREAD_ANYDEPTH);
        if (imgBAY.empty() || imgBAY.channels() != 1)
        {
            qWarning() << "[DevelopThread] Invalid RAW Bayer image:" << in_name;
            return;
        }

        // ----------------------------------------------------------
        // 1) Exposure on RAW Bayer before demosaic
        // ----------------------------------------------------------
        cv::Mat exposedBayer;
        if (std::abs(ExposureEV) > 1e-4f)
        {
            const double gain = std::pow(2.0, static_cast<double>(ExposureEV));

            if (imgBAY.depth() == CV_16U)
                imgBAY.convertTo(exposedBayer, CV_16U, gain, 0.0);
            else
                imgBAY.convertTo(exposedBayer, CV_8U, gain, 0.0);
        }
        else
        {
            exposedBayer = imgBAY;
        }

        // ----------------------------------------------------------
        // 2) Preview-style base path
        //    12/16-bit Bayer -> 8-bit Bayer first, just like previewthread
        // ----------------------------------------------------------
        cv::Mat bayer8;
        if (exposedBayer.depth() == CV_16U)
        {
            // 12-bit data in 16-bit container -> 8-bit display domain
            exposedBayer.convertTo(bayer8, CV_8UC1, (1.0 / 16.0));
        }
        else
        {
            exposedBayer.convertTo(bayer8, CV_8UC1);
        }

        // ----------------------------------------------------------
        // 3) Demosaic
        // ----------------------------------------------------------
        cv::Mat img8;
        cv::cvtColor(bayer8, img8, cv::COLOR_BayerGB2BGR, 3);

        // ----------------------------------------------------------
        // 4) Flip to match previewthread / cartridge orientation
        // ----------------------------------------------------------
        cv::flip(img8, img8, -1);

        // ----------------------------------------------------------
        // 5) Base S-Log-ish look + sigmoid curve
        // ----------------------------------------------------------
        cv::LUT(img8, lut8_array, img8);
        cv::LUT(img8, lutS_array, img8);

        // ----------------------------------------------------------
        // 6) Optional extra user S-curves
        // ----------------------------------------------------------
        if (filmlook)
        {
            cv::Mat f32;
            img8.convertTo(f32, CV_32FC3, 1.0 / 255.0);

            applySCurvesFloat(f32,
                              lutSCurveRed,
                              lutSCurveGreen,
                              lutSCurveBlue);

            cv::max(f32, 0.0f, f32);
            cv::min(f32, 1.0f, f32);

            f32.convertTo(img8, CV_8UC3, 255.0);
        }

        // ----------------------------------------------------------
        // 7) Brightness / Saturation / Contrast in HSV
        // ----------------------------------------------------------
        {
            cv::Mat imgHSV;
            std::vector<cv::Mat> channels;

            cv::cvtColor(img8, imgHSV, cv::COLOR_BGR2HSV);
            cv::split(imgHSV, channels);

            channels[2].convertTo(channels[2], -1, (Contrast / 100.0), Bright);
            channels[1].convertTo(channels[1], -1, (Sat / 100.0), 0.0);

            cv::merge(channels, imgHSV);
            cv::cvtColor(imgHSV, img8, cv::COLOR_HSV2BGR);
        }

        // ----------------------------------------------------------
        // 8) RGB gain trim
        // ----------------------------------------------------------
        {
            std::vector<cv::Mat> channels;
            cv::split(img8, channels); // B, G, R

            channels[0].convertTo(channels[0], -1, Blue,  0.0);
            channels[1].convertTo(channels[1], -1, Green, 0.0);
            channels[2].convertTo(channels[2], -1, Red,   0.0);

            cv::merge(channels, img8);
        }

        // ----------------------------------------------------------
        // 9) Output bit depth
        // ----------------------------------------------------------
        if (bitdepth >= 16)
        {
            // 8-bit display image packed into 16-bit container for preview/export path
            img8.convertTo(imgBGR, CV_16UC3, 257.0);
        }
        else
        {
            imgBGR = img8;
        }

        // ----------------------------------------------------------
        // 10) Super 8 development effects (final image domain)
        // ----------------------------------------------------------
        applySuper8LightLeak(imgBGR);
        applySuper8Grain(imgBGR);
        applyScratches(imgBGR, i);
        applyDust(imgBGR, i);

        ImageDeveloped = true;
        emit FrameDeveloped(i);
    }
    catch (const cv::Exception& e)
    {
        qWarning() << "[DevelopThread] OpenCV EX:" << e.what();
        ImageDeveloped = false;
    }
}



void DevelopThread::calc_LutCurve()
{
    // 1) Decide the pivot in normalized [0..1] based on grading mode
    float defaultPivot =
        (m_gradingMode == GradingMode::ACES) ? 0.41f : 0.50f;

    // If user has not set pivot yet (m_sCurvePivot < 0),
    // use the default for the current mode.
    float pivotNorm = (m_sCurvePivot >= 0.0f) ? m_sCurvePivot : defaultPivot;

    // Clamp to a safe range
    pivotNorm = std::clamp(pivotNorm, 0.0f, 1.0f);

    // Convert normalized pivot to 0..4095 scale
    const double pivot = static_cast<double>(pivotNorm) * 4095.0;

    auto buildCurve = [&](double strength, ushort* lut)
    {
        // 0 → identity (no curve at all, no grey wash)
        if (strength <= 0.0) {
            for (int i = 0; i < 4096; ++i) {
                lut[i] = static_cast<ushort>(i);
            }
            return;
        }

        // This is the "base" curve level you liked (8 was your sweet spot)
        const double baseCurve = 8.0;

        for (int i = 0; i < 4096; ++i)
        {
            double x = static_cast<double>(i);
            double y = 0.0;

            if (strength < baseCurve)
            {
                // ---- LOW-STRENGTH REGION: blend identity ↔ S-curve(at 8) ----
                double kBase = baseCurve / 4095.0;
                double sBase = 1.0 / (1.0 + std::exp(-kBase * (x - pivot)));
                double hard  = sBase * 4095.0;

                double ident = x;
                double t     = strength / baseCurve;  // 0..1

                y = (1.0 - t) * ident + t * hard;
            }
            else
            {
                // ---- ORIGINAL BEHAVIOUR FOR "REAL" VALUES (>= 8) ----
                double k = strength / 4095.0;
                double s = 1.0 / (1.0 + std::exp(-k * (x - pivot)));
                y = s * 4095.0;
            }

            if (y < 0.0)   y = 0.0;
            if (y > 4095.) y = 4095.;

            lut[i] = static_cast<ushort>(std::lround(y));
        }
    };

    buildCurve(CurveRed,   lutSCurveRed);
    buildCurve(CurveGreen, lutSCurveGreen);
    buildCurve(CurveBlue,  lutSCurveBlue);

    // Emit a copy that the UI can safely use for painting
    QVector<quint16> r(4096), g(4096), b(4096);
    for (int i = 0; i < 4096; ++i) {
        r[i] = lutSCurveRed[i];
        g[i] = lutSCurveGreen[i];
        b[i] = lutSCurveBlue[i];
    }
    emit sCurvesUpdated(r, g, b);
}


void DevelopThread::setGradingMode(int mode)
{
    // 0 = ACES, 1 = Log
    if (mode == 0)
        m_gradingMode = GradingMode::ACES;
    else
        m_gradingMode = GradingMode::Log;
}


void DevelopThread::setLogMode(int mode)
{
    switch (mode)
    {
    case 1:  m_logMode = LogMode::Log16;   break;
    case 2:  m_logMode = LogMode::ArriLogC3; break;
    case 3:  m_logMode = LogMode::ACEScct; break;
    default: m_logMode = LogMode::Linear;  break;
    }
}

void DevelopThread::setToneCurveMode(int mode)
{
    switch (mode)
    {
    case 1: m_toneCurveMode = ToneCurveMode::SCurves;     break;
    case 2: m_toneCurveMode = ToneCurveMode::FilmicHable; break;
    case 3: m_toneCurveMode = ToneCurveMode::Reinhard;    break;
    case 4: m_toneCurveMode = ToneCurveMode::ACESLike;    break;
    case 0:
    default:
        m_toneCurveMode = ToneCurveMode::None;
        break;
    }
}

void DevelopThread::setSCurvePivot(int value)
{
    // value expected 0..100 (from slider)
    // map 0..100 -> 0.2..0.8 range (to avoid extreme pivots at 0 and 1)
    float t = std::clamp(value / 100.0f, 0.0f, 1.0f);
    m_sCurvePivot = 0.2f + t * 0.6f;  // 0.2 .. 0.8

    qDebug() << "S-curve pivot set to" << m_sCurvePivot;

    // You probably already have a mechanism that recomputes
    // SCurves when CurveRed/Green/Blue change. Call it here:
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

    float s = clamp02(p.grain); // 0..1 from slider

    // --- Make the first part of the slider much more gentle ---
    // gamma > 1 => small values become REALLY small (more “refined” at the low end)
    // Try 3.0–4.0. Higher = gentler start.
   // const float gamma = 3.6f;
   // float k = std::pow(s, gamma);   // k is the "effective" grain amount
    // baseline ensures a tiny grain even at low slider values (but still 0 when s==0)
    const float base = 0.10f;            // try 0.06 .. 0.15
    const float gamma = 2.2f;            // try 1.8 .. 2.6 (lower = more visible mid-range)
    float k = (s <= 0.0f) ? 0.0f : (base + (1.0f - base) * std::pow(s, gamma));

    if (k <= 1e-6f) return;

    // 1) Noise base
    cv::Mat noise(imgBgr.rows, imgBgr.cols, CV_32FC1);
    cv::randn(noise, 0.0f, 1.0f);

    // 2) Grain size: increase blur with slider (fine->coarse), but cap it
    // Use sqrt-like growth: size increases, but not too aggressively at the end.
    //float size = std::sqrt(s);                  // 0..1
    //float sigma = 0.15f + 1.8f * size;          // cap coarseness here (tune 1.4..2.2)
    // size grows slower at the start -> less blur -> sharper fine grain
    float size = std::pow(s, 1.6f);      // try 1.4 .. 2.2
    float sigma = 0.05f + 1.9f * size;   // try 0.02..0.08 and 1.6..2.2
    int ksize = int(std::ceil(sigma * 3) * 2 + 1);
    if (ksize < 3) ksize = 3;
    if ((ksize % 2) == 0) ksize += 1;

    cv::GaussianBlur(noise, noise, cv::Size(ksize, ksize), sigma, sigma, cv::BORDER_REFLECT);

    // 3) Amplitude: keep it low and controlled; scale by k (already gamma-shaped)
    // These caps are what stop 100% from getting “too heavy”.
    const float maxAmp8  = 6.0f;     // strong but not awful (try 4..8)
    const float maxAmp16 = 600.0f;   // 16-bit equivalent (try 400..900)

    float amp8  = maxAmp8  * k;
    float amp16 = maxAmp16 * k;

    // 4) Apply as 3-channel noise
    cv::Mat n3;
    cv::cvtColor(noise, n3, cv::COLOR_GRAY2BGR);

    if (imgBgr.type() == CV_8UC3)
    {
        cv::Mat f; imgBgr.convertTo(f, CV_32FC3);
        f += n3 * amp8;

        cv::threshold(f, f, 255.0, 255.0, cv::THRESH_TRUNC);
        cv::threshold(f, f, 0.0,   0.0,   cv::THRESH_TOZERO);
        f.convertTo(imgBgr, CV_8UC3);
    }
    else if (imgBgr.type() == CV_16UC3)
    {
        cv::Mat f; imgBgr.convertTo(f, CV_32FC3);
        f += n3 * amp16;

        cv::threshold(f, f, 65535.0, 65535.0, cv::THRESH_TRUNC);
        cv::threshold(f, f, 0.0,     0.0,     cv::THRESH_TOZERO);
        f.convertTo(imgBgr, CV_16UC3);
    }
}

//static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static inline float clamp01(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }

static float leakEnvelope(int t, int up, int hold, int down)
{
    if (t < 0) return 0.f;
    if (t < up) {
        // 0..1
        return float(t + 1) / float(up);
    }
    t -= up;
    if (t < hold) {
        return 1.0f;
    }
    t -= hold;
    if (t < down) {
        // 1..0
        return 1.0f - float(t + 1) / float(down);
    }
    return 0.f;
}


void DevelopThread::applySuper8LightLeakVertical(cv::Mat &imgBgr, int frameIndex, float sliderAmount01)
{
    float s = clamp02(sliderAmount01);

    // Early out if slider is basically off: reset arming/state
    if (s <= 0.0001f) {
        m_prevLeakSlider = 0.0f;
        m_leakBurst.active = false;
        m_leakBurst.phaseFrame = 0;
        m_leakBurst.nextAllowed = std::chrono::steady_clock::now();
        return;
    }

    // Read params once
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }

    auto now = std::chrono::steady_clock::now();

    // Helper: init a "look" (side/center/width/bloom/strength) for a leak burst.
    // For planned leaks we want mid-left / mid-right feel (more central than edge-only).
    auto initLeakLook = [&](bool forPlanned)
    {
        m_leakBurst.side = m_rng.uniform(0, 2); // 0=left, 1=right

        if (forPlanned) {
            // More "mid" entry (still left/right biased)
            if (m_leakBurst.side == 0) m_leakBurst.bandCenter = m_rng.uniform(0.18f, 0.40f);
            else                       m_leakBurst.bandCenter = m_rng.uniform(0.60f, 0.82f);

            // Planned should be able to wash out strongly; keep it wide.
            // Still allow slider to widen further (so you can choose how huge it gets).
            float sCurve = std::pow(clamp02(s), 1.7f);
            const float minW = 0.35f;   // already quite wide
            const float maxW = 0.98f;   // near full wash
            float targetW = minW + (maxW - minW) * sCurve;
            float randMul = m_rng.uniform(0.85f, 1.15f);
            m_leakBurst.bandWidth = clamp02(targetW * randMul);

            float baseBloom = 14.0f + 26.0f * sCurve;  // more bloom for planned
            float bloomMul  = m_rng.uniform(0.9f, 1.25f);
            m_leakBurst.sigmaBloom = std::min(60.0f, baseBloom * bloomMul);

            m_leakBurst.strengthMul = m_rng.uniform(0.95f, 1.20f);
        }
        else {
            // Random leak: your existing edge-biased behavior
            if (m_leakBurst.side == 0) m_leakBurst.bandCenter = m_rng.uniform(0.06f, 0.22f);
            else                       m_leakBurst.bandCenter = m_rng.uniform(0.78f, 0.94f);

            // Slider controls maximum "wash width". Randomness still makes each burst different.
            float sCurve = std::pow(clamp02(s), 1.7f);

            const float minW = 0.10f;
            const float maxW = 0.95f;
            float targetW = minW + (maxW - minW) * sCurve;

            float randMul = m_rng.uniform(0.75f, 1.25f);
            m_leakBurst.bandWidth = clamp02(targetW * randMul);

            float baseBloom = 8.0f + 18.0f * sCurve;  // 8..26
            float bloomMul  = m_rng.uniform(0.8f, 1.3f);
            m_leakBurst.sigmaBloom = std::min(40.0f, baseBloom * bloomMul);

            m_leakBurst.strengthMul = m_rng.uniform(0.75f, 1.10f);
        }
    };

    // ---------------------------------------------------------------------
    // 1) Planned scene leaks (always allowed if present; not gated by checkbox)
    //     Envelope: 5 up / 5 full / 5 down via plannedLeakEnvelope()
    //     Goal: FULL blowout during mid 5 frames.
    // ---------------------------------------------------------------------
    float plannedEnv = 0.0f;
    int plannedF = -1;

    for (int k = 0; k < 5; ++k) {
        int F = p.leakFrames[k];
        if (F <= 0) continue;

        float e = plannedLeakEnvelope(frameIndex, F);
        if (e > plannedEnv) { plannedEnv = e; plannedF = F; }
    }

    if (plannedEnv > 0.0f)
    {
        // Randomize planned look at the start of the planned window: frame F-5
        if (frameIndex == plannedF - 5) {
            initLeakLook(true);
        }

        // Make planned leaks fully blow out at peak (mid 5 frames).
        // We keep slider as "intensity control" but ensure full wash at peak.
        // Strategy:
        // - Use the planned envelope for timing.
        // - Boost A strongly + add a global wash floor at high envelope values.
        float A = clamp02(plannedEnv * m_leakBurst.strengthMul);
        if (plannedEnv >= 0.999f) A = 1.0f;   // guarantee full blast in the middle 5 frames
        applyLeakMask(imgBgr, frameIndex, A);//1.0f instead of A
        return;
    }

    // ---------------------------------------------------------------------
    // 2) Random leaks (optional, controlled by checkbox)
    // ---------------------------------------------------------------------
    if (!p.applyRandomLeaks)
        return;

    // Trigger condition: slider crosses from ~0 to >0
    bool risingEdge = (m_prevLeakSlider <= 0.0001f && s > 0.0001f);
    m_prevLeakSlider = s;

    // If not active, maybe start a new burst (rising edge or periodic)
    if (!m_leakBurst.active)
    {
        bool canStart = (now >= m_leakBurst.nextAllowed);

        if (risingEdge && canStart)
        {
            m_leakBurst.active = true;
            m_leakBurst.phaseFrame = 0;
            initLeakLook(false);

            m_leakBurst.rampUp = 5;
            m_leakBurst.hold   = 5;
            m_leakBurst.rampDown = 5;
        }
        else if (s > 0.10f && canStart)
        {
            m_leakBurst.active = true;
            m_leakBurst.phaseFrame = 0;
            initLeakLook(false);

            m_leakBurst.rampUp = 5;
            m_leakBurst.hold   = 5;
            m_leakBurst.rampDown = 5;
        }
        else
        {
            return; // no burst right now
        }
    }

    // Compute random-burst envelope
    float env = leakEnvelope(m_leakBurst.phaseFrame,
                             m_leakBurst.rampUp,
                             m_leakBurst.hold,
                             m_leakBurst.rampDown);

    m_leakBurst.phaseFrame++;

    // End of burst -> cooldown
    if (env <= 0.0001f)
    {
        m_leakBurst.active = false;
        m_leakBurst.phaseFrame = 0;

        int jitterMs = cv::theRNG().uniform(-2000, 3000);
        m_leakBurst.nextAllowed = now + std::chrono::milliseconds(15000 + jitterMs);
        return;
    }

    // Random leak intensity responds to slider + envelope + per-burst random
    float A = clamp02(s * env * m_leakBurst.strengthMul);
    if (A <= 0.0001f) return;

    applyLeakMask(imgBgr, frameIndex, A);
}


void DevelopThread::applySuper8LightLeak(cv::Mat &imgBgr)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }
    if (!p.enabled)//this is the check for whether to apply the super8 effects or not
        return;
    applySuper8LightLeakVertical(imgBgr, m_currentFrameIndex, p.lightLeak);
}



void DevelopThread::applyLeakMask(cv::Mat &imgBgr, int frameIndex, float A)
{
    if (A <= 0.0001f) return;

    const int inType = imgBgr.type();
    const float maxVal = (inType == CV_8UC3) ? 255.0f : (inType == CV_16UC3) ? 65535.0f : 0.0f;
    if (maxVal <= 0.0f) return;

    cv::Mat f; imgBgr.convertTo(f, CV_32FC3);
    int w = f.cols, h = f.rows;

    float cx = m_leakBurst.bandCenter * (w - 1);
    float sigmaX = std::max(2.0f, m_leakBurst.bandWidth * (w - 1) * 0.45f);

    cv::Mat band(1, w, CV_32FC1);
    for (int x = 0; x < w; ++x) {
        float dx = (x - cx);
        band.at<float>(0, x) = std::exp(-(dx*dx) / (2.0f*sigmaX*sigmaX));
    }

    cv::Mat streak1d(1, w, CV_32FC1);
    cv::RNG rng = cv::RNG((uint64)frameIndex * 1315423911U + 0xBADC0FFEEULL);
    cv::randu(streak1d, 0.55f, 1.25f);
    cv::GaussianBlur(streak1d, streak1d, cv::Size(31, 1), 0, 0, cv::BORDER_REFLECT);

    cv::Mat mask;
    cv::repeat(band.mul(streak1d), h, 1, mask);
    cv::GaussianBlur(mask, mask, cv::Size(0, 0), m_leakBurst.sigmaBloom, m_leakBurst.sigmaBloom, cv::BORDER_REFLECT);

    cv::Vec3f leakColor(0.05f, 0.22f, 1.00f);

    float strength = (0.45f * A) * maxVal;

    for (int y = 0; y < h; ++y) {
        const float *m = mask.ptr<float>(y);
        cv::Vec3f *row = f.ptr<cv::Vec3f>(y);
        for (int x = 0; x < w; ++x) {
            float a = m[x] * strength;
            row[x][0] += leakColor[0] * a;
            row[x][1] += leakColor[1] * a;
            row[x][2] += leakColor[2] * a;
        }
    }

    cv::threshold(f, f, maxVal, maxVal, cv::THRESH_TRUNC);
    cv::threshold(f, f, 0.0f,   0.0f,   cv::THRESH_TOZERO);

    if (inType == CV_8UC3)       f.convertTo(imgBgr, CV_8UC3);
    else if (inType == CV_16UC3) f.convertTo(imgBgr, CV_16UC3);
}

// DevelopThread.cpp

// DevelopThread.cpp
// Persistent / “living” Super8 scratches: they last multiple frames, fade in/out,
// and can slowly drift (“walk”) and/or “dance” within ~5% of frame width.
//
// Drop-in replacement for your previous applyScratches().
// Requires: <algorithm>, <cmath>, <opencv2/imgproc.hpp>, <opencv2/core.hpp>

void DevelopThread::applyScratches(cv::Mat &imgBgr, int frameIndex)
{
    Super8DevParams p;
    {
        QMutexLocker lock(&m_super8Mutex);
        p = m_super8;
    }
    if (!p.enabled) return;

   // const float s = clamp02(p.scratches); // 0..1 from slider
    float sRaw = clamp02(p.scratches);

    // Scale full slider range to only 0..33% effect strength
    const float maxEffect = 0.33f;
    float s = sRaw * maxEffect;


    if (s <= 0.0001f) return;

    if (imgBgr.empty() || imgBgr.channels() != 3) return;

    const int inType = imgBgr.type();
    const float maxVal =
        (inType == CV_8UC3)  ? 255.0f :
            (inType == CV_16UC3) ? 65535.0f : 0.0f;
    if (maxVal <= 0.0f) return;

    cv::Mat f;
    imgBgr.convertTo(f, CV_32FC3);

    const int w = f.cols;
    const int h = f.rows;

    // --- deterministic RNG per frame (stable look when re-rendering same frame) ---
    auto hash32 = [](uint32_t x) -> uint32_t {
        x ^= x >> 16; x *= 0x7feb352dU;
        x ^= x >> 15; x *= 0x846ca68bU;
        x ^= x >> 16;
        return x;
    };
    // ---- persistence: keep the same scratch "layout" for 1–2 frames, but still deterministic ----
    int blockSize = 1 + (hash32((uint32_t)(frameIndex / 6) ^ 0x6A1D2C3Bu) % 2); // 1 or 2, changes slowly
    int groupFrame = frameIndex / blockSize;

    uint32_t seed = hash32((uint32_t)groupFrame ^ 0xA53C9E1BU) ^ hash32((uint32_t)w * 131u + (uint32_t)h);
    cv::RNG rng((uint64)seed);

    // phase within the block (0..blockSize-1)
    const int phase = frameIndex - groupFrame * blockSize;


    // --- how many scratches ---
    const float resScale = std::sqrt((float)w / 1280.0f); // ~1 at 1280 wide
    const int baseCount  = (int)std::round(s * 10.0f * resScale);
    const int nScratches = std::max(0, baseCount + rng.uniform(0, 3));
    if (nScratches <= 0) return;

    cv::Mat maskBlack(h, w, CV_32FC1, cv::Scalar(0));
    cv::Mat maskBlue (h, w, CV_32FC1, cv::Scalar(0));

    for (int i = 0; i < nScratches; ++i)
    {
        // X position
        const int x0 = rng.uniform(0, w);

        // Keep only a tiny slant so it stays "vertical Super8"
        // Set to 0 if you want perfectly vertical.
        const int slant = rng.uniform(-2, 3);
        const int xEnd  = std::clamp(x0 + slant, 0, w - 1);

        // Thickness: mostly thin, sometimes a bit thicker
        int thickness = 1
                        + ((rng.uniform(0, 100) < 20) ? 1 : 0)
                        + ((rng.uniform(0, 100) <  8) ? 1 : 0);
        thickness = std::max(1, (int)std::round(thickness * (0.8f + 0.6f * resScale)));
        thickness = std::min(thickness, 2); // or 3 for 4K

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
        const bool bluish = (rng.uniform(0, 100) < 14);

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
    const float blueTint  = 0.35f;  // how "blue" the rare blue scratches become

    std::vector<cv::Mat> ch(3);
    cv::split(f, ch);

    // Darken (all channels): c *= (1 - mask * blackGain)
    cv::Mat darkFactor = 1.0f - (maskBlack * blackGain);
    cv::min(darkFactor, 1.0f, darkFactor);
    cv::max(darkFactor, 0.10f, darkFactor);

    ch[0] = ch[0].mul(darkFactor);
    ch[1] = ch[1].mul(darkFactor);
    ch[2] = ch[2].mul(darkFactor);

    // Bluish tint: add to Blue channel a bit, very subtle to G/R
    if (cv::countNonZero(maskBlue > 0.0005f) > 0)
    {
        cv::Mat bAdd = maskBlue * (blueTint * maxVal);
        cv::Mat gAdd = maskBlue * (0.08f * blueTint * maxVal);
        cv::Mat rAdd = maskBlue * (0.02f * blueTint * maxVal);

        ch[0] += bAdd;
        ch[1] += gAdd;
        ch[2] += rAdd;
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
    float sx = std::max(0.20f, radiusPx * (0.70f + 0.90f * uA));
    float sy = std::max(0.20f, radiusPx * (0.70f + 0.90f * uB));

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

    float amount = clamp02(p.dust); // 0..1 from slider

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
    // Mix frameIndex into seed; add a constant so frameIndex=0 still yields "random"
    std::mt19937 rng(uint32_t(frameIndex * 11027u + 12345u));

    std::uniform_int_distribution<int> ix(0, w - 1);
    std::uniform_int_distribution<int> iy(0, h - 1);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    std::uniform_real_distribution<float> rad(0.7f, 2.2f);

    // ---- Density model ----
    // Tune: bigger denominator -> fewer specks.
    // amount scales the count linearly.
    int count = int((double(w) * double(h) / 70000.0) * double(amount));

    // keep sane bounds (prevents “dust storms” on huge frames if slider is high)
    count = std::clamp(count, 0, 2500);

    // Rare bright “pinholes” (2–5% usually feels right)
    const float brightProb = 0.1; // 3%

    // Strength tuning (dark dust scales with amount; bright is rare and fairly strong)
    //const float darkDeltaBase   = -0.30f * amount * maxVal;  // subtractive
    const float darkDeltaBase   = -0.30f * 0.15 * maxVal;  // subtractive
    const float brightDeltaBase =  0.55f * maxVal;           // additive, rare

    std::uniform_real_distribution<float> darkScale(0.65f, 1.05f);
    std::uniform_real_distribution<float> brightScale(0.50f, 1.00f);

    for (int k = 0; k < count; ++k) {
        const int x = ix(rng);
        const int y = iy(rng);

        const bool bright = (u01(rng) < brightProb);

        if (bright) {
            // --- Bright pinhole: tiny, mostly sharp, with a subtle micro-halo ---
            cv::Vec3f& p = f.at<cv::Vec3f>(y, x);
            const float d = brightDeltaBase * brightScale(rng);

            p[0] = std::clamp(p[0] + d, 0.0f, maxVal);
            p[1] = std::clamp(p[1] + d, 0.0f, maxVal);
            p[2] = std::clamp(p[2] + d, 0.0f, maxVal);

            // tiny halo (very subtle; keeps it from looking like dead pixel)
            if (u01(rng) < 0.50f) {
                drawSoftDust(f, x, y, 0.85f, 0.12f * d, maxVal);
            }
        } else {
            // --- Dark dust: soft blob ---
            const float r = rad(rng);
            const float d = darkDeltaBase * darkScale(rng);
            // 10% chance: make it a tiny fiber instead of a blob
            if (u01(rng) < 0.30f) {
                std::uniform_real_distribution<float> ang(0.0f, 6.2831853f);
                std::uniform_int_distribution<int>    ln(8, 30);

                // Slightly weaker than blobs, otherwise fibers dominate
                drawDustFiber(f, x, y, ln(rng), ang(rng), 0.35f * d, maxVal);
            } else {
                drawSoftDust(f, x, y, r, d, maxVal);
            }
        }
    }

    f.convertTo(imgBgr, inType);
}

