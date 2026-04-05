#include "developthread.h"
#include <QtCore>
#include <QDebug>
#include <QMetaObject>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/photo.hpp>

#include "processing.h"

using namespace cv;
using namespace std;

namespace
{
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

DevelopThread::DevelopThread(QObject* parent) : QObject(parent)
{
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
    renderingBusy = true;
    cancelRequested_.store(false, std::memory_order_relaxed);

    // Normalize size as your original code
    if ((height == 720) || (height == 480)) {
        height = 1080;
        width  = 1440;
    }
    const cv::Size S(width, height);

    char vid_name[256];
    std::snprintf(vid_name, sizeof(vid_name),
                  "C:/Users/patri/Videos/XiCapture_old/DigitalSuper8_output/Super8mpg%02d.%s",
                  Roll, file_type);

    cv::VideoWriter w;
    if (!w.open(vid_name, codec, fps, S, true)) {
        emit StatusUpdate("Could not open the output video for write.", true);
        renderingBusy = false;              // make sure we reset this on failure
        return;
    }

    emit StatusUpdate("Developing film, please wait.", false);

    cv::Mat HDimage;
    for (int i = first_frame; i <= frames; ++i) {
        if (cancelRequested_.load(std::memory_order_relaxed)) {
            emit StatusUpdate("Render cancelled.", true);
            break;
        }

        onDevelopFrame(i);
        if (!ImageDeveloped)
            continue;

        // --- Always feed 8-bit BGR to VideoWriter ---
        cv::Mat frameOut;

        if (imgBGR.depth() == CV_16U) {
            // 12-bit data in 16-bit container (0..4095) → 0..255
            imgBGR.convertTo(frameOut, CV_8UC3, 255.0 / 4095.0, 0.0);

            // If you want gamma on the video as well, mirror the behavior:
            if (gammacorrect) {
                frameOut = correctGamma(frameOut, 1.0 / 2.2);
            }
        } else {
            // already 8-bit
            frameOut = imgBGR;
        }

        if (height == 1080) {
            cv::resize(frameOut, HDimage, S, 0.0, 0.0, cv::INTER_CUBIC);
            w << HDimage;
        } else {
            w << frameOut;
        }
    }

    emit StatusUpdate("Status OK, waiting...", false);
    emit renderFinished();
    w.release();
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
    try
    {
        ImageDeveloped = false;

        // ----------------------------------------------------------
        // 0. Load RAW Bayer
        // ----------------------------------------------------------
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
        // 1. Debayer → CameraRGB (linear integer)
        // ----------------------------------------------------------
        cv::Mat imgCamRGB;
        cv::cvtColor(imgBAY, imgCamRGB, cv::COLOR_BayerGB2BGR);

        const bool rawWas16Bit = (imgCamRGB.depth() == CV_16U);

        // ----------------------------------------------------------
        // 2. Normalize to linear float [0..1]
        // ----------------------------------------------------------
        cv::Mat linearCam;
        if (rawWas16Bit)
            imgCamRGB.convertTo(linearCam, CV_32FC3, 1.0f / 4095.0f);
        else
            imgCamRGB.convertTo(linearCam, CV_32FC3, 1.0f / 255.0f);

        // ----------------------------------------------------------
        // 3. Apply exposure in linear light
        // ----------------------------------------------------------
        if (std::abs(ExposureEV) > 1e-4)
        {
            float g = std::pow(2.0f, ExposureEV);
            linearCam *= g;
        }

        // ----------------------------------------------------------
        // 4. White balance (per-channel scaling)
        // ----------------------------------------------------------
        // (You can refine these, but this works well)
        // Voor nu neutraal (1,1,1); later kun je dit koppelen aan sliders.
        const float wbR = 1.0f;
        const float wbG = 1.0f;
        const float wbB = 1.0f;
        applyWhiteBalance(linearCam, wbR, wbG, wbB);

        // ----------------------------------------------------------
        // 5. Mini-IDT: CameraRGB → sRGB → ACEScg
        // ----------------------------------------------------------

        // 5a. Apply your camera CCM (diagonal)
        static const cv::Matx33f kCCM(
            1.264223f, 0.0f,      0.0f,
            0.0f,      0.830212f, 0.0f,
            0.0f,      0.0f,      0.995531f
            );
        applyColorMatrix(linearCam, kCCM);

        // 5b. sRGB primaries → ACEScg primaries
        cv::Mat acesCG;
        linearCam.copyTo(acesCG);
        applySRGBtoACEScg(acesCG);

        clamp01(acesCG);

        // ----------------------------------------------------------
        // 6. Log mode: ACEScg → (optional) LOG16 OR ACEScct
        // ----------------------------------------------------------
        cv::Mat logBase;

        switch (m_logMode)
        {
        case LogMode::Log16:
        {
            cv::Mat tmp16;
            acesCG.convertTo(tmp16, CV_16UC3, 4095.0);
            tmp16 = LOG16(tmp16, lut16);             // legacy S-Log style
            tmp16.convertTo(logBase, CV_32FC3, 1.0f/4095.0f);
            break;
        }
        case LogMode::ACEScct:
            applyACEScct(acesCG, logBase);          // official ACEScct curve
            //logBase = acesCG.clone();
            break;

        case LogMode::Linear:
        default:
            logBase = acesCG.clone();
            break;
        }

        clamp01(logBase);

        // ----------------------------------------------------------
        // 7. Tone mapping (creative: filmlook, Hable, Reinhard, None)
        // ----------------------------------------------------------
        cv::Mat graded = logBase.clone();

        switch (m_toneCurveMode)
        {
        case ToneCurveMode::SCurves:
            applySCurvesFloat(graded,
                              lutSCurveRed,
                              lutSCurveGreen,
                              lutSCurveBlue);
            break;

        case ToneCurveMode::FilmicHable:
            applyFilmicHable(graded);
            break;

        case ToneCurveMode::Reinhard:
            applyReinhardGamma(graded, 2.2f);
            break;

        case ToneCurveMode::ACESLike:
            applyACES(graded);
         //   applyACES_FilmTone(graded); // lightweight RRT-like
            break;

        case ToneCurveMode::None:
        default:
            break;
        }

        clamp01(graded);

        // ----------------------------------------------------------
        // 8. Local adjustments
        // ----------------------------------------------------------
        applyBrightnessContrast(graded, Contrast, Bright);
        applySaturation(graded, Sat);
        applyChannelGains(graded, Blue, Green, Red);

        clamp01(graded);

        // ----------------------------------------------------------
        // 9. RRT + ODT Rec709  (true ACES output transform)
        // ----------------------------------------------------------
        cv::Mat final709;
        applyRRT_ODT_Rec709(graded, final709);

        clamp01(final709);

        // ----------------------------------------------------------
        // 10. Convert to 8/16 bit for display/output
        // ----------------------------------------------------------
        if (bitdepth == 16)
            final709.convertTo(imgBGR, CV_16UC3, 4095.0);
        else
            final709.convertTo(imgBGR, CV_8UC3, 255.0);

        // ----------------------------------------------------------
        // 11. Flip & publish
        // ----------------------------------------------------------
        cv::flip(imgBGR, imgBGR, -1);
        ImageDeveloped = true;

        emit FrameDeveloped(i);
    }
    catch (const cv::Exception& e)
    {
        qWarning() << "[DevelopThread] OpenCV EX:" << e.what();
        return;
    }
}



void DevelopThread::calc_LutCurve()
{
    auto buildCurve = [&](double strength, ushort* lut)
    {
        // 0 → identity (no curve at all, no grey wash)
        if (strength <= 0.0) {
            for (int i = 0; i < 4096; ++i) {
                lut[i] = static_cast<ushort>(i);
            }
            return;
        }

        // This is the "base" curve level that you like (8 was your sweet spot)
        const double baseCurve = 8.0;

        // Pivot stays where your original curve had it
        const double pivot = 2450.0;

        for (int i = 0; i < 4096; ++i)
        {
            double x = static_cast<double>(i);
            double y = 0.0;

            if (strength < baseCurve)
            {
                // ---- LOW-STRENGTH REGION: blend identity ↔ S-curve(at 8) ----
                // First compute the "hard" S-curve using the baseCurve = 8
                double kBase = baseCurve / 4095.0;
                double sBase = 1.0 / (1.0 + std::exp(-kBase * (x - pivot)));
                double hard  = sBase * 4095.0;

                double ident = x;
                double t     = strength / baseCurve;  // 0..1

                // Blend: small strengths = near identity, strength=8 = full S(8)
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

void DevelopThread::setLogMode(int mode)
{
    switch (mode)
    {
    case 0:
        m_logMode = LogMode::Linear;
        break;
    case 1:
        m_logMode = LogMode::Log16;
        break;
    case 2:
        m_logMode = LogMode::ACEScct;
        break;
    default:
        m_logMode = LogMode::Linear;
        break;
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



