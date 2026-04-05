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
    if ((height == 720) || (height == 480)) { height = 1080; width = 1440; }
    const Size S(width, height);

    char vid_name[256];
    std::snprintf(vid_name, sizeof(vid_name),
                  "C:/Users/patri/Videos/XiCapture_old/DigitalSuper8_output/Super8mpg%02d.%s",
                  Roll, file_type);

    cv::VideoWriter w;
    if (!w.open(vid_name, codec, fps, S, true)) {
        emit StatusUpdate("Could not open the output video for write.", true);
        return;
    }

    emit StatusUpdate("Developing film, please wait.", false);

    Mat HDimage;
    for (int i = first_frame; i <= frames; ++i) {
        if (cancelRequested_.load(std::memory_order_relaxed)) {
            emit StatusUpdate("Render cancelled.", true);
            break;
        }

        onDevelopFrame(i);

        if (height == 1080) {
            resize(imgBGR, HDimage, S, 4.0, 1.0, INTER_CUBIC);
            if (ImageDeveloped) w << HDimage;
        } else {
            if (ImageDeveloped) w << imgBGR;
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


// ----------------- Your existing code below (UNTOUCHED) -----------------

void DevelopThread:: onDevelopFrame(int i)
{
    char in_name[256];
    vector<Mat> images;
    Mat fusion;
    Mat imgBAY;
    Mat imgBGRlow, imgBGRhigh;
    Size S = Size(width, height);
    cv::Mat img_HSV(S, CV_16UC3);
    Mat dst;
    vector<Mat> channels;

    sprintf(in_name, "%s/images%d/XiCapture%03d.pgm", work_path, Roll, (i));
    imgBAY = imread(in_name, IMREAD_ANYDEPTH);

    if(imgBAY.empty()){
        ImageDeveloped = false;
        qDebug() << "FROM function Develop Frame: No data in image";
        return;
    }
    cvtColor(imgBAY,imgBGR,COLOR_BayerGB2BGR, 3);

    qDebug() << "successfully demosaiced, bit depth = " << imgBGR.depth() << "number of channels: "<< imgBGR.channels();

    if(imgBGR.depth() == 2){
        if (bitdepth == 8){
            if (!HDR){
                qDebug() << "Converting 16bit to 8 bit, dev bitdepth = " << bitdepth << " NO HDR!!";
                imgBGR.convertTo(imgBGR, CV_8UC3, (1.0/16.0));
                imgBGR = correctGamma(imgBGR, 2.2);
            }
            else{
                qDebug() << "Converting 16bit to 8bit, dev bitdepth = " << bitdepth << " APPLYING HDR!!";
                imgBGR.convertTo(imgBGR, CV_8UC3, (1.0/16.0));
                imgBGR.convertTo(imgBGRlow, CV_8UC3, 1, -32);
                imgBGR.convertTo(imgBGRhigh, CV_8UC3, 1, 32);
                images.push_back(imgBGR);
                images.push_back(imgBGRlow);
                images.push_back(imgBGRhigh);
                qDebug() << "Applied HDR, successfully filled the vector with 8 bit";

                images.pop_back();
                images.pop_back();
                images.pop_back();

                fusion.convertTo(imgBGR, CV_8UC3, 255, 0);
                imgBGR = correctGamma(imgBGR, 2.2);
            }

        }
        else{

            if (!HDR){
                qDebug() << "Keeping 16 bit, dev bitdepth = " << bitdepth << " NO HDR!!";
                imgBGR = LOG16(imgBGR, lut16);
                if(filmlook) FilmLook16(imgBGR, imgBGR, 48, 0.6, 192, 0.3, lutSCurveBlue, lutSCurveGreen, lutSCurveRed);
            }
            else{
                qDebug() << "Keeping 16 bit because bitdepth = " << bitdepth << " APPLYING HDR!!";
                imgBGR = LOG16(imgBGR, lut16);
                if(filmlook) FilmLook16(imgBGR, imgBGR, 48, 0.6, 192, 0.3, lutSCurveBlue, lutSCurveGreen, lutSCurveRed);
            }
        }
    }
    else{
        // 8 bit image case
        if (!HDR){
            qDebug() << "8 bit, no HDR";
            LUT( imgBGR, lut8_array, imgBGR );
            LUT( imgBGR, lutS_array, imgBGR );
        }
        else{
            imgBGR.convertTo(imgBGRlow, CV_8UC3, 1, -32);
            imgBGR.convertTo(imgBGRhigh, CV_8UC3, 1, 32);
            images.push_back(imgBGR);
            images.push_back(imgBGRlow);
            images.push_back(imgBGRhigh);
            qDebug() << "Applied HDR, successfully filled the vector with 8 bit";

            images.pop_back();
            images.pop_back();
            images.pop_back();

            fusion.convertTo(imgBGR, CV_8UC3, 255, 0);
            imgBGR = correctGamma(imgBGR, 2.2);
        }
    }

    // Color & brightness corrections
    if(imgBGR.depth() == 2){
        imgBGR.convertTo(imgBGR, CV_32FC3, (1./4095.), 0);
    }
    cvtColor(imgBGR, img_HSV, COLOR_RGB2HSV_FULL);
    split(img_HSV,channels);
    if(img_HSV.depth() == 5){
        channels[2].convertTo(channels[2], -1, (Contrast/100.0), (Bright/255.));
    }
    else{
        channels[2].convertTo(channels[2], -1, (Contrast/100.0), (Bright));
    }
    channels[1].convertTo(channels[1], -1, (Sat/100.0), 0);
    merge(channels, img_HSV);
    cvtColor(img_HSV, imgBGR, COLOR_HSV2RGB_FULL);

    split(imgBGR,channels);
    channels[0].convertTo(channels[0], -1, Blue, 0);
    channels[1].convertTo(channels[1], -1, Green, 0);
    channels[2].convertTo(channels[2], -1, Red, 0);
    merge(channels,imgBGR);

    if(imgBGR.depth() == 5){
        imgBGR.convertTo(imgBGR, CV_8UC3, 255.0, 0);
        if(gammacorrect){
            imgBGR = correctGamma(imgBGR, (1.0/2.2));
        }
        qDebug() << "corrected mid tones";
        if (HDR){
            imgBGR.convertTo(imgBGRlow, CV_8UC3, 1, -64);
            imgBGR.convertTo(imgBGRhigh, CV_8UC3, 1, 64);
            images.push_back(imgBGR);
            images.push_back(imgBGRlow);
            images.push_back(imgBGRhigh);
            qDebug() << "Applied HDR, successfully filled the vector with 8 bit";

            images.pop_back();
            images.pop_back();
            images.pop_back();

            fusion.convertTo(imgBGR, CV_8UC3, 255, 0);
        }
    }
    else{
        if (filmlook) imgBGR = FilmLook(imgBGR, 48, 0.6, 192, 0.3);
        qDebug() << "entered into filmlook";
    }

    // 26072024: flip image
    flip(imgBGR, imgBGR, -1);

    ImageDeveloped = true;
    emit FrameDeveloped(i);
    return;
}

void DevelopThread::calc_LutCurve()
{
    for (int i = 0; i < 4096; i++)
    {
        lutSCurveRed[i]   = saturate_cast<ushort>((1/(1+ exp((double)(-(CurveRed/4095.0)*((i-2450)-0))))) * 4095.0);
        lutSCurveGreen[i] = saturate_cast<ushort>((1/(1+ exp((double)(-(CurveGreen/4095.0)*((i-2450)-0))))) * 4095.0);
        lutSCurveBlue[i]  = saturate_cast<ushort>((1/(1+ exp((double)(-(CurveBlue/4095.0)*((i-2450)-0))))) * 4095.0);
    }
    // Emit a copy that the UI can safely use for painting
    QVector<quint16> r(4096), g(4096), b(4096);
    for (int i=0;i<4096;++i) { r[i]=lutSCurveRed[i]; g[i]=lutSCurveGreen[i]; b[i]=lutSCurveBlue[i]; }
    emit sCurvesUpdated(r, g, b);
}
