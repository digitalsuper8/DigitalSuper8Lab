#include <QThread>
#include "processing.h"
#include "ui_processing.h"
#include "settingsdialog.h"
#include "developdialog.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <iostream> // for standard I/O
#include <QDebug>
#include <QFileDialog>
#include <fstream>
#include <QMessageBox>
#include <QTimer>
#include <QSettings>
#include <QActionGroup>
#include "dngbatchdialog.h"
//#include "scurvepreviewlut.h"
#include <algorithm>
#include <array>
#include <QRegularExpression>


using namespace std;
using namespace cv;

EmulsionPreset Processing::getEmulsionPreset(EmulsionLook look) const
{
    EmulsionPreset p{};

    // redelijke defaults
    p.toneCurveMode = 1;   // SLOG
    p.exposureEV    = 4.0f; // This is the index 4 for value 0.0EV
    p.sCurveR       = 8.0f;
    p.sCurveG       = 8.0f;
    p.sCurveB       = 8.0f;
    p.brightness    = -50.0f;
    p.contrast      = 100.0f;
    p.saturation    = 170.0f;
    p.rGain         = 140.0f;
    p.gGain         = 125.0f;
    p.bGain         = 125.0f;

    switch (look)
    {
    case EmulsionLook::Ektachrome100D:
        // Ektachrome 100D – daylight slide film:
        // - iets onderbelicht om highlights te beschermen
        // - hoge microcontrast, frisse kleuren
        // - duidelijke blues en iets koelere look
        p.toneCurveMode = 3;      // Filmic base curve
        p.exposureEV    = 3.5f;   // iets onder 0EV
        p.sCurveR       = 9.0f;
        p.sCurveG       = 8.0f;
        p.sCurveB       = 10.0f;  // iets steilere blue-curve

        p.brightness    = -55.0f; // net wat donkerder dan default -50
        p.contrast      = 115.0f; // wat meer punch dan 100
        p.saturation    = 190.0f; // iets meer kleur dan 170

        // kleurgevoel: helder, licht koel
        p.rGain         = 135.0f;
        p.gGain         = 125.0f;
        p.bGain         = 140.0f;
        break;


    case EmulsionLook::Vision3_500T:
        // Kodak Vision3 500T – tungsten negative:
        // - zachte roll-off, veel latitude
        // - wat warmer in tungsten-licht, niet super contrasty
        // - iets meer exposure (negatief houdt van over)
        p.toneCurveMode = 2;      // ACES-like base curve
        p.exposureEV    = 4.5f;   // iets boven 0EV
        p.sCurveR       = 7.0f;
        p.sCurveG       = 7.0f;
        p.sCurveB       = 7.0f;   // wat zachtere S-curves

        p.brightness    = -45.0f; // net een tikje helderder
        p.contrast      = 90.0f;  // iets minder contrast dan default 100
        p.saturation    = 165.0f; // iets minder saturatie dan default 170

        // warmere balans: iets meer rood, iets minder blauw
        p.rGain         = 150.0f;
        p.gGain         = 130.0f;
        p.bGain         = 115.0f;
        break;


    case EmulsionLook::Wolfen400:
        // Wolfen/ORWO-achtige 400 film:
        // - contrastrijk, “vintage” vibe
        // - lichte warmte, maar niet zo tungsten-warm als 500T
        p.toneCurveMode = 4;      // Reinhard base, daarna stevige S-curves
        p.exposureEV    = 4.0f;   // rond neutraal
        p.sCurveR       = 10.0f;
        p.sCurveG       = 9.0f;
        p.sCurveB       = 9.0f;

        p.brightness    = -50.0f; // gelijk aan default
        p.contrast      = 120.0f; // duidelijk meer contrast
        p.saturation    = 180.0f; // lichte boost t.o.v. default

        // warm maar minder extreem dan 500T
        p.rGain         = 145.0f;
        p.gGain         = 120.0f;
        p.bGain         = 120.0f;
        break;


    case EmulsionLook::None:
    default:
        // laat defaults staan (jouw SLOG + algemene waarden)
        break;
    }

    return p;
}

static std::array<int,5> parseLeakFrames(const QString &text)
{
    std::vector<int> v;
    v.reserve(5);

    // split on comma, semicolon, whitespace
    QString t = text;
    t.replace(';', ',');
    const QStringList parts = t.split(QRegularExpression("[,\\s]+"), Qt::SkipEmptyParts);

    for (const QString &p : parts) {
        bool ok = false;
        int n = p.trimmed().toInt(&ok);
        if (!ok) continue;
        if (n <= 0) continue;
        v.push_back(n);
    }

    // unique + sort
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());

    std::array<int,5> out{0,0,0,0,0};
    for (size_t i = 0; i < out.size() && i < v.size(); ++i)
        out[i] = v[i];

    return out;
}


Processing::Processing(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::Processing)



{
    PlayingVideo = false;
   // RenderingVideo = false;
    filmlook = true;
    gammacorrect = false;
    ui->setupUi(this);
    //Debounce timer for the super8 effects sliders
    m_super8PreviewTimer = new QTimer(this);
    m_super8PreviewTimer->setSingleShot(true);
    m_super8PreviewTimer->setInterval(30); // 30–60ms feels good

    connect(m_super8PreviewTimer, &QTimer::timeout, this, [this]() {
        // Re-develop the current frame immediately
        on_horizontalSlider_3_valueChanged(ImgViewCount);
    });
    // end of debounce timer instantiation and connecting it to the slot on_horizontalSlider_3_valueChanged(ImgViewCount)

  //  qRegisterMetaType<Super8DevParams>("Super8DevParams");

    hookUpSuper8DevUi();

    setupThemeActions();

    //Fullscreen:
    // === Fullscreen toggle ===
    QAction *toggleFullscreenAct = new QAction(this);
    toggleFullscreenAct->setShortcut(QKeySequence(Qt::SHIFT | Qt::Key_F));
    toggleFullscreenAct->setShortcutContext(Qt::WindowShortcut);

    // Connect the action
    connect(toggleFullscreenAct, &QAction::triggered, this, [this]() {
        if (isFullScreen()) {
            showNormal();
            menuBar()->show();
        } else {
            showFullScreen();
            menuBar()->hide();   // optional
        }
    });

    // Optionally add to the window so it receives the shortcut
    addAction(toggleFullscreenAct);


    // Load last-used theme from QSettings (optional but nice)
    QSettings settings("STEEMERS", "DigitalSuper8");
    const QString themeId = settings.value("ui/theme", "kodak_yellow").toString();
    applyTheme(themeId);


    // Make sure the preview area wants to expand; don’t let QLabel stretch pixels itself.
    ui->Label_showimage->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->Label_showimage->setScaledContents(false);
    ui->Label_showimage->setMinimumSize(100, 80); // avoids 0x0 corner cases

    for (int i = 0; i < 4096; i++)
    {
        lut16[i] = saturate_cast<ushort>(((3757.0-32.0)*0.432699)*log10((((double)i-(32.0))/(470.0-32.0))+0.037584) + (0.616596*4095));

    }
    sprintf(work_path, "C:/Users/patri/Videos/XiImages");

    Roll = 2;
    ASA = 100;
    width = 640;
    height = 480;

    // create the worker thread
    devQThread = new QThread(this);

    // create the worker (same name: developThread)
    devThread = new DevelopThread();
    devThread->moveToThread(devQThread);

    // init timer once thread starts
    connect(devQThread, &QThread::started, devThread, &DevelopThread::init);

    // lifetime management
    connect(devQThread, &QThread::finished, devThread, &QObject::deleteLater);

    // existing signals
    connect(devThread, SIGNAL(FrameDeveloped(int)), this, SLOT(onFrameDeveloped(int)));
    connect(this, SIGNAL(DevelopFrame(int)), devThread, SLOT(onDevelopFrame(int)));
    connect(devThread, SIGNAL(StatusUpdate(QString,bool)), this, SLOT(onStatusUpdate(QString,bool)));
    connect(devThread, &DevelopThread::sCurvesUpdated,
            ui->sCurvePreview, &SCurvePreviewLUT::setLUTs);

    connect(this, &Processing::super8DevParamsChanged,
            devThread, &DevelopThread::setSuper8DevParams,
            Qt::DirectConnection);



    // start the worker thread’s event loop
    devQThread->start();
    connect(ui->comboLogMode,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &Processing::onLogModeChanged);

    connect(ui->comboToneCurve,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &Processing::onToneCurveChanged);

    connect(ui->comboGradingMode,
            QOverload<int>::of(&QComboBox::currentIndexChanged),
            this,
            &Processing::on_comboGradingMode_currentIndexChanged);

    sprintf(imgcounter_filename, "%s/imgcounter%d.txt", work_path, Roll);
    qDebug() << " IIMGCOUNTER IS: " << imgcounter_filename;
    imgcount = GetNumber(imgcounter_filename);

     connect(ui->horizontalSlider_3, &QSlider::sliderMoved,
            this, [this](int v){ on_horizontalSlider_3_valueChanged(v); });


    QTimer::singleShot(0, this, [this]{
        on_LastButton_clicked();
    });

    devThread->first_frame = 1;

    onStatusUpdate("Status OK, waiting...", false);
    devThread->filmlook = false;

    // ---- Base curve default: ACES Filmic ----
    ui->comboLogMode->setCurrentIndex(2);  // "ACES Filmic (HDR)"
    devThread->useLog = false;
    devThread->HDR    = true;
    devThread->setLogMode(ui->comboLogMode->currentIndex());
    devThread->setToneCurveMode(ui->comboToneCurve->currentIndex());

    onUpdateUI();
   // ui->SCurveBox->
    ui->Red_Slider->setValue(100);//140
    ui->Green_Slider->setValue(100);//120
    ui->Blue_Slider->setValue(100);//160
    ui->Brightness_slider->setValue(0);//-50
    ui->Saturation_slider->setValue(100);//240
    ui->Contrast_slider->setValue(99);//250
    ui->ExposureCombo->setCurrentIndex(4); // 4 = "0 EV"
    // After devQThread->start(); in the constructor:
    devThread->CurveRed   = ui->spinBox_RedCurve->value();
    devThread->CurveGreen = ui->spinBox_GreenCurve->value();
    devThread->CurveBlue  = ui->spinBox_BlueCurve->value();
    devThread->calc_LutCurve();   // this will emit sCurvesUpdated → paint first curves

    ui->spinBox_RedCurve->setMaximum(64);
    ui->spinBox_GreenCurve->setMaximum(64);
    ui->spinBox_BlueCurve->setMaximum(64);


    //NEW to simplify the UI (I added)

    // ----------------------------------------------------------
    // Simplified DS8Lab mode: hide legacy wide-gamut / multi-mode controls
    // ----------------------------------------------------------
    this->setWindowTitle("DS8Lab - Simple Processing");

    ui->comboGradingMode->hide();
    ui->comboLogMode->hide();
    ui->comboToneCurve->hide();
    ui->comboEmulsionLook->hide();
    ui->GammaCorrectionBox->hide();


    ui->SCurveBox->setText("Extra S-Curve");
    ui->devGroup->setTitle("Simple Development");
    ui->groupBox->setTitle("Simple Grade");
    ui->rgbGroup->setTitle("Color Balance");
    ui->label_11->setText("Simple preview-style processing");

    // Previewthread-like defaults
    ui->bitDepth->setValue(8);

    ui->Red_Slider->setValue(140);
    ui->Green_Slider->setValue(100);
    ui->Blue_Slider->setValue(160);

    ui->Brightness_slider->setValue(-40);
    ui->Saturation_slider->setValue(250);
    ui->Contrast_slider->setValue(100);
    ui->ExposureCombo->setCurrentIndex(4); // 0 EV

    ui->SCurveBox->setChecked(false);
    ui->spinBox_RedCurve->setValue(8);
    ui->spinBox_GreenCurve->setValue(8);
    ui->spinBox_BlueCurve->setValue(8);
    ui->sliderSCurvePivot->setValue(50);

    devThread->bitdepth = 8;
    devThread->filmlook = false;
    devThread->HDR = false;
    devThread->gammacorrect = false;

    devThread->Red = 1.40f;
    devThread->Green = 1.00f;
    devThread->Blue = 1.60f;
    devThread->Bright = -40.0f;
    devThread->Sat = 250.0f;
    devThread->Contrast = 100.0f;
    devThread->ExposureEV = 0.0f;

    devThread->CurveRed   = ui->spinBox_RedCurve->value();
    devThread->CurveGreen = ui->spinBox_GreenCurve->value();
    devThread->CurveBlue  = ui->spinBox_BlueCurve->value();
    devThread->calc_LutCurve();
    }

Processing::~Processing()
{
    if (devThread) {
        devThread->cancelRender();
        devThread->pause(); // stop timer if playing
    }
    if (devQThread) {
        devQThread->quit();
        devQThread->wait();
    }
    delete ui;
}

static float slider01(int v)  // v = 0..100
{
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    return float(v) / 100.0f;
}

Super8DevParams Processing::readSuper8DevParamsFromUi() const
{
    Super8DevParams p;

    p.enabled = ui->devEffectsEnableBox->isChecked();
    p.applyRandomLeaks = ui->devRandomLeaksCheckBox->isChecked();

    // your existing sliders...
    p.lightLeak = slider01(ui->devLightLeakSlider->value());
    p.grain     = slider01(ui->devGrainSlider->value());
    p.dust      = slider01(ui->devDustSlider->value());
    p.scratches = slider01(ui->devScratchesSlider->value());
    p.weave     = slider01(ui->devWeaveSlider->value());

    p.fadeInWarmthFrames = ui->devFadeInWarmthSlider->value();
    p.fadeOutYellowShift = slider01(ui->devFadeOutYellowSlider->value());

    // planned leak frames
    auto arr = parseLeakFrames(ui->leakFramesLineEdit->text());
    for (int i = 0; i < 5; ++i)
        p.leakFrames[i] = arr[i];

    return p;
}

void Processing::hookUpSuper8DevUi()
{
    auto publish = [this]() {
        m_super8Dev = readSuper8DevParamsFromUi();
        emit super8DevParamsChanged(m_super8Dev);

        // Refresh current frame preview (debounced)
        if (m_super8PreviewTimer)
            m_super8PreviewTimer->start();
    };

    // Fire once at startup so downstream has initial values:
    publish();

    // --- Basic enable ---
    connect(ui->devEffectsEnableBox, &QCheckBox::toggled, this, publish);

    // --- Random leaks checkbox (new) ---
    connect(ui->devRandomLeaksCheckBox, &QCheckBox::toggled, this, publish);

    // --- Planned leak frames line edit (new) ---
    // Fires while typing; debounced by m_super8PreviewTimer
    connect(ui->leakFramesLineEdit, &QLineEdit::textChanged, this, publish);

    // --- Sliders: connect both valueChanged + sliderMoved to be bulletproof ---
    auto hookSlider = [this, &publish](QSlider *s) {
        if (!s) return;
        s->setTracking(true);
        connect(s, &QSlider::valueChanged, this, publish);
        connect(s, &QSlider::sliderMoved,  this, publish);
    };

    hookSlider(ui->devScratchesSlider);
    hookSlider(ui->devDustSlider);
    hookSlider(ui->devLightLeakSlider);
    hookSlider(ui->devWeaveSlider);
    hookSlider(ui->devGrainSlider);
    hookSlider(ui->devFadeInWarmthSlider);
    hookSlider(ui->devFadeOutYellowSlider);
}


void Processing::onUpdateUI()
{
    ui->label_6->setText(QString::number(Roll));
    ui->label_7->setText(QString::number(ASA));
    ui->label_8->setText(QString::number(width));
    ui->label_9->setText(QString::number(height));
    ui->label->setText(QString::number(imgcount));
    ui->progressBar->setValue(imgcount);
    ImgViewCount = imgcount;
    ui->horizontalSlider_3->setMinimum(0);

  //  ui->horizontalSlider_3->setMinimum(1);
  //  ui->horizontalSlider_3->setMaximum(imgcount);
  //  ui->horizontalSlider_3->setValue(qMax(1, imgcount));
  //  ImgViewCount = qMax(1, imgcount);

    ui->horizontalSlider_3->setMaximum(ImgViewCount);
    ui->horizontalSlider_3->setValue(ImgViewCount);
}



void Processing::on_Settings_Button_clicked()
{
    SettingsDialog sDialog;
    connect(&sDialog,SIGNAL(RollChanged(int,int,int,int,int,int,int)), this,SLOT(onRollChanged(int,int,int,int,int,int,int)));
    connect(&sDialog,SIGNAL(ASAChanged(int,float)), this,SLOT(onASAChanged(int,float)));
    connect(&sDialog,SIGNAL(ExposureChanged(int)), this,SLOT(onExposureChanged(int)));
    connect(&sDialog,SIGNAL(Bitdepth_SettingChanged(int)), this,SLOT(on_Bitdepth_SettingChanged(int)));
    connect(&sDialog, SIGNAL(StatusUpdate(QString, bool)), this,SLOT(onStatusUpdate(QString, bool)));
    sDialog.setModal(true);
    sDialog.exec();
}

void Processing::onASAChanged(int setASA, float setGain)
{
    ASA = setASA;
    onUpdateUI();
}

void Processing::onExposureChanged(int setExp)
{
     return;
}

void Processing::onRollChanged(int setRoll, int setExp, int setDown, int setOffx, int setOffy, int setWidth, int setHeight)
{
     Roll = setRoll; width = setWidth; height = setHeight;

     sprintf(imgcounter_filename, "%s/imgcounter%d.txt", work_path, Roll);
     imgcount = GetNumber(imgcounter_filename);
     onUpdateUI();
     return;
}

void Processing::on_DevelopFilm_Button_clicked()
{
    // Make sure devThread knows the current capture context:
    devThread->width  = width;
    devThread->height = height;
    devThread->Roll   = Roll;

    std::strncpy(devThread->work_path, work_path, sizeof(devThread->work_path)-1);
    devThread->work_path[sizeof(devThread->work_path)-1] = '\0';
    // work_path is already copied elsewhere when you select the dir (good)


    DevelopDialog dDialog(this);

    // No need to connect the old signals anymore; the dialog will launch the render itself.
    // (If you still want the signals for legacy UI text, you can keep them, but don't start the thread here.)

    // Hand the dialog the context it needs: the thread to run on, current frame, and last frame
    dDialog.setDevelopContext(devThread, ImgViewCount, imgcount);
    dDialog.setModal(true);
    dDialog.exec();   // On OK, DevelopDialog::onDialogAccepted() calls devThread->startRenderToFile()
}


//void Processing::onDevelopSettings_Accept(int fps, int cod)
//{
//    switch (cod){
//    case (1):
//        devThread->codec = VideoWriter::fourcc('I','Y','U','V');
//        sprintf(devThread->file_type, "avi");
//        qDebug() << "Codec set to IYUV";
//        break;
//    case (2):
//        //devThread->codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
//        devThread->codec = VideoWriter::fourcc('M', 'P', '4', 'V');
//        sprintf(devThread->file_type, "mp4");
//        qDebug() << "Codec set to MJPG";
//        break;
 //   default:
//        devThread->codec = VideoWriter::fourcc('I','Y','U','V');
//        qDebug() << "Codec set to IYUV";
//        sprintf(devThread->file_type, "avi");
//        break;
//    }
//
//    devThread->fps = fps;
//    devThread->width = width;
//    devThread->height = height;
//    devThread->Roll = Roll;
//    devThread->frames = imgcount;
//    devThread->start();
//
//}


void Processing::onFrameDeveloped(int i)
{
    cv::Mat show = devThread->imgBGR.clone(); // <-- deep copy, not a ref

    if (show.depth() == CV_16U) {
        show.convertTo(show, CV_8UC3, 255.0 / 4095.0);
    }


    cv::cvtColor(show, show, cv::COLOR_BGR2RGB);
    QImage qimg(show.data, show.cols, show.rows, int(show.step), QImage::Format_RGB888);

    setPreview(qimg.copy());  // <-- hand an owned image to the UI

    ui->label_FrameView->setNum(i);
}


void Processing::on_Back_Button_clicked()
{
    ImgViewCount--;
    if (ImgViewCount < 0){
        ImgViewCount = 0;
    }
    else{
    ui->horizontalSlider_3->setValue(ImgViewCount);
    }
}

void Processing::on_LastButton_clicked()
{
    ImgViewCount = imgcount;

  //  ImgViewCount = std::max(1, imgcount);
  //  ui->horizontalSlider_3->setValue(ImgViewCount);

    // //ui->horizontalSlider_3->setValue(ImgViewCount);

    on_horizontalSlider_3_valueChanged(ImgViewCount);
    qDebug() << "Imageviewcount value is: " << ImgViewCount;

}



void Processing::on_ForwardButton_clicked()
{
    ImgViewCount++;
    if (ImgViewCount > imgcount){
        ImgViewCount = imgcount;
    }
    else{
    ui->horizontalSlider_3->setValue(ImgViewCount);
    }
}

void Processing::on_BaseCurveCombo_currentIndexChanged(int index)
{
    devThread->baseCurveMode = index;



    // Ensure sane defaults
  //  devThread->useLog = false;
  //  devThread->HDR    = false;

  //  switch (index)
  //  {
  //  case 0: // None (Linear)
  //      // Linear base → only exposure + CCM in DevelopThread
  //      devThread->useLog = false;
  //      devThread->HDR    = false;
  //      break;
//
  //  case 1: // LOG16 (Sony-style)
  //      devThread->useLog = true;
  //      devThread->HDR    = false;
  //      break;
//
  //  case 2: // ACES Filmic (HDR)
  //      devThread->useLog = false;
  //      devThread->HDR    = true;
  //      break;
//
    //default:
    //    // Fallback: linear
    //    devThread->useLog = false;
    //    devThread->HDR    = false;
    //    break;
    //}

    // Re-develop current frame so user sees effect immediately
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_Red_Slider_valueChanged(int value)
{
    devThread->Red = (float) (value/100.0);
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_Blue_Slider_valueChanged(int value)
{
    devThread->Blue = (float) (value/100.0);
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_Green_Slider_valueChanged(int value)
{
    devThread->Green = (float) (value/100.0);
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}



void Processing::on_Brightness_slider_valueChanged(int value)
{
    devThread->Bright = value;
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_Saturation_slider_valueChanged(int value)
{
      devThread->Sat = value;
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_Contrast_slider_valueChanged(int value)
{
    devThread->Contrast = saturate_cast<float> (value);
     on_horizontalSlider_3_valueChanged(ImgViewCount);

}

void Processing::on_ExposureCombo_currentIndexChanged(int index)
{
    // Half-stop steps from -2.0 EV to +2.0 EV
    static const float evValues[9] = {
        -2.0f,  // index 0: -2.0 EV
        -1.5f,  // index 1: -1.5 EV
        -1.0f,  // index 2: -1.0 EV
        -0.5f,  // index 3: -0.5 EV
        0.0f,  // index 4:  0.0 EV
        0.5f,  // index 5: +0.5 EV
        1.0f,  // index 6: +1.0 EV
        1.5f,  // index 7: +1.5 EV
        2.0f   // index 8: +2.0 EV
    };

    if (index < 0 || index >= 9)
        return;

    devThread->ExposureEV = evValues[index];

    // Re-develop current frame so the user sees the change immediately
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

//void Processing::onStartFromCurrentFrame(bool currentframe)
//{
//
//    if (currentframe == true){
//        devThread->first_frame = ImgViewCount;
//    }
//    else{
//        devThread->first_frame = 1;
//    }
//}

void Processing::on_bitDepth_valueChanged(int arg1)
{
    devThread->bitdepth = arg1;
}



void Processing::onStatusUpdate(QString str, bool alarm)
{
    ui->StatusUpdate->setText(str);
    QPalette palette = ui->StatusUpdate->palette();
    palette.setColor(ui->StatusUpdate->backgroundRole(), Qt::yellow);
    ui->StatusUpdate->setPalette(palette);
    ui->StatusUpdate->setAutoFillBackground(alarm);
}

void Processing::on_HDR_Develop_clicked(bool)
{
    // HDR mode is no longer part of the simplified pipeline.
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}


void Processing::on_SCurveBox_clicked(bool checked)
{
    filmlook = checked;
    devThread->filmlook = checked;
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_spinBox_RedCurve_valueChanged(int arg1)
{
    devThread->CurveRed = (double)arg1;
    devThread->calc_LutCurve();
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_spinBox_GreenCurve_valueChanged(int arg1)
{
   devThread->CurveGreen = (double)arg1;
   devThread->calc_LutCurve();
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_spinBox_BlueCurve_valueChanged(int arg1)
{
    devThread->CurveBlue = (double)arg1;
    devThread->calc_LutCurve();
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_SelectDir_Button_clicked()
{
     char *path;
    QFileDialog dialog(this);

    dialog.setFileMode(QFileDialog::Directory);
    dialog.setOption(QFileDialog::ShowDirsOnly);
    dialog.setViewMode(QFileDialog::List);
    QString WindowName = "Open WorkDirectory";
    dialog.setWindowTitle(WindowName);
    QString dirName;
    QByteArray ba;
    if (dialog.exec())
        dirName = dialog.selectedFiles()[0];
    ba = dirName.toLocal8Bit();
    path = ba.data();
    strcpy_s(work_path,path);
    strcpy_s(devThread->work_path, path);
    sprintf(imgcounter_filename, "%s/imgcounter%d.txt", work_path, Roll);
    imgcount = GetNumber(imgcounter_filename);
    onUpdateUI();
    qDebug() << work_path;
}

void Processing::on_GammaCorrectionBox_clicked(bool)
{
    // No separate gamma mode in the simplified pipeline.
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

//Here enter the functions that will be used across the classes

//Mat PrepareFrameForDisplay(char work_path[256], int Roll, int frame, float Blue, float Green, float Red, float Bright, float Sat, float Contrast, bool filmlook, bool gammacorrect, ushort *lut16, ushort *lutSCurveRed, ushort *lutSCurveGreen, ushort *lutSCurveBlue)
//{
//    char frame_name[256];
//    sprintf(frame_name, "%s/images%d/XiCapture%03d.pgm", work_path, Roll, frame);
//    Mat img;
//    Mat img_HSV;
//    Mat imgBAY;
//    vector<Mat> channels; //HSVchannels;
//    float factor = (256 * (Contrast + 255)) / (255 * (256 - Contrast));
//    imgBAY = imread(frame_name, IMREAD_ANYDEPTH | IMREAD_ANYCOLOR);
//    if(imgBAY.empty()){
//                qDebug() << "FROM function PrepareFrameForDisplay: No data in image, frame-name is: " << frame_name << "Roll is: " << Roll;
//                return(img);
//            }
//    cvtColor(imgBAY, img, COLOR_BayerGB2RGB, 3);//26072024: this line works and colors are displayed correct on screen
//    flip(img, img, -1);// 26072024: added this to flip the image upside down.
//
//    if(img.depth() == 2){ // 12 bit image case
//    img = LOG16(img, lut16);
//    if(filmlook) FilmLook16(img, img, 48, 0.6, 192, 0.3, lutSCurveRed, lutSCurveGreen, lutSCurveBlue);
//    img.convertTo(img, CV_32FC3, (1./4095.), 0);
//
//    cvtColor(img, img_HSV, COLOR_RGB2HSV_FULL); //change the color image from BGR to YCrCb format
//    split(img_HSV,channels); //split the image into channels
//    channels[2].convertTo(channels[2], -1, (Contrast/100.0), (Bright/255.));
//    channels[1].convertTo(channels[1], -1, (Sat/100.0), 0);
//    merge(channels, img_HSV);
//    cvtColor(img_HSV, img, COLOR_HSV2RGB_FULL);
//
//    split(img,channels); //split the image into channels
//
//    channels[0].convertTo(channels[0], -1, Blue, 0);
//    channels[1].convertTo(channels[1], -1, Green, 0);
//    channels[2].convertTo(channels[2], -1, Red, 0);
//
//    merge(channels,img);
//    img.convertTo(img, CV_8UC3, 255.0, 0);
//    if(gammacorrect){
//        img = correctGamma(img, (1.0/2.2));
//    }
//   }
//    else{ // 8 bit image case
//    img = correctGamma(img, 2.2);
//    cvtColor(img, img_HSV, COLOR_BGR2HSV); //change the color image from BGR to YCrCb format
//    split(img_HSV,channels); //split the image into channels
//
//    channels[2].convertTo(channels[2], -1, (Contrast/100.0), Bright);
//    channels[1].convertTo(channels[1], -1, (Sat/100.0), 0);
//
//    merge(channels, img_HSV);
//    cvtColor(img_HSV, img, COLOR_HSV2BGR);
//
//    split(img,channels); //split the image into channels
//
//   channels[0].convertTo(channels[0], -1, Blue, 0);
//    channels[1].convertTo(channels[1], -1, Green, 0);
//    channels[2].convertTo(channels[2], -1, Red, 0);
//
//    merge(channels,img);
//    if (filmlook){
//        cvtColor(img,img, COLOR_BGR2RGB);
//        img = FilmLook(img, 48, 0.6, 192, 0.3);
//        cvtColor(img,img, COLOR_RGB2BGR);
//    }
//    }
//
//   return (img);
//
//}

bool copy_file( const char* srce_file, const char* dest_file )
{
    ifstream srce( srce_file, ios::binary ) ;
    if(!srce.is_open())
    {
      qDebug() << "error! the file doesn't exist";
      return(false);
    }
    else {

        std::ofstream dest( dest_file, std::ios::binary ) ;
        dest << srce.rdbuf() ;
    return(true);
    }
}

cv::Mat correctGamma( Mat& img, double gamma) {
    double inverse_gamma = 1.0 / gamma;
    Mat lut_matrix(1, 256.0, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ ){
        ptr[i] = (int)( pow( (double) i / (255.0), inverse_gamma ) * (255.0) );
        }
    Mat result;
    LUT( img, lut_matrix, result );
    return result;
}

cv::Mat LOG8( Mat& img, Mat& lut8){
    Mat result;
    LUT( img, lut8, result );
    return result;
}

cv::Mat LOG16(Mat& img, ushort *lut16)
{
    Mat dst = img.clone();
    const int channels = dst.channels();
    switch (channels)
    {
    case 1:
    {
    MatIterator_<ushort> it, end;
    for (it = dst.begin<ushort>(), end = dst.end<ushort>(); it != end; it++)
    *it = lut16[(*it)];
    break;

    }
    case 3:
    {
    MatIterator_<Vec3w> it, end;

    for (it = dst.begin<Vec3w>(), end = dst.end<Vec3w>(); it != end; it++)
    {
    (*it)[0] = lut16[((*it)[0])];
    (*it)[1] = lut16[((*it)[1])];
    (*it)[2] = lut16[((*it)[2])];
    }
     break;
    }
    }
    img = dst.clone();
    return(img);

}

void GammaCorrection(Mat& src, Mat& dst, double fGamma )
{
ushort lut[4096];
for (int i = 0; i < 4096; i++)
{
lut[i] = saturate_cast<ushort>(pow((double)(i / 4095.0), fGamma) * 4095.0);
}

dst = src.clone();
const int channels = dst.channels();
switch (channels)
{
case 1:
{
MatIterator_<ushort> it, end;
for (it = dst.begin<ushort>(), end = dst.end<ushort>(); it != end; it++)
*it = lut[(*it)];
break;

}
case 3:
{
MatIterator_<Vec3w> it, end;

for (it = dst.begin<Vec3w>(), end = dst.end<Vec3w>(); it != end; it++)
{

(*it)[0] = lut[((*it)[0])];
(*it)[1] = lut[((*it)[1])];
(*it)[2] = lut[((*it)[2])];
}
break;
}
}
}

cv::Mat FilmLook(Mat& img, int kneelow, double RClow, int kneehigh, double RChigh) {
    Mat lutkodak_matrix(1, 256.0, CV_8UC1);
    Mat lutkodak_matrix2(1, 256.0, CV_8UC1);
    vector<Mat> channels;
    uchar * ptrkodak = lutkodak_matrix.ptr();
    uchar * ptrkodak2 = lutkodak_matrix2.ptr();
    for( int i = 0; i < 256; i++ ){
        ptrkodak[i] = (int)((1/(1+ exp((double)(-(12.0/255.0)*((i-128)-0))))) * 255.0);
     }
    Mat result, resultHSV;
    LUT( img, lutkodak_matrix, result);
    return result;
}


void FilmLook16(Mat& src, Mat& dst, double kneelow, double RClow, double kneehigh, double RChigh, ushort *lutRed, ushort *lutGreen, ushort *lutBlue)
{
dst = src.clone();
const int channels = dst.channels();
switch (channels)
{
case 1:
{
break;

}
case 3:
{
MatIterator_<Vec3w> it, end;
for (it = dst.begin<Vec3w>(), end = dst.end<Vec3w>(); it != end; it++)
{
(*it)[0] = lutRed[((*it)[0])];
(*it)[1] = lutGreen[((*it)[1])];
(*it)[2] = lutBlue[((*it)[2])];
}
break;
}
}

}

int GetNumber(const char* number_file)
{
    ifstream counterfile;
    int count;
    counterfile.open (number_file, ios::in | ios::binary);
    counterfile >> count;
    counterfile.close();
    return count;
}

void Processing::setPreview(const QImage& img)
{
    m_preview = img;          // keep original; we’ll rescale on demand
    refreshPreview();
}

void Processing::resizeEvent(QResizeEvent* e)
{
    QMainWindow::resizeEvent(e);
    refreshPreview();         // rescale when the window changes size / fullscreen
}

void Processing::refreshPreview()
{
    if (m_preview.isNull() || !ui->Label_showimage)
        return;

    const QSize labelSize = ui->Label_showimage->size();
    if (labelSize.isEmpty())
        return;

    // HiDPI-aware target size
    const qreal dpr = devicePixelRatioF();
    const QSize targetPx(qMax(1, int(labelSize.width()  * dpr)),
                         qMax(1, int(labelSize.height() * dpr)));

    // Scale with aspect ratio + good quality
    QImage scaled = m_preview.scaled(targetPx, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Tag DPR so it stays crisp on Retina/4K
    QPixmap pm = QPixmap::fromImage(scaled);
    pm.setDevicePixelRatio(dpr);

    ui->Label_showimage->setPixmap(pm);
}

void Processing::on_playButton_clicked()
{
    PlayingVideo = true;
    // Ensure devThread has the current capture context (you already set these elsewhere, but it’s safe):
    devThread->width  = width;
    devThread->height = height;
    devThread->Roll   = Roll;

    // Playback from current slider position to the last available frame
    const int startAt = ui->horizontalSlider_3->value();
    const int endAt   = imgcount;  // inclusive last index

    // FPS: read from ui->fpsCombo (fallback 24 if parse fails)
    bool ok = false;
    double fps = ui->fpsCombo->currentText().toDouble(&ok);
    if (!ok || fps <= 0.0) fps = 24.0;

    devThread->setRange(startAt, endAt);
    devThread->setFps(fps);
    devThread->setLooping(ui->loopCheck->isChecked());


    devThread->play();
}

void Processing::on_pauseButton_clicked()
{
    devThread->pause();
    PlayingVideo = false;
}

void Processing::on_stopButton_clicked()
{
    // Stop live playback (if any)
    devThread->stop();

    // Also cancel an in-progress RenderToFile (safe no-op if not rendering)
    devThread->cancelRender();
    PlayingVideo = false;
}


// While dragging, pause playback so you don't fight the timer
void Processing::on_horizontalSlider_3_sliderPressed()
{
    devThread->pause();
    PlayingVideo = false;
}

void Processing::on_horizontalSlider_3_valueChanged(int i)
{
    if(PlayingVideo || devThread->renderingBusy) return;
    //ImgViewCount = std::max(1, i);//ensures minimum is 1
  //   ImgViewCount = std::max(1, i);

    devThread->width  = width;
    devThread->height = height;
    devThread->Roll   = Roll;
    ImgViewCount = i;

    devThread->seek(i);
    // Optional immediate refresh (one-off render):
    devThread->onDevelopFrame(i);
    //emit DevelopFrame(i);
    //PlayingVideo = false;
}

// When the user releases the slider, resume if desired:
void Processing::on_horizontalSlider_3_sliderReleased()
{
  //  if (/* was playing before drag */) devThread->play();
}

void Processing::applyEmulsionPreset(const EmulsionPreset &p)
{
    ui->SCurveBox->setChecked(true);

    // In the simplified pipeline, the preset only drives look controls.
    ui->spinBox_RedCurve->setValue(static_cast<int>(p.sCurveR));
    ui->spinBox_GreenCurve->setValue(static_cast<int>(p.sCurveG));
    ui->spinBox_BlueCurve->setValue(static_cast<int>(p.sCurveB));

    // exposureEV in your preset is a float, but ExposureCombo uses fixed indices.
    // Map the common half-stop values back to combo indices.
    int expIndex = 4; // default 0 EV
    if      (p.exposureEV <= -1.75f) expIndex = 0;
    else if (p.exposureEV <= -1.25f) expIndex = 1;
    else if (p.exposureEV <= -0.75f) expIndex = 2;
    else if (p.exposureEV <= -0.25f) expIndex = 3;
    else if (p.exposureEV <=  0.25f) expIndex = 4;
    else if (p.exposureEV <=  0.75f) expIndex = 5;
    else if (p.exposureEV <=  1.25f) expIndex = 6;
    else if (p.exposureEV <=  1.75f) expIndex = 7;
    else                             expIndex = 8;

    ui->ExposureCombo->setCurrentIndex(expIndex);

    ui->Brightness_slider->setValue(static_cast<int>(p.brightness));
    ui->Contrast_slider->setValue(static_cast<int>(p.contrast));
    ui->Saturation_slider->setValue(static_cast<int>(p.saturation));

    ui->Red_Slider->setValue(static_cast<int>(p.rGain));
    ui->Green_Slider->setValue(static_cast<int>(p.gGain));
    ui->Blue_Slider->setValue(static_cast<int>(p.bGain));
}

void Processing::on_comboEmulsionLook_currentIndexChanged(int index)
{
    // index 0..3 moet overeenkomen met enum EmulsionLook
    EmulsionLook look = static_cast<EmulsionLook>(index);

    EmulsionPreset p = getEmulsionPreset(look);
    applyEmulsionPreset(p);
}

void Processing::onLogModeChanged(int)
{
    // Kept for UI compatibility, but no longer switches pipeline branches.
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::onToneCurveChanged(int)
{
    // Kept for UI compatibility, but no longer switches pipeline branches.
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_comboGradingMode_currentIndexChanged(int)
{
    // Kept for UI compatibility, but no longer switches pipeline branches.
    on_horizontalSlider_3_valueChanged(ImgViewCount);
}
void Processing::on_sliderSCurvePivot_valueChanged(int value)
{
    if (devThread) {
        devThread->setSCurvePivot(value);
    }

    // Retrigger a develop of current frame if that’s your pattern:
     on_horizontalSlider_3_valueChanged(ImgViewCount);
}

void Processing::on_actionExport_CinemaDNG_triggered()
{
    // Gather some context: current roll folder, RAW files, etc.
  //  QString defaultOutputFolder;
  //  QStringList inputFiles;

    // Example: if DevelopThread knows the current roll / source folder:
    // defaultOutputFolder = m_developThread->currentRollFolder();
    // inputFiles = m_developThread->currentRawFileList();

    DngBatchDialog dlg(this);
  //  dlg.setDefaultOutputFolder(defaultOutputFolder);
  //  dlg.setInputFiles(inputFiles);
    dlg.setModal(true);
    dlg.exec();  // modal dialog; use show() if you prefer non-modal
    //dlg.show();
}

void Processing::applyTheme(const QString &themeId)
{
    QString qssPath;

    if (themeId == "kodak_dark") {
        qssPath = ":/themes/kodak_dark.qss";
        ui->actionThemeKodakDark->setChecked(true);
    } else {
        qssPath = ":/themes/kodak_yellow.qss";
        ui->actionThemeKodakLight->setChecked(true);
    }

    QFile f(qssPath);
    if (!f.open(QFile::ReadOnly | QFile::Text))
        return;

    const QString style = QString::fromUtf8(f.readAll());
    qApp->setStyleSheet(style);

    // remember choice
    QSettings s("STEEMERS", "DigitalSuper8Processing");
    s.setValue("ui/theme", themeId);
}
void Processing::setupThemeActions()
{
    m_themeActionGroup = new QActionGroup(this);
    m_themeActionGroup->setExclusive(true);

    ui->actionThemeKodakLight->setCheckable(true);
    ui->actionThemeKodakDark->setCheckable(true);

    m_themeActionGroup->addAction(ui->actionThemeKodakLight);
    m_themeActionGroup->addAction(ui->actionThemeKodakDark);

    // Connect
    connect(ui->actionThemeKodakLight, &QAction::triggered,
            this, &Processing::onThemeKodakLight);
    connect(ui->actionThemeKodakDark, &QAction::triggered,
            this, &Processing::onThemeKodakDark);
}

void Processing::onThemeKodakLight()
{
    applyTheme("kodak_yellow");
}

void Processing::onThemeKodakDark()
{
    applyTheme("kodak_dark");
}
