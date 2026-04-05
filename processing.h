#ifndef PROCESSING_H
#define PROCESSING_H
#include "developthread.h"
#include "super8devparams.h"

#include <QMainWindow>


using namespace std;
using namespace cv;

namespace Ui {
class SCurvePreviewLUT;
class Processing;
}

enum class EmulsionLook {
    None = 0,
    Ektachrome100D,
    Vision3_500T,
    Wolfen400
};

struct EmulsionPreset {
    int   toneCurveMode;
    float exposureEV;
    float sCurveR, sCurveG, sCurveB;
    float brightness;
    float contrast;
    float saturation;
    float rGain, gGain, bGain;
};

//struct Super8DevParams
//{
//    bool enabled = false;
//
//    // 0..1 intensities (we’ll map sliders 0..100 -> 0..1)
//    float scratches = 0.10f;
//    float dust      = 0.05f;
//    float lightLeak = 0.10f;
//    float weave     = 0.03f;
//    float grain     = 0.15f;
//
//    // fade lengths / strengths
//    int   fadeInWarmthFrames = 12;   // 0..120
//    float fadeOutYellowShift = 0.70f; // 0..1
//};


class Processing : public QMainWindow
{
    Q_OBJECT
//class SCurvePreviewLUT; // forward declaration
public:
    explicit Processing(QWidget* parent = nullptr);
    ~Processing();

    void setPreview(const QImage& img);   // call whenever you have a new frame
    // (optional) convenience if you use OpenCV:
    void setPreview(const uchar* data, int w, int h, int bytesPerLine, QImage::Format fmt);



protected:
    void resizeEvent(QResizeEvent* e) override;
    double CurveRed, CurveGreen, CurveBlue;

    QThread* devQThread = nullptr;
    DevelopThread* devThread = nullptr;

     bool filmlook;
     bool gammacorrect;
      char imgcounter_filename[256];
      int Roll, ASA, width, height, imgcount;

      ushort lut16[4096];
      ushort lutSCurveRed[4096];
      ushort lutSCurveGreen[4096];
      ushort lutSCurveBlue[4096];
      char work_path[256];

signals:
    void DevelopFrame(int);
    void super8DevParamsChanged(const Super8DevParams &p);

public slots:
    void onLogModeChanged(int index);
    void onToneCurveChanged(int index);
    void on_ExposureCombo_currentIndexChanged(int index);
    void onExposureChanged(int);
    void onRollChanged(int,int,int,int,int,int,int);
    void onASAChanged(int, float);
    void onUpdateUI();
 //   void onDevelopSettings_Accept(int, int);
    void onFrameDeveloped (int);
 //   void onStartFromCurrentFrame(bool);
    void onStatusUpdate(QString, bool);
//    void onFrameName(char);

 //   void calc_LutCurve();


private slots:
    void onThemeKodakLight();
    void onThemeKodakDark();
    void on_actionExport_CinemaDNG_triggered();  // auto-connected by name if you like
    void on_sliderSCurvePivot_valueChanged(int value);
     void on_comboGradingMode_currentIndexChanged(int index);
     void on_comboEmulsionLook_currentIndexChanged(int index);
    void on_BaseCurveCombo_currentIndexChanged(int index);
    void on_playButton_clicked();
    void on_pauseButton_clicked();
    void on_stopButton_clicked();

    void on_horizontalSlider_3_sliderPressed();
 //   void on_horizontalslider_3_valueChanged(int);
    void on_horizontalSlider_3_sliderReleased();

    void on_Settings_Button_clicked();

    void on_DevelopFilm_Button_clicked();

   // void on_pushButton_6_clicked();

    void on_Back_Button_clicked();

    void on_LastButton_clicked();

    void on_ForwardButton_clicked();

    void on_horizontalSlider_3_valueChanged(int value);

      void on_Red_Slider_valueChanged(int value);

    void on_Blue_Slider_valueChanged(int value);

    void on_Green_Slider_valueChanged(int value);

    void on_Brightness_slider_valueChanged(int value);

    void on_Saturation_slider_valueChanged(int value);

    void on_Contrast_slider_valueChanged(int value);

    void on_bitDepth_valueChanged(int arg1);

//    void on_Bitdepth_SettingChanged(int arg1);

    void on_HDR_Develop_clicked(bool checked);

     void on_SCurveBox_clicked(bool checked);

     void on_spinBox_RedCurve_valueChanged(int arg1);

     void on_spinBox_GreenCurve_valueChanged(int arg1);

     void on_spinBox_BlueCurve_valueChanged(int arg1);

     void on_SelectDir_Button_clicked();

     void on_GammaCorrectionBox_clicked(bool checked);


private:
     // Helper to actually load and apply a theme
     void applyTheme(const QString &themeId);   // <-- this is what you asked for
    void setupThemeActions();

    QActionGroup *m_themeActionGroup = nullptr;
    QString m_currentThemeId;

     EmulsionPreset getEmulsionPreset(EmulsionLook look) const;
     void applyEmulsionPreset(const EmulsionPreset &p);
     void refreshPreview();                // scale cached image to label

    Ui::Processing* ui = nullptr;
    QImage m_preview;
    cv::Mat img;
    int ImgViewCount;
    bool PlayingVideo;

    Super8DevParams m_super8Dev;
    Super8DevParams readSuper8DevParamsFromUi() const;
    void hookUpSuper8DevUi();
    QTimer *m_super8PreviewTimer = nullptr;//Debounce timer for sliders for super8 effects
   // bool RenderingVideo;
//    int wid_bitdepth;
    //QMutex mutex;
};

//cv::Mat PrepareFrameForDisplay(char work_path[256], int Roll, int frame, float Blue, float Green, float Red, float Bright, float Sat, float Contrast, bool filmlook, bool gammacorrect, ushort *lut16, ushort *lutSCurveRed, ushort *lutSCurveGreen, ushort *lutSCurveBlue);
bool copy_file( const char* srce_file, const char* dest_file );
cv::Mat correctGamma(Mat& img, double gamma );
void GammaCorrection(Mat& src, Mat& dst, double fGamma);
void FilmLook16(Mat& src, Mat& dst, double kneelow, double RClow, double kneehigh, double RChigh, ushort *lutRed, ushort *lutGreen, ushort *lutBlue);
Mat Scratch(Mat& img1);
cv::Mat FilmLook(Mat& img, int kneelow, double RClow, int kneehigh, double RChigh);
cv::Mat LOG8( Mat& img, Mat& lut8);
cv::Mat LOG16(Mat& img, ushort *lut16);

int GetNumber(const char* number_file);

#endif // WIDGET_H
