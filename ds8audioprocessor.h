#ifndef DS8AUDIOPROCESSOR_H
#define DS8AUDIOPROCESSOR_H

#include <QObject>
#include <QString>
#include <QByteArray>
#include <QList>

class DS8AudioProcessor : public QObject
{
    Q_OBJECT
public:
    explicit DS8AudioProcessor(QObject *parent = nullptr);

    struct Settings
    {
        int sampleRate;
        int channels;
        int bitsPerSample;

        // Nieuw:
        int blockFrames;             // bv 18 of 36
        double outlierLongFactor;    // bv 1.8
        double outlierCapFactor;     // bv 1.2

        Settings()
            : sampleRate(48000)
            , channels(1)
            , bitsPerSample(16)
            , blockFrames(18)
            , outlierLongFactor(1.8)
            , outlierCapFactor(1.2)
        {}
    };

    struct Result
    {
        bool ok = false;
        QString error;
        QString wavPath;
        int sourceSamples = 0;
        int targetSamples = 0;
        int filesUsed = 0;
    };

    static Result buildStretchedWavForRoll(const QString &workPath,
                                           int roll,
                                           int firstFrame,
                                           int lastFrame,
                                           double fps,
                                           const QString &outputWavPath,
                                           const Settings &settings = Settings());

private:
    static bool readPcmFile(const QString &path, QByteArray &bytes, QString &err);

    static bool writeWavFile(const QString &outPath,
                             const QByteArray &pcmData,
                             const Settings &settings,
                             QString &err);

    static QByteArray stretchPcm16Mono(const QByteArray &srcBytes,
                                       int targetSamples,
                                       QString &err);

    static QByteArray makeSilence16Mono(int sampleCount);

    static QString findPcmPathForFrame(const QString &audioDirPath, int frame);

    static double medianOfInts(const QList<int> &values);
};

#endif // DS8AUDIOPROCESSOR_H
