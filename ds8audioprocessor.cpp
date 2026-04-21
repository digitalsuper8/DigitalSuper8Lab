#include "ds8audioprocessor.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDataStream>
#include <QList>

#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

DS8AudioProcessor::DS8AudioProcessor(QObject *parent)
    : QObject(parent)
{
}

bool DS8AudioProcessor::readPcmFile(const QString &path, QByteArray &bytes, QString &err)
{
    QFile f(path);
    if (!f.exists()) {
        err = QStringLiteral("PCM file does not exist: %1").arg(path);
        return false;
    }
    if (!f.open(QIODevice::ReadOnly)) {
        err = QStringLiteral("Could not open PCM file: %1").arg(path);
        return false;
    }

    bytes = f.readAll();
    f.close();

    if (bytes.isEmpty()) {
        err = QStringLiteral("PCM file is empty: %1").arg(path);
        return false;
    }

    if ((bytes.size() % 2) != 0) {
        err = QStringLiteral("PCM file has odd byte count: %1").arg(path);
        return false;
    }

    return true;
}

QByteArray DS8AudioProcessor::stretchPcm16Mono(const QByteArray &srcBytes,
                                               int targetSamples,
                                               QString &err)
{
    QByteArray out;

    if (targetSamples <= 0) {
        err = QStringLiteral("Target sample count invalid.");
        return out;
    }

    if (srcBytes.isEmpty()) {
        err.clear();
        return makeSilence16Mono(targetSamples);
    }

    const int srcSamples = srcBytes.size() / 2;
    if (srcSamples <= 0) {
        err = QStringLiteral("No source samples.");
        return out;
    }

    const qint16 *src = reinterpret_cast<const qint16*>(srcBytes.constData());

    std::vector<qint16> dst;
    dst.resize(static_cast<size_t>(targetSamples));

    if (srcSamples == 1) {
        std::fill(dst.begin(), dst.end(), src[0]);
    } else {
        const double scale = static_cast<double>(srcSamples - 1) /
                             static_cast<double>(std::max(1, targetSamples - 1));

        for (int i = 0; i < targetSamples; ++i) {
            const double pos = i * scale;
            const int idx0 = static_cast<int>(std::floor(pos));
            const int idx1 = std::min(idx0 + 1, srcSamples - 1);
            const double frac = pos - idx0;

            const double s0 = static_cast<double>(src[idx0]);
            const double s1 = static_cast<double>(src[idx1]);
            double y = s0 + (s1 - s0) * frac;

            if (y < -32768.0) y = -32768.0;
            if (y >  32767.0) y =  32767.0;

            dst[static_cast<size_t>(i)] = static_cast<qint16>(std::lround(y));
        }
    }

    out.resize(targetSamples * 2);
    std::memcpy(out.data(), dst.data(), static_cast<size_t>(out.size()));
    return out;
}

QByteArray DS8AudioProcessor::makeSilence16Mono(int sampleCount)
{
    if (sampleCount <= 0)
        return QByteArray();

    QByteArray out;
    out.resize(sampleCount * 2);
    std::memset(out.data(), 0, static_cast<size_t>(out.size()));
    return out;
}

QString DS8AudioProcessor::findPcmPathForFrame(const QString &audioDirPath, int frame)
{
    const QStringList candidates = {
        QStringLiteral("%1/XiCapture%2.pcm").arg(audioDirPath).arg(frame, 3, 10, QChar('0')),
        QStringLiteral("%1/XiCapture%2.pcm").arg(audioDirPath).arg(frame, 4, 10, QChar('0')),
        QStringLiteral("%1/XiCapture%2.pcm").arg(audioDirPath).arg(frame, 5, 10, QChar('0')),
        QStringLiteral("%1/XiCapture%2.pcm").arg(audioDirPath).arg(frame, 6, 10, QChar('0')),
        QStringLiteral("%1/XiCapture%2.pcm").arg(audioDirPath).arg(frame)
    };

    for (const QString &p : candidates) {
        if (QFileInfo::exists(p))
            return p;
    }
    return QString();
}

double DS8AudioProcessor::medianOfInts(const QList<int> &values)
{
    if (values.isEmpty())
        return 0.0;

    std::vector<int> v;
    v.reserve(static_cast<size_t>(values.size()));
    for (int x : values)
        v.push_back(x);

    std::sort(v.begin(), v.end());

    const int n = static_cast<int>(v.size());
    if ((n % 2) == 1)
        return static_cast<double>(v[n / 2]);

    return 0.5 * (static_cast<double>(v[(n / 2) - 1]) + static_cast<double>(v[n / 2]));
}

bool DS8AudioProcessor::writeWavFile(const QString &outPath,
                                     const QByteArray &pcmData,
                                     const Settings &settings,
                                     QString &err)
{
    QFile f(outPath);
    if (!f.open(QIODevice::WriteOnly)) {
        err = QStringLiteral("Could not create WAV file: %1").arg(outPath);
        return false;
    }

    const quint16 audioFormat = 1;
    const quint16 numChannels = static_cast<quint16>(settings.channels);
    const quint32 sampleRate = static_cast<quint32>(settings.sampleRate);
    const quint16 bitsPerSample = static_cast<quint16>(settings.bitsPerSample);
    const quint16 blockAlign = static_cast<quint16>((numChannels * bitsPerSample) / 8);
    const quint32 byteRate = sampleRate * blockAlign;
    const quint32 dataSize = static_cast<quint32>(pcmData.size());
    const quint32 riffSize = 36u + dataSize;

    QDataStream ds(&f);
    ds.setByteOrder(QDataStream::LittleEndian);

    f.write("RIFF", 4);
    ds << riffSize;
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    ds << quint32(16);
    ds << audioFormat;
    ds << numChannels;
    ds << sampleRate;
    ds << byteRate;
    ds << blockAlign;
    ds << bitsPerSample;

    f.write("data", 4);
    ds << dataSize;
    f.write(pcmData);

    f.close();
    return true;
}

DS8AudioProcessor::Result DS8AudioProcessor::buildStretchedWavForRoll(const QString &workPath,
                                                                      int roll,
                                                                      int firstFrame,
                                                                      int lastFrame,
                                                                      double fps,
                                                                      const QString &outputWavPath,
                                                                      const Settings &settings)
{
    Result r;

    if (fps <= 0.0) {
        r.error = QStringLiteral("FPS must be > 0.");
        return r;
    }

    if (settings.channels != 1 || settings.bitsPerSample != 16) {
        r.error = QStringLiteral("This processor currently expects mono 16-bit PCM.");
        return r;
    }

    if (settings.blockFrames <= 0) {
        r.error = QStringLiteral("blockFrames must be > 0.");
        return r;
    }

    if (firstFrame < 1)
        firstFrame = 1;

    if (lastFrame < firstFrame) {
        r.error = QStringLiteral("Invalid frame range.");
        return r;
    }

    const QString audioDirPath = QStringLiteral("%1/audio%2").arg(workPath).arg(roll);
    QDir audioDir(audioDirPath);
    if (!audioDir.exists()) {
        r.error = QStringLiteral("Audio directory not found: %1").arg(audioDirPath);
        return r;
    }

    const int totalFrameCount = lastFrame - firstFrame + 1;
    const int totalTargetSamples = std::max(
        1,
        static_cast<int>(std::llround((static_cast<double>(totalFrameCount) / fps) * settings.sampleRate))
        );

    QByteArray finalPcm;
    int filesUsed = 0;
    int totalSourceSamples = 0;

    for (int blockStart = firstFrame; blockStart <= lastFrame; blockStart += settings.blockFrames) {

        const int blockEnd = std::min(blockStart + settings.blockFrames - 1, lastFrame);
        const int blockFrameCount = blockEnd - blockStart + 1;

        QList<QByteArray> blockFramePcms;
        QList<int> blockSampleCounts;
        blockFramePcms.reserve(blockFrameCount);

        // Eerst alles inlezen om mediaan te bepalen
        for (int frame = blockStart; frame <= blockEnd; ++frame) {
            QByteArray frameBytes;

            const QString pcmPath = findPcmPathForFrame(audioDirPath, frame);
            if (!pcmPath.isEmpty()) {
                QString err;
                if (readPcmFile(pcmPath, frameBytes, err)) {
                    const int sampleCount = frameBytes.size() / 2;
                    if (sampleCount > 0) {
                        blockSampleCounts.append(sampleCount);
                        totalSourceSamples += sampleCount;
                        ++filesUsed;
                    } else {
                        frameBytes.clear();
                    }
                }
            }

            blockFramePcms.append(frameBytes);
        }

        if (blockSampleCounts.isEmpty()) {
            // Hele blok zonder audio -> gewoon exacte stilte voor blokduur
            const int blockTargetSamples = std::max(
                1,
                static_cast<int>(std::llround((static_cast<double>(blockFrameCount) / fps) * settings.sampleRate))
                );
            finalPcm.append(makeSilence16Mono(blockTargetSamples));
            continue;
        }

        const int medianSamples = std::max(
            1,
            static_cast<int>(std::llround(medianOfInts(blockSampleCounts)))
            );

        const int outlierCapSamples = std::max(
            1,
            static_cast<int>(std::llround(static_cast<double>(medianSamples) * settings.outlierCapFactor))
            );

        QByteArray blockConcatenated;

        for (int i = 0; i < blockFramePcms.size(); ++i) {
            QByteArray frameBytes = blockFramePcms[i];

            if (frameBytes.isEmpty()) {
                // ontbrekende PCM -> stilte van normale frame-lengte
                blockConcatenated.append(makeSilence16Mono(medianSamples));
                continue;
            }

            const int sampleCount = frameBytes.size() / 2;
            if (sampleCount <= 0) {
                blockConcatenated.append(makeSilence16Mono(medianSamples));
                continue;
            }

            // Extreme lange outlier? Dan lokaal afkappen vóór concat.
            if (static_cast<double>(sampleCount) >
                (static_cast<double>(medianSamples) * settings.outlierLongFactor))
            {
                QString err;
                QByteArray capped = stretchPcm16Mono(frameBytes, outlierCapSamples, err);
                if (capped.isEmpty()) {
                    r.error = err.isEmpty()
                    ? QStringLiteral("Failed to cap long PCM outlier.")
                    : err;
                    return r;
                }
                blockConcatenated.append(capped);
            } else {
                // normale PCM ongewijzigd laten
                blockConcatenated.append(frameBytes);
            }
        }

        const int blockTargetSamples = std::max(
            1,
            static_cast<int>(std::llround((static_cast<double>(blockFrameCount) / fps) * settings.sampleRate))
            );

        QString err;
        QByteArray stretchedBlock = stretchPcm16Mono(blockConcatenated, blockTargetSamples, err);
        if (stretchedBlock.isEmpty()) {
            r.error = err.isEmpty()
            ? QStringLiteral("Failed to stretch audio block.")
            : err;
            return r;
        }

        finalPcm.append(stretchedBlock);
    }

    if (finalPcm.isEmpty()) {
        r.error = QStringLiteral("No audio could be built.");
        return r;
    }

    // Veiligheidsnet: exact gelijk maken aan totale videolengte.
    if ((finalPcm.size() / 2) != totalTargetSamples) {
        QString err;
        QByteArray corrected = stretchPcm16Mono(finalPcm, totalTargetSamples, err);
        if (corrected.isEmpty()) {
            r.error = err.isEmpty()
            ? QStringLiteral("Final correction stretch failed.")
            : err;
            return r;
        }
        finalPcm = corrected;
    }

    QDir().mkpath(QFileInfo(outputWavPath).absolutePath());

    QString err;
    if (!writeWavFile(outputWavPath, finalPcm, settings, err)) {
        r.error = err;
        return r;
    }

    r.ok = true;
    r.wavPath = outputWavPath;
    r.sourceSamples = totalSourceSamples;
    r.targetSamples = totalTargetSamples;
    r.filesUsed = filesUsed;
    return r;
}
