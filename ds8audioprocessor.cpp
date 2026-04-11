#include "ds8audioprocessor.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QDataStream>
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

    if (srcBytes.isEmpty()) {
        err = QStringLiteral("No source PCM bytes.");
        return out;
    }

    const int srcSamples = srcBytes.size() / 2;
    if (srcSamples <= 0) {
        err = QStringLiteral("No source samples.");
        return out;
    }

    if (targetSamples <= 0) {
        err = QStringLiteral("Target sample count invalid.");
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

    if (firstFrame < 1) firstFrame = 1;
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

    QByteArray concatenated;
    int filesUsed = 0;

    for (int frame = firstFrame; frame <= lastFrame; ++frame) {
        const QString pcmPath =
            QStringLiteral("%1/XiCapture%2.pcm")
                .arg(audioDirPath)
                .arg(frame, 3, 10, QChar('0'));

        if (!QFileInfo::exists(pcmPath))
            continue;

        QByteArray bytes;
        QString err;
        if (!readPcmFile(pcmPath, bytes, err))
            continue;

        concatenated.append(bytes);
        ++filesUsed;
    }

    if (concatenated.isEmpty() || filesUsed == 0) {
        r.error = QStringLiteral("No usable PCM files found in %1 for frames %2..%3.")
                      .arg(audioDirPath)
                      .arg(firstFrame)
                      .arg(lastFrame);
        return r;
    }

    const int sourceSamples = concatenated.size() / 2;
    const int frameCount = (lastFrame - firstFrame + 1);
    const double seconds = static_cast<double>(frameCount) / fps;
    const int targetSamples = std::max(1, static_cast<int>(std::llround(seconds * settings.sampleRate)));

    QString err;
    const QByteArray stretched = stretchPcm16Mono(concatenated, targetSamples, err);
    if (stretched.isEmpty()) {
        r.error = err;
        return r;
    }

    QDir().mkpath(QFileInfo(outputWavPath).absolutePath());

    if (!writeWavFile(outputWavPath, stretched, settings, err)) {
        r.error = err;
        return r;
    }

    r.ok = true;
    r.wavPath = outputWavPath;
    r.sourceSamples = sourceSamples;
    r.targetSamples = targetSamples;
    r.filesUsed = filesUsed;
    return r;
}