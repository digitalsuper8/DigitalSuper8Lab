#include "ds8videomuxer.h"

#include <QFileInfo>
#include <QProcess>
#include <QStringList>

DS8VideoMuxer::Result DS8VideoMuxer::muxAudioIntoVideo(const QString &videoPath,
                                                       const QString &wavPath,
                                                       const QString &outputPath,
                                                       const QString &ffmpegProgram)
{
    Result r;
    r.outputPath = outputPath;

    if (!QFileInfo::exists(videoPath)) {
        r.error = QStringLiteral("Video file not found: %1").arg(videoPath);
        return r;
    }

    if (!QFileInfo::exists(wavPath)) {
        r.error = QStringLiteral("WAV file not found: %1").arg(wavPath);
        return r;
    }

    const QString ext = QFileInfo(outputPath).suffix().toLower();

    QStringList args;
    args << "-y"
         << "-i" << videoPath
         << "-i" << wavPath
         << "-map" << "0:v:0"
         << "-map" << "1:a:0"
         << "-c:v" << "copy";

    if (ext == "mp4" || ext == "m4v") {
        args << "-c:a" << "aac" << "-b:a" << "192k";
    } else {
        args << "-c:a" << "pcm_s16le";
    }

    args << "-shortest"
         << outputPath;

    QProcess p;
    p.start(ffmpegProgram, args);

    if (!p.waitForStarted()) {
        r.error = QStringLiteral("Could not start ffmpeg. Put ffmpeg.exe in PATH or next to the app.");
        return r;
    }

    p.waitForFinished(-1);

    r.stdOut = QString::fromLocal8Bit(p.readAllStandardOutput());
    r.stdErr = QString::fromLocal8Bit(p.readAllStandardError());

    if (p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
        r.error = QStringLiteral("ffmpeg failed:\n%1").arg(r.stdErr);
        return r;
    }

    r.ok = true;
    return r;
}