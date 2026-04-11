#ifndef DS8VIDEOMUXER_H
#define DS8VIDEOMUXER_H

#include <QString>

class DS8VideoMuxer
{
public:
    struct Result
    {
        bool ok = false;
        QString error;
        QString outputPath;
        QString stdOut;
        QString stdErr;
    };

    static Result muxAudioIntoVideo(const QString &videoPath,
                                    const QString &wavPath,
                                    const QString &outputPath,
                                    const QString &ffmpegProgram = QStringLiteral("ffmpeg"));
};

#endif // DS8VIDEOMUXER_H