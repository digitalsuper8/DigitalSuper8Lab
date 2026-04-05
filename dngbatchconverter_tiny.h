#pragma once

#include <QObject>
#include <QDir>
#include <QAtomicInt>
#include <QFileInfoList>
#include <QMutex>
#include <QPointer>
#include <QByteArray>

// TinyDNG-based CinemaDNG writer (no external process, no libtiff)
// -----------------------------------------------------------------------------
// This Qt6 QObject scans /media/microSD/images<roll>/XiCapture*.pgm (16-bit PGM
// with maxval 4095), reads RAW Bayer planes, and writes CinemaDNG-compatible
// DNGs to a mounted USB volume (label SUPER8 by default) using TinyDNG
// (https://github.com/syoyo/tinydng, MIT). Output is uncompressed DNG to avoid
// extra dependencies; you can enable lossless JPEG later if desired.
// -----------------------------------------------------------------------------
class DngBatchConverterTiny : public QObject {
    Q_OBJECT
    Q_PROPERTY(QString workPath READ workPath WRITE setWorkPath NOTIFY pathsChanged)
    Q_PROPERTY(QString usbPath READ usbPath WRITE setUsbPath NOTIFY pathsChanged)
    Q_PROPERTY(QString usbLabel READ usbLabel WRITE setUsbLabel NOTIFY pathsChanged)
    Q_PROPERTY(QString cfaPattern READ cfaPattern WRITE setCfaPattern NOTIFY parametersChanged)
    Q_PROPERTY(int blackLevel READ blackLevel WRITE setBlackLevel NOTIFY parametersChanged)
    Q_PROPERTY(int whiteLevel READ whiteLevel WRITE setWhiteLevel NOTIFY parametersChanged)
    Q_PROPERTY(QString whiteBalance READ whiteBalance WRITE setWhiteBalance NOTIFY parametersChanged)
    Q_PROPERTY(QString dcpPath READ dcpPath WRITE setDcpPath NOTIFY parametersChanged)
    Q_PROPERTY(QString srcGlob READ srcGlob WRITE setSrcGlob NOTIFY parametersChanged)
    Q_PROPERTY(QString destRollPattern READ destRollPattern WRITE setDestRollPattern NOTIFY parametersChanged)
    Q_PROPERTY(QString destFilePattern READ destFilePattern WRITE setDestFilePattern NOTIFY parametersChanged)

public:
    explicit DngBatchConverterTiny(QObject *parent = nullptr);
    ~DngBatchConverterTiny() override;

    // Config
    QString workPath() const { return m_workPath; }
    void setWorkPath(const QString &p);

    QString usbPath() const { return m_usbPath; }
    void setUsbPath(const QString &p);

    QString usbLabel() const { return m_usbLabel; }
    void setUsbLabel(const QString &label);

    QString cfaPattern() const { return m_cfa; }
    void setCfaPattern(const QString &cfa);

    int blackLevel() const { return m_black; }
    void setBlackLevel(int v);

    int whiteLevel() const { return m_white; }
    void setWhiteLevel(int v);

    QString whiteBalance() const { return m_wb; } // format: "r,g,b" or empty
    void setWhiteBalance(const QString &wb);

    QString dcpPath() const { return m_dcp; }
    void setDcpPath(const QString &path);

    QString srcGlob() const { return m_srcGlob; }
    void setSrcGlob(const QString &glob);

    QString destRollPattern() const { return m_destRollPattern; }
    void setDestRollPattern(const QString &pat);

    QString destFilePattern() const { return m_destFilePattern; }
    void setDestFilePattern(const QString &pat);

    // Extract numeric suffix from filenames like "XiCapture000.pgm" or "XiCapture1000.pgm"
    static int parseIndexFromName(const QString &filename);

public slots:
    void convertAllRolls();
    void convertRoll(int roll);
    void cancel();

signals:
    void StatusUpdateServer(QByteArray, int);
    void progress(int roll, int index, int total);
    void fileConverted(int roll, int index, const QString &srcPgm, const QString &dstDng);
    void rollFinished(int roll, int converted, int skipped, int failed);
    void finished();
    void errorOccured(const QString &message);

    void pathsChanged();
    void parametersChanged();

private:
    struct PgmInfo {
        int width = 0;
        int height = 0;
        int maxval = 0;   // expect 4095
        int headerBytes = 0; // bytes in header before pixel data
        bool bigEndian = true; // PGM spec is big endian for 16-bit
    };

    struct FrameTask {
        int index = -1;          // sequential index from sort order
        QString srcPath;         // full .pgm path
        QString dstPath;         // full .dng path
    };

    static QFileInfoList sortedPgmList(const QDir &dir, const QString &glob);
    static QString rollFolderName(int roll, const QString &pattern);
    static QString destFileName(int roll, int index, const QString &pattern);

    // Extract numeric suffix from filenames like "XiCapture000.pgm" or "XiCapture1000.pgm"
    //static int parseIndexFromName(const QString &filename);

    bool ensureUsbMounted(QString *mountPointOut) const;

    // Core steps
    bool readPgm16(const QString &path, PgmInfo *info, QByteArray *pixelBytes, QString *err);
    bool writeDngTiny(const QString &dstPath,
                      const PgmInfo &info,
                      const quint16 *pixelsLE, // little-endian 16-bit samples (we prepare this)
                      const QString &cfa,
                      int black, int white,
                      const QString &wb,
                      const QString &dcp,
                      QString *err);

    void convertImpl(int roll);

    // data
    QString m_workPath = "/media/microSD";
    QString m_usbPath;                 // if empty, locate by label
    QString m_usbLabel = "SUPER8";

    QString m_cfa = "RGGB";
    int m_black = 64;
    int m_white = 4095;
    QString m_wb;      // e.g. "2.0,1.0,1.5"
    QString m_dcp;     // optional .dcp file path

    QString m_srcGlob = "XiCapture*.pgm";
    QString m_destRollPattern = "A%03d";
    QString m_destFilePattern = "A%03d_%06d.dng";

    QAtomicInt m_cancel{0};
    mutable QMutex m_mutex;
};
