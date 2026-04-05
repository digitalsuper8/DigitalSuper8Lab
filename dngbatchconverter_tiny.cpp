#include "dngbatchconverter_tiny.h"

#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QProcessEnvironment>

#include <QtEndian>
#include <algorithm>

// TinyDNG writer
// Add `tiny_dng_writer.h` to your project (MIT). No libtiff needed.
// Define TINY_DNG_WRITER_IMPLEMENTATION in exactly one translation unit.
#define TINY_DNG_WRITER_IMPLEMENTATION
#include "tiny_dng_writer.h"  // from https://github.com/syoyo/tinydng

DngBatchConverterTiny::DngBatchConverterTiny(QObject *parent) : QObject(parent) {}
DngBatchConverterTiny::~DngBatchConverterTiny() = default;

void DngBatchConverterTiny::setWorkPath(const QString &p) { QMutexLocker lk(&m_mutex); m_workPath = p; emit pathsChanged(); }
void DngBatchConverterTiny::setUsbPath(const QString &p)  { QMutexLocker lk(&m_mutex); m_usbPath = p;  emit pathsChanged(); }
void DngBatchConverterTiny::setUsbLabel(const QString &l) { QMutexLocker lk(&m_mutex); m_usbLabel = l; emit pathsChanged(); }
void DngBatchConverterTiny::setCfaPattern(const QString &c) { QMutexLocker lk(&m_mutex); m_cfa = c; emit parametersChanged(); }
void DngBatchConverterTiny::setBlackLevel(int v) { QMutexLocker lk(&m_mutex); m_black = v; emit parametersChanged(); }
void DngBatchConverterTiny::setWhiteLevel(int v) { QMutexLocker lk(&m_mutex); m_white = v; emit parametersChanged(); }
void DngBatchConverterTiny::setWhiteBalance(const QString &wb) { QMutexLocker lk(&m_mutex); m_wb = wb; emit parametersChanged(); }
void DngBatchConverterTiny::setDcpPath(const QString &p) { QMutexLocker lk(&m_mutex); m_dcp = p; emit parametersChanged(); }
void DngBatchConverterTiny::setSrcGlob(const QString &g) { QMutexLocker lk(&m_mutex); m_srcGlob = g; emit parametersChanged(); }
void DngBatchConverterTiny::setDestRollPattern(const QString &p) { QMutexLocker lk(&m_mutex); m_destRollPattern = p; emit parametersChanged(); }
void DngBatchConverterTiny::setDestFilePattern(const QString &p) { QMutexLocker lk(&m_mutex); m_destFilePattern = p; emit parametersChanged(); }

void DngBatchConverterTiny::cancel() { m_cancel.storeRelease(1); }


// --- Minimal PGM (P5) header/token parser without QTextStream (Qt6-safe) ----
static bool readNextToken(QFile &f, QByteArray &tok)
{
    tok.clear();
    char ch = 0;
    // Skip whitespace and comments
    while (f.getChar(&ch)) {
        if (ch == '#') { f.readLine(); continue; }
        if (isspace(static_cast<unsigned char>(ch))) { continue; }
        tok.append(ch); break;
    }
    if (tok.isEmpty()) return false;
    // Read until next whitespace
    while (f.getChar(&ch)) {
        if (isspace(static_cast<unsigned char>(ch))) break;
        tok.append(ch);
    }
    return true;
}

bool DngBatchConverterTiny::readPgm16(const QString &path, PgmInfo *info, QByteArray *pixelBytes, QString *err)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly)) { if (err) *err = QStringLiteral("Cannot open %1").arg(path); return false; }

    QByteArray tok;
    if (!readNextToken(f, tok) || tok != "P5") { if (err) *err = QStringLiteral("PGM magic P5 not found in %1").arg(path); return false; }

    if (!readNextToken(f, tok)) { if (err) *err = QStringLiteral("PGM width missing: %1").arg(path); return false; }
    bool ok=false; int w = tok.toInt(&ok); if (!ok) { if (err) *err = QStringLiteral("Bad width in %1").arg(path); return false; }

    if (!readNextToken(f, tok)) { if (err) *err = QStringLiteral("PGM height missing: %1").arg(path); return false; }
    int h = tok.toInt(&ok); if (!ok) { if (err) *err = QStringLiteral("Bad height in %1").arg(path); return false; }

    if (!readNextToken(f, tok)) { if (err) *err = QStringLiteral("PGM maxval missing: %1").arg(path); return false; }
    int maxv = tok.toInt(&ok); if (!ok) { if (err) *err = QStringLiteral("Bad maxval in %1").arg(path); return false; }

    // After maxval, there is a single whitespace before binary data. Consume one char if it's whitespace.
    char sep = 0; if (f.peek(&sep,1)==1 && isspace(static_cast<unsigned char>(sep))) { char dummy; f.getChar(&dummy); }

    info->width = w; info->height = h; info->maxval = maxv; info->bigEndian = true; info->headerBytes = f.pos();

    const qint64 expectedBytes = qint64(w) * qint64(h) * 2;
    QByteArray data = f.read(expectedBytes);
    if (data.size() != expectedBytes) { if (err) *err = QStringLiteral("Unexpected EOF in %1").arg(path); return false; }

    // Convert BE->LE (PGM 16-bit is BE by spec)
    quint16 *p = reinterpret_cast<quint16*>(data.data());
    const int count = w*h;
    for (int i=0;i<count;i++) p[i] = qFromBigEndian(p[i]);

    *pixelBytes = std::move(data);
    return true;
}

QFileInfoList DngBatchConverterTiny::sortedPgmList(const QDir &dir, const QString &glob) {
    QFileInfoList list = dir.entryInfoList(QStringList{glob}, QDir::Files, QDir::Name);
    std::sort(list.begin(), list.end(), [](const QFileInfo &a, const QFileInfo &b){
        const int ia = DngBatchConverterTiny::parseIndexFromName(a.fileName());
        const int ib = DngBatchConverterTiny::parseIndexFromName(b.fileName());
        if (ia >= 0 && ib >= 0) return ia < ib; // numeric compare when both parse
        return a.fileName().compare(b.fileName(), Qt::CaseInsensitive) < 0;
    });
    return list;
}

QString DngBatchConverterTiny::rollFolderName(int roll, const QString &pattern) {
    return QString::asprintf(pattern.toUtf8().constData(), roll);
}

QString DngBatchConverterTiny::destFileName(int roll, int index, const QString &pattern) {
    return QString::asprintf(pattern.toUtf8().constData(), roll, index);
}

int DngBatchConverterTiny::parseIndexFromName(const QString &filename)
{
    // Matches XiCapture000.pgm, XiCapture0123.pgm, etc. Case-insensitive, any number of digits
    static const QRegularExpression re(R"(^xicapture(\d+)\.pgm$)", QRegularExpression::CaseInsensitiveOption);
    const auto m = re.match(filename);
    if (!m.hasMatch()) return -1;
    bool ok=false;
    int v = m.captured(1).toInt(&ok);
    return ok ? v : -1;
}

bool DngBatchConverterTiny::ensureUsbMounted(QString *mountPointOut) const {
    QMutexLocker lk(&m_mutex);
    const QString explicitMount = m_usbPath;
    const QString label = m_usbLabel;
    lk.unlock();

    if (!explicitMount.isEmpty()) {
        QDir d(explicitMount);
        if (d.exists()) { *mountPointOut = explicitMount; return true; }
        return false;
    }

    const QStringList roots = { "/media", "/media/usb", "/mnt", "/run/media", "/run/media/" + qEnvironmentVariable("USER") };
    for (const QString &root : roots) {
        QDir d(root + "/" + label);
        if (d.exists()) { *mountPointOut = d.absolutePath(); return true; }
    }
    return false;
}

// --- TinyDNG write helper -----------------------------------------------------

static int cfaStringToIndex(const QString &cfa) {
    const QString u = cfa.trimmed().toUpper();
    if (u == "RGGB") return 0;
    if (u == "GRBG") return 1;
    if (u == "GBRG") return 2;
    if (u == "BGGR") return 3;
    return 0; // default RGGB
}

bool DngBatchConverterTiny::writeDngTiny(const QString &dstPath,
                                         const PgmInfo &info,
                                         const quint16 *pixelsLE,
                                         const QString &cfa,
                                         int black, int /*white*/,
                                         const QString &/*wb*/,
                                         const QString &/*dcp*/,
                                         QString *err)
{
    using namespace tinydngwriter;

    DNGImage img;
    img.SetBigEndian(false);

    // ---- Core image description (order matters) ----
    img.SetImageWidth(unsigned(info.width));
    img.SetImageLength(unsigned(info.height));

    img.SetSamplesPerPixel(1);
    const unsigned short bps[1] = { 16 };
    img.SetBitsPerSample(1, bps);
    const unsigned short sampleFmt[1] = { SAMPLEFORMAT_UINT };
    img.SetSampleFormat(1, sampleFmt);

    img.SetPlanarConfig(PLANARCONFIG_CONTIG);
    img.SetCompression(COMPRESSION_NONE);
    img.SetPhotometric(PHOTOMETRIC_CFA);

    // Strips
    img.SetRowsPerStrip(unsigned(info.height));

    // ---- CFA tags ----
    img.SetCFARepeatPatternDim(2, 2);
    unsigned char cfa4[4] = {0,1,1,2}; // default RGGB
    const QString U = cfa.trimmed().toUpper();
    if      (U == "RGGB") { unsigned char t[4] = {0,1,1,2}; memcpy(cfa4,t,4); }
    else if (U == "GRBG") { unsigned char t[4] = {1,0,2,1}; memcpy(cfa4,t,4); }
    else if (U == "GBRG") { unsigned char t[4] = {1,2,0,1}; memcpy(cfa4,t,4); }
    else if (U == "BGGR") { unsigned char t[4] = {2,1,1,0}; memcpy(cfa4,t,4); }
    img.SetCFAPattern(4, cfa4);

    // Identity / housekeeping
    img.SetUniqueCameraModel("XIMEA MU9");
    img.SetDNGVersion(1, 4, 0, 0);
    img.SetSoftware("DigitalSuper8v5.3");
    img.SetCalibrationIlluminant1(21);

    // Illuminant + neutral
 //   img.SetCalibrationIlluminant1(21); // D65
 //   double asn[3] = {1.0, 1.0, 1.0};
    double asn[3] = {0.857142857, 1.0, 0.75};//Alternatief is: (0.6567, 1.0, 0.8339)
    img.SetAsShotNeutral(3, asn);

    // ---- Simple Color/Forward matrices (D65) ----
    // sRGB <-> XYZ (D65), row-major
 //   const double M_RGB2XYZ[9] = {
 //       0.4124564, 0.3575761, 0.1804375,
 //       0.2126729, 0.7151522, 0.0721750,
 //       0.0193339, 0.1191920, 0.9503041
 //   };
 //   const double M_XYZ2RGB[9] = {
 //        3.24045484, -1.53713885, -0.49853155,
 //       -0.96926639,  1.87601093,  0.04155608,
 //        0.05564342, -0.20402585,  1.05722516
 //   };

    const double colorM1[9] = {
        3.24045484 * 1.00, -1.53713885 * 0.95, -0.49853155 * 0.90,
        -0.96926639 * 0.95,  1.87601093 * 1.00,  0.04155608 * 0.90,
        0.05564342 * 0.90, -0.20402585 * 0.95,  1.05722516 * 1.00
   };
    const double forwardM1[9] = {
        0.4124564, 0.3575761, 0.1804375,
        0.2126729, 0.7151522, 0.0721750,
        0.0193339, 0.1191920, 0.9503041
    };

    // DNG expects: ColorMatrix1 = XYZ->Camera, ForwardMatrix1 = Camera->XYZ.
    // If we approximate Camera == sRGB:
       // D65
   // img.SetColorMatrix1(3, M_XYZ2RGB);    // XYZ -> sRGB  (pretend camera)
   // img.SetForwardMatrix1(3, M_RGB2XYZ);  // sRGB -> XYZ  (pretend camera)

    img.SetColorMatrix1(3, colorM1);

    img.SetColorMatrix1(3, colorM1);
    img.SetForwardMatrix1(3, forwardM1);

        // Optionally duplicate as *2:
        // img.SetCalibrationIlluminant2(21);
        // img.SetColorMatrix2(9, colorM1);
        // img.SetForwardMatrix2(9, forwardM1);


    // ---- Levels for Resolve ----
    const double bl[1] = { double(black) };
    const double wl[1] = { 4095.0 };
    img.SetBlackLevelRational(1, bl);
    img.SetWhiteLevelRational(1, wl);

    // --- Added for more pleasing image:
    // Brighten midtones for nicer defaults
    img.SetBaselineExposure(1.3);



    // Define crop to full image (Resolve respects it)
 //   img.SetDefaultCropOrigin(0, 0);
 //   img.SetDefaultCropSize(unsigned(info.width), unsigned(info.height));

    // Optional identity calibration
    const double camcal[9] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    img.SetCameraCalibration1(9, camcal);

    // Give it a friendly profile name
  //  img.SetProfileName("DigitalSuper8 Default");


    // ---- Pixel data ----
    const int width  = info.width;
    const int height = info.height;
    const size_t pixels = size_t(width) * size_t(height);

    std::vector<uint16_t> buf(pixels);

    // Copy + clamp to 12-bit (optional but nice since WhiteLevel = 4095)
    for (size_t i = 0; i < pixels; ++i) {
        buf[i] = pixelsLE[i] & 0x0FFF;
    }

    bool rotate180 = true;  // same flag as above

    if (rotate180) {
        // Rotate image 180°: (x,y) -> (w-1-x, h-1-y)
        for (int y = 0; y < height / 2; ++y) {
            uint16_t* topRow    = &buf[y * width];
            uint16_t* bottomRow = &buf[(height - 1 - y) * width];

            for (int x = 0; x < width; ++x) {
                std::swap(topRow[x], bottomRow[width - 1 - x]);
            }
        }

        // If height is odd, we still need to reverse the middle row
        if (height % 2 != 0) {
            int mid = height / 2;
            uint16_t* row = &buf[mid * width];
            for (int x = 0; x < width / 2; ++x) {
                std::swap(row[x], row[width - 1 - x]);
            }
        }
    }

    const size_t bytes = pixels * sizeof(uint16_t);
    img.SetImageData(reinterpret_cast<const unsigned char*>(buf.data()), bytes);



  //  const size_t pixels = size_t(info.width) * size_t(info.height);
  //  const size_t bytes  = pixels * sizeof(uint16_t);
  //  std::vector<uint16_t> buf(pixels);
  //  memcpy(buf.data(), pixelsLE, bytes);
  //  img.SetImageData(reinterpret_cast<const unsigned char*>(buf.data()), bytes);

    // ---- Write ----
    DNGWriter writer(false);
    writer.AddImage(&img);
    std::string error;
    if (!writer.WriteToFile(dstPath.toStdString().c_str(), &error)) {
        if (err) *err = QString::fromStdString(error);
        return false;
    }
    return true;
}





void DngBatchConverterTiny::convertAllRolls() {
    QMutexLocker lk(&m_mutex);
    const QString root = m_workPath;
    lk.unlock();

    QDir rootDir(root);
    if (!rootDir.exists()) {
        emit errorOccured(QStringLiteral("Work path does not exist: %1").arg(root));
        emit finished();
        return;
    }

    const auto entries = rootDir.entryInfoList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name);
    QList<int> rolls;
    QRegularExpression re(R"(^images(\d+)$)");
    for (const QFileInfo &fi : entries) {
        const auto m = re.match(fi.fileName());
        if (m.hasMatch()) rolls << m.captured(1).toInt();
    }
    std::sort(rolls.begin(), rolls.end());

    for (int r : rolls) {
        if (m_cancel.loadAcquire()) break;
        convertImpl(r);
    }

    emit finished();
}

void DngBatchConverterTiny::convertRoll(int roll) { convertImpl(roll); }

void DngBatchConverterTiny::convertImpl(int roll)
{
     m_cancel.storeRelease(0);    // ready for a new run
  //  errorOccured("This is a test");
    if (m_cancel.loadAcquire()) return;

    QString usbMount;
    if (!ensureUsbMounted(&usbMount)) {
        emit errorOccured(QStringLiteral("USB drive not found. Set usbPath or attach volume labeled '%1'.").arg(m_usbLabel));
        return;
    }

    QString srcRoot, destRollPat, destFilePat, glob, cfa, wb, dcp; int black, white;
    {
        QMutexLocker lk(&m_mutex);
        srcRoot = m_workPath; destRollPat = m_destRollPattern; destFilePat = m_destFilePattern; glob = m_srcGlob;
        cfa = m_cfa; black = m_black; white = m_white; wb = m_wb; dcp = m_dcp;
    }

    const QString rollDirName = QStringLiteral("images%1").arg(roll);
    const QDir srcDir(srcRoot + "/" + rollDirName);
    if (!srcDir.exists()) { emit errorOccured(QStringLiteral("Missing source directory: %1").arg(srcDir.absolutePath())); emit rollFinished(roll,0,0,0); return; }

    const QFileInfoList pgms = sortedPgmList(srcDir, glob);
    if (pgms.isEmpty()) { emit rollFinished(roll,0,0,0); return; }

    const QString rollFolder = rollFolderName(roll, destRollPat);
    QDir dstDir(usbMount + "/" + rollFolder);
    if (!dstDir.exists()) { QDir(usbMount).mkpath(rollFolder); dstDir = QDir(usbMount + "/" + rollFolder); }
    if (!dstDir.exists()) { emit errorOccured(QStringLiteral("Cannot create destination: %1").arg(dstDir.absolutePath())); return; }

    int converted=0, skipped=0, failed=0, total=pgms.size();

    for (int i=0;i<pgms.size();++i) {
        if (m_cancel.loadAcquire()) break;
        const QFileInfo &fi = pgms.at(i);
        int index = parseIndexFromName(fi.fileName());
        if (index < 0) index = i; // fallback to sequential if pattern didn't match
        const QString dstName = destFileName(roll, index, destFilePat);
        const QString dstPath = dstDir.absoluteFilePath(dstName);

        emit progress(roll, index, total);
        QByteArray Status;
        Status.append("Progress: ");
        emit StatusUpdateServer(Status, index);
        if (QFile::exists(dstPath)) { ++skipped; continue; }

        PgmInfo pgm; QByteArray px; QString err;
        if (!readPgm16(fi.absoluteFilePath(), &pgm, &px, &err)) { ++failed; emit errorOccured(err); continue; }
        if (pgm.maxval != 65535) {//was 4095
            // Still valid; but warn user so they can adjust white level
            emit errorOccured(QStringLiteral("Warning: %1 has maxval=%2 (expected 4095). Using whiteLevel=%3")
                              .arg(fi.fileName()).arg(pgm.maxval).arg(white));
        }

        // px currently contains little-endian 16-bit samples
        const quint16 *pixelsLE = reinterpret_cast<const quint16*>(px.constData());
        QString werr;
        if (!writeDngTiny(dstPath, pgm, pixelsLE, cfa, black, white, wb, dcp, &werr)) {
            ++failed; emit errorOccured(QStringLiteral("Write failed for %1 -> %2: %3").arg(fi.fileName(), dstName, werr)); continue;
        }

        ++converted;
        emit fileConverted(roll, index, fi.absoluteFilePath(), dstPath);
    }

    emit rollFinished(roll, converted, skipped, failed);
    QByteArray Status;
    Status = "Role " + QByteArray::number(roll) + " done: " + QByteArray::number(converted) + " OK, " + QByteArray::number(failed) + " failed, " + QByteArray::number(skipped) + " skipped.";
    emit StatusUpdateServer(Status, converted);
}
