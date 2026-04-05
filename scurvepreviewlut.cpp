#include "scurvepreviewlut.h"
#include <QPainter>
#include <QPainterPath>
#include <algorithm>


SCurvePreviewLUT::SCurvePreviewLUT(QWidget* parent) : QWidget(parent) {
    setMinimumHeight(180);
    setMinimumWidth(300);
}


void SCurvePreviewLUT::setLUTs(const QVector<quint16>& r,
                               const QVector<quint16>& g,
                               const QVector<quint16>& b)
{
    r_ = r;
    g_ = g;
    b_ = b;

    // Compute a sensible max from the data (12-bit or 16-bit; it doesn’t really matter for drawing)
    max_ = 1;
    auto updateMax = [this](const QVector<quint16>& v){
        for (quint16 val : v)
            if (val > max_)
                max_ = val;
    };
    updateMax(r_);
    updateMax(g_);
    updateMax(b_);
    if (max_ == 0)
        max_ = 1;

    update();
}

void SCurvePreviewLUT::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing, true);

    // Use the widget's base color as background
    const QColor bg = palette().color(QPalette::Base);
    p.fillRect(rect(), bg);

    const QRectF box = rect().adjusted(12, 12, -12, -12);

    // Detect whether we're on a dark or light theme
    const bool isDark = bg.value() < 128; // value = brightness in HSV (0–255)

    // --- Grid ---
    QColor gridColor;
    if (isDark) {
        // Light grid on dark background
        gridColor = QColor(255, 255, 255, 60);   // soft white-ish
    } else {
        // Dark grid on light background
        gridColor = QColor(0, 0, 0, 50);
    }

    p.setPen(QPen(gridColor, 1));
    for (int i = 1; i < 4; ++i) {
        const double x = box.left() + i * box.width() / 4.0;
        const double y = box.top()  + i * box.height() / 4.0;
        p.drawLine(QPointF(x, box.top()),    QPointF(x, box.bottom()));
        p.drawLine(QPointF(box.left(), y),   QPointF(box.right(), y));
    }

    // --- Linear reference (diagonal) ---
    QColor diagColor;
    if (isDark) {
        diagColor = QColor(255, 255, 255, 100);  // brighter on dark
    } else {
        diagColor = QColor(0, 0, 0, 90);         // your original
    }

    p.setPen(QPen(diagColor, 1, Qt::DashLine));
    p.drawLine(box.topLeft(), box.bottomRight());

    // --- S-curves (unchanged) ---
    if (!r_.isEmpty()) drawCurve(p, r_, QColor(220, 60, 60),   box, max_);
    if (!g_.isEmpty()) drawCurve(p, g_, QColor(60, 170, 60),   box, max_);
    if (!b_.isEmpty()) drawCurve(p, b_, QColor(60, 100, 220),  box, max_);
}


void SCurvePreviewLUT::drawCurve(QPainter& p,
                                 const QVector<quint16>& lut,
                                 const QColor& color,
                                 const QRectF& box,
                                 int maxValue) {
    if (lut.size() < 2) return;


    QPainterPath path;
    const int N = lut.size();
    for (int i=0; i<N; ++i) {
        const double x = (N==1) ? 0.0 : double(i) / double(N-1);
        const double y = std::clamp(double(lut[i]) / double(maxValue), 0.0, 1.0);
        const QPointF w(box.left() + x * box.width(),
                        box.bottom() - y * box.height());
        if (i==0) path.moveTo(w); else path.lineTo(w);
    }
    p.setPen(QPen(color, 2));
    p.drawPath(path);
}
