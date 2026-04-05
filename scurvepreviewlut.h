#pragma once
#include <QWidget>
#include <QVector>


class SCurvePreviewLUT : public QWidget {
    Q_OBJECT
public:
    explicit SCurvePreviewLUT(QWidget* parent=nullptr);


public slots:
    void setLUTs(const QVector<quint16>& r,
                 const QVector<quint16>& g,
                 const QVector<quint16>& b);


protected:
    void paintEvent(QPaintEvent*) override;


private:
    void drawCurve(QPainter& p, const QVector<quint16>& lut, const QColor& color,
                   const QRectF& box, int maxValue);


    QVector<quint16> r_, g_, b_;
    int max_ = 4095;
};
