#ifndef COLLAPSIBLEPANEL_H
#define COLLAPSIBLEPANEL_H

#include <QWidget>

class QToolButton;

class CollapsiblePanel : public QWidget
{
    Q_OBJECT
    Q_PROPERTY(QString title READ title WRITE setTitle)
    Q_PROPERTY(bool collapsed READ isCollapsed WRITE setCollapsed)

public:
    explicit CollapsiblePanel(QWidget *parent = nullptr);
    explicit CollapsiblePanel(const QString &title, QWidget *parent = nullptr);

    QString title() const;
    void setTitle(const QString &title);

    bool isCollapsed() const { return m_collapsed; }

public slots:
    void setCollapsed(bool collapsed);

protected:
    bool event(QEvent *e) override;

private slots:
    void ensureInitialized();
    void toggleCollapsed();

private:
    QWidget* findContentWidget() const;
    void applyHeaderVisual();

    QToolButton *m_headerButton = nullptr;
    QWidget     *m_contentWidget = nullptr;
    QString      m_title;
    bool         m_collapsed = false;
    bool         m_initialized = false;
};

#endif // COLLAPSIBLEPANEL_H
