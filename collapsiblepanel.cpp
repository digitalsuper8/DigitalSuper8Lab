#include "CollapsiblePanel.h"

#include <QToolButton>
#include <QVBoxLayout>
#include <QGridLayout>
#include <QBoxLayout>
#include <QStyle>
#include <QEvent>
#include <QMetaObject>

CollapsiblePanel::CollapsiblePanel(QWidget *parent)
    : QWidget(parent)
{
    // Run after setupUi() (queued), but before first paint in most cases
    QMetaObject::invokeMethod(this, "ensureInitialized", Qt::QueuedConnection);
}

CollapsiblePanel::CollapsiblePanel(const QString &title, QWidget *parent)
    : QWidget(parent), m_title(title)
{
    QMetaObject::invokeMethod(this, "ensureInitialized", Qt::QueuedConnection);
}

QString CollapsiblePanel::title() const
{
    return m_title;
}

void CollapsiblePanel::setTitle(const QString &title)
{
    m_title = title;
    if (m_headerButton) {
        m_headerButton->setText(m_title);
    }
}

bool CollapsiblePanel::event(QEvent *e)
{
    // If the widget gets polished/parented later, ensure we init.
    if (!m_initialized && (e->type() == QEvent::Polish || e->type() == QEvent::Show)) {
        QMetaObject::invokeMethod(this, "ensureInitialized", Qt::QueuedConnection);
    }
    return QWidget::event(e);
}

QWidget* CollapsiblePanel::findContentWidget() const
{
    // Prefer a direct child with a "content-ish" name.
    const auto kids = findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly);
    for (QWidget *w : kids) {
        if (!w) continue;
        if (qobject_cast<QToolButton*>(w)) continue;
        const QString n = w->objectName().toLower();
        if (n.contains("content") || n.contains("body") || n.contains("container")) {
            return w;
        }
    }
    // Otherwise first non-button direct child
    for (QWidget *w : kids) {
        if (!w) continue;
        if (qobject_cast<QToolButton*>(w)) continue;
        return w;
    }
    return nullptr;
}

void CollapsiblePanel::ensureInitialized()
{
    if (m_initialized) return;
    m_initialized = true;

    // Grab content created by .ui (e.g. devEffectsContent)
    m_contentWidget = findContentWidget();
    if (!m_contentWidget) {
        m_contentWidget = new QWidget(this);
    }

    // Create header button
    m_headerButton = new QToolButton(this);
    m_headerButton->setToolButtonStyle(Qt::ToolButtonTextBesideIcon);
    m_headerButton->setAutoRaise(true);
    m_headerButton->setText(m_title.isEmpty() ? objectName() : m_title);
    m_headerButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

    applyHeaderVisual();

    connect(m_headerButton, &QToolButton::clicked, this, &CollapsiblePanel::toggleCollapsed);

    // If Designer already assigned a layout to this widget, reuse it by inserting header at top.
    if (QLayout *existing = layout()) {
        if (auto *box = qobject_cast<QBoxLayout*>(existing)) {
            box->insertWidget(0, m_headerButton);
        } else if (auto *grid = qobject_cast<QGridLayout*>(existing)) {
            // Put header on row 0 spanning all columns
            int cols = qMax(1, grid->columnCount());
            grid->addWidget(m_headerButton, 0, 0, 1, cols);
        } else {
            // Fallback: wrap existing layout into a new vbox (best-effort)
            auto *wrap = new QVBoxLayout();
            wrap->setContentsMargins(0,0,0,0);
            wrap->setSpacing(0);
            // Move old layout items into a container widget
            QWidget *container = new QWidget(this);
            container->setLayout(existing);
            wrap->addWidget(m_headerButton);
            wrap->addWidget(container);
            setLayout(wrap);
        }
    } else {
        // Normal path: create our own layout (header + existing content widget)
        auto *main = new QVBoxLayout(this);
        main->setContentsMargins(0, 0, 0, 0);
        main->setSpacing(0);
        main->addWidget(m_headerButton);
        main->addWidget(m_contentWidget);
        setLayout(main);
    }

    // Start expanded by default
    m_collapsed = false;
    setCollapsed(m_collapsed);
}

void CollapsiblePanel::applyHeaderVisual()
{
    // Use explicit icons so it always shows even with custom QSS
    const QIcon down  = style()->standardIcon(QStyle::SP_ArrowDown);
    const QIcon right = style()->standardIcon(QStyle::SP_ArrowRight);
    m_headerButton->setIcon(m_collapsed ? right : down);
}

void CollapsiblePanel::toggleCollapsed()
{
    setCollapsed(!m_collapsed);
}

void CollapsiblePanel::setCollapsed(bool collapsed)
{
    m_collapsed = collapsed;

    if (m_contentWidget) {
        m_contentWidget->setVisible(!m_collapsed);
    }
    if (m_headerButton) {
        applyHeaderVisual();
    }
}
