#pragma once
#include <QMetaType>

struct Super8DevParams
{
    bool enabled = false;

    float scratches = 0.10f;
    float dust      = 0.05f;
    float lightLeak = 0.10f;
    float weave     = 0.03f;
    float grain     = 0.15f;

    int   fadeInWarmthFrames = 12;
    float fadeOutYellowShift = 0.70f;

    // Planned scene leaks (0 means unused). Max 5 events.
    int   leakFrames[5] = {0,0,0,0,0};

    // Random leaks (your existing burst behavior)
    bool  applyRandomLeaks = true;
};

//Q_DECLARE_METATYPE(Super8DevParams)
