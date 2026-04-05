# Digital Super8 Lab

Digital Super8 Lab is the Windows desktop processing and development application for the Digital Super8 project.

It is used to:

- preview and develop captured Super 8 image sequences
- apply grading and film-look adjustments
- render developed video files
- export CinemaDNG
- experiment with Super 8 specific development effects

## Main technologies

- Qt 6
- C++
- OpenCV

## Project status

This repository contains the current working Windows version of DS8 Lab.

## Notes

- Build output folders are not tracked in Git.
- Large generated files such as rendered videos, WAV files, PCM files, and image sequence work folders are excluded from Git.
- OpenCV runtime/build artifacts are not intended to live in the repository.

## Typical structure

- Source code: C++ / Qt UI files
- Build directory: outside the source tree
- Work/output directories: outside the source tree where possible

## Current development

Active development currently includes:

- audio WAV generation from per-frame PCM files
- stretching WAV output to exact rendered video duration
- optional muxing of WAV audio into rendered video
