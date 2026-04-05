cmake_minimum_required(VERSION 3.21)
project(DigitalSuper8Lab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# ---- Qt ----
find_package(Qt6 REQUIRED COMPONENTS Widgets)

# ---- OpenCV ----
# Tip: you can also pass this from Qt Creator's CMake options instead of hardcoding
set(OpenCV_DIR "C:/opencv/build/x64/vc16/lib")
find_package(OpenCV REQUIRED CONFIG)

# ---- Sources ----
set(SOURCES
    main.cpp
  #  MainWindow.cpp
  #  MainWindow.h
  #  MainWindow.ui
    processing.cpp
    processing.h
    processing.ui
    settingsdialog.cpp
    settingsdialog.h
    settingsdialog.ui
    developdialog.cpp
    developdialog.h
    developdialog.ui
    developthread.cpp
    developthread.h
    dngbatchconverter_tiny.cpp
    dngbatchconverter_tiny.h
    themes.qrc
)

add_executable(${PROJECT_NAME} ${SOURCES}
  scurvepreviewlut.h scurvepreviewlut.cpp
  dngbatchdialog.h dngbatchdialog.cpp dngbatchdialog.ui
  icons.qrc
  collapsiblepanel.h collapsiblepanel.cpp
  super8devparams.h)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        Qt6::Widgets
        ${OpenCV_LIBS}
)

target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# ---- Convenience: copy OpenCV DLLs after build (so Run works immediately) ----
if (WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${OpenCV_DIR}/../bin" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying OpenCV DLLs next to the executable..."
    )
endif()


if (WIN32)
    target_sources(${PROJECT_NAME} PRIVATE appicon.rc)
endif()

message(STATUS "Using OpenCV version: ${OpenCV_VERSION}")
