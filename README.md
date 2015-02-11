# Sudoku-Recognition

From the webcam stream, detect a sudoku and recognize the digits. Unfortunately, sometimes, there are one or two false detection.
The detection of the sudoku border is not perfect. You have to reach 20 detections of the sudoku border (in red), and it should take ~50% of the screen.

# Lib

- OpenCV 2.4.9

# Compilation 

```g++ -std=c++11 `pkg-config --cflags opencv` main.cpp ImageProcessing/*/*.cpp Training/*/*.cpp `pkg-config --libs opencv` -o output```
