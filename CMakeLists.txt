cmake_minimum_required(VERSION 2.8)
project( SignsRecognition )
find_package( OpenCV REQUIRED )

file(GLOB SignsRecognition_SRC
    "*.h"
    "*.cpp"
    "*.hpp"
)

add_executable(SignsRecognition ${SignsRecognition_SRC})
target_link_libraries( SignsRecognition ${OpenCV_LIBS} )
