#include "opencv2/opencv.hpp"
#include <iostream>
#include "sign.hpp"

using namespace std;
using namespace cv;

Sign z1;



int main() {



    //wczytaj
    z1.znak = imread("/home/mateusz/OCV_Project/Sign_Recognition/na3/5.jpg");
    namedWindow("pierwszenstwo",WINDOW_AUTOSIZE);

   // cvtColor(z1.znak, z1.znak, COLOR_BGR2RGB);

    //cvtColor(z1.znak, z1.znak, COLOR_RGB2HLS);
   // inRange(z1.znak, cv::Scalar(0, 0, 20), cv::Scalar(100, 250, 255), z1.znak);

    Mat gray;
    cvtColor(z1.znak, gray, CV_BGR2GRAY);
    imshow("gray",gray);
    threshold(gray, gray, 10, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);

         Mat dist;
        distanceTransform(gray, dist, CV_DIST_L2, 5);
        // Normalize the distance image for range = {0.0, 1.0}

        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1., NORM_MINMAX);
        imshow("Distance Transform Image", dist);

        threshold(dist, dist, .2, 2., CV_THRESH_BINARY);
           // Dilate a bit the dist image
           Mat kernel1 = Mat::ones(3, 3, CV_8UC1);
           dilate(dist, dist, kernel1);//dylatacja
           imshow("Peaks", dist); //wykrycie wierzcholkow

           Mat dist_8u;
              dist.convertTo(dist_8u, CV_8U);
              // Find total markers
              vector<vector<Point> > contours;
              findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

              // Create the marker image for the watershed algorithm
              Mat markers = Mat::zeros(dist.size(), CV_32SC1);
              // Draw the foreground markers
              for (size_t i = 0; i < contours.size(); i++)
                  drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i)+1), -1);
              // Draw the background marker
              circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
              imshow("Markers", markers*10000);

              // Perform the watershed algorithm
              watershed(z1.znak, markers);
              Mat mark = Mat::zeros(markers.size(), CV_8UC1);
              markers.convertTo(mark, CV_8UC1);
              bitwise_not(mark, mark);
            //  imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
                                            // image looks like at that point
              // Generate random colors
              vector<Vec3b> colors;
              for (size_t i = 0; i < contours.size(); i++)
              {
                  int b = theRNG().uniform(0, 255);
                  int g = theRNG().uniform(0, 255);
                  int r = theRNG().uniform(0, 255);
                  colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
              }
              // Create the result image
              Mat dst = Mat::zeros(markers.size(), CV_8UC3);
              // Fill labeled objects with random colors
              for (int i = 0; i < markers.rows; i++)
              {
                  for (int j = 0; j < markers.cols; j++)
                  {
                      int index = markers.at<int>(i,j);
                      if (index > 0 && index <= static_cast<int>(contours.size()))
                          dst.at<Vec3b>(i,j) = colors[index-1];
                      else
                          dst.at<Vec3b>(i,j) = Vec3b(0,0,0);
                  }
              }
              // Visualize the final image
              imshow("Final Result", dst);

    while(1){





    //inRange(z1.znak, cv::Scalar(159, 135, 135), cv::Scalar(179, 255, 255), z1.znak);


    imshow("pierwszenstwo",gray);


    char c = waitKey(15);
    if(c == 's')
        break;
    }


    return 0;
}

