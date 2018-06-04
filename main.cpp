#include "opencv2/opencv.hpp"
#include <iostream>
#include "sign.hpp"

#include </home/mateusz/GIT/dlib/dlib/svm_threaded.h>
#include </home/mateusz/GIT/dlib/dlib/gui_widgets.h>
#include </home/mateusz/GIT/dlib/dlib/image_processing.h>
#include </home/mateusz/GIT/dlib/dlib/data_io.h>
#include </home/mateusz/GIT/dlib/dlib/image_transforms.h>
#include </home/mateusz/GIT/dlib/dlib/cmd_line_parser.h>

using namespace std;
using namespace cv;


Sign z1;

struct TrafficSign {
  string name;
  string svm_path;
  dlib::rgb_pixel color;
  TrafficSign(string name, string svm_path, dlib::rgb_pixel color) :
    name(name), svm_path(svm_path), color(color) {};
};


int main() {



    //wczytaj
    z1.znak = imread("/home/mateusz/OCV_Project/Sign_Recognition/na3/4.jpg");
    namedWindow("pierwszenstwo",WINDOW_AUTOSIZE);

   // cvtColor(z1.znak, z1.znak, COLOR_BGR2RGB);

    Mat gray;
    cvtColor(z1.znak, gray, CV_BGR2GRAY);

    threshold(gray, gray, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    imshow("gray",gray);

    Mat dist;
    distanceTransform(gray, dist, CV_DIST_L2, 5); //distance transform
    normalize(dist, dist, 0, 1.75, NORM_MINMAX);
    threshold(dist, dist, .2, 2., CV_THRESH_BINARY);
    //imshow("Distance Transform Image", dist);

          // morphology
    Mat kernel1 = Mat::ones(8, 8, CV_8UC1);
    dilate(dist, dist, kernel1);//dylatacja
    //imshow("Peaks", dist); //wykrycie wierzcholkow

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
    //circle(markers, Point(5,5), 3, CV_RGB(255,255,255), -1);
    //imshow("Markers", markers*10000);

    // Perform the watershed algorithm
    watershed(z1.znak, markers);

    Mat mark = Mat::zeros(markers.size(), CV_8UC1);
    markers.convertTo(mark, CV_8UC1);
    bitwise_not(mark, mark);
             //imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
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
     Mat src_gray;
     cvtColor( dst, src_gray, CV_BGR2GRAY );

    Mat canny_output;
    vector<Vec4i> hierarchy;
    vector<vector<Point>> kontury;
     Canny( src_gray, canny_output,10, 255, 3 );
     //imshow("canny",canny_output);
      /// Find contours
      findContours( canny_output, kontury, hierarchy,  CV_RETR_CCOMP  , CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        cout<<kontury.size()<<endl;
      Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
        for( int i = 0; i< kontury.size(); i++ )
           {
             Scalar color = Scalar(8, 35,255);
             drawContours( drawing, kontury, i, color, 2, 8,5, 0, Point() );
           }
      imshow("kont",drawing);

     cv::Size s = canny_output.size();
     cout<<s.area()<<endl;


     vector<Rect> boundRect( kontury.size() );
     Mat crop;
     z1.znak.copyTo(crop);
     for(int i=0;i<kontury.size();i++){
         //cout<<contourArea(kontury[i])<<endl;
     boundRect[i] = boundingRect( Mat(kontury[i]) );

     if((contourArea(kontury[i],true)< s.area()/2) && (contourArea(kontury[i],true)>20)){//&& contourArea(kontury[i])>5)

     rectangle( z1.znak, boundRect[i].tl(), boundRect[i].br(), CV_RGB(200,200,0), 2, 8, 0 );
     crop = crop(boundRect[i]);
     }
              // Visualize the final image
    }

     Mat kola;
     crop.copyTo(kola);



     cvtColor( z1.znak, z1.znak, CV_BGR2GRAY );

     /// Reduce the noise so we avoid false circle detection
     GaussianBlur( z1.znak, z1.znak, Size(9, 9), 2, 2 );


   vector<Vec3f> circles;

     /// Apply the Hough Transform to find the cir1cles
     HoughCircles( z1.znak, circles, CV_HOUGH_GRADIENT, 1, 30, 200, 100, 0, 0 );
          /// Draw the circles detected
     for( size_t i = 0; i < circles.size(); i++ )
     {
         Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
         int radius = cvRound(circles[i][2]);
         // circle center
         circle( z1.znak, center, 3, Scalar(0,255,0), -1, 8, 0 );
         // circle outline
         circle( z1.znak, center, radius, Scalar(0,0,255), 3, 8, 0 );
      }
     imshow("wykrywanie kol",z1.znak);



              imshow("Final Result", dst);
              cvtColor(crop,crop,CV_RGB2HSV);
              Mat red;
              inRange(crop, cv::Scalar(0, 100, 100), cv::Scalar(179, 255, 255), red);//red
              imshow("ROI",crop);
              imshow("RED",red);




    while(1)
    {
    imshow("pierwszenstwo",z1.znak);
    char c = waitKey(15);
    if(c == 's')
        break;
    }

    return 0;
}

