//
//  main.cpp
//  hw4
//
//  Created by Gayatri Prabhu on 4/2/17.
//  Copyright © 2017 Gayatri Prabhu. All rights reserved.
//

#include <iostream>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>

using namespace std;
using namespace cv;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
    
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    
    string line, path, classlabel;
    
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, ',');
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            Mat im=imread(path,0);
            resize(im, im, Size(277, 388));
            images.push_back(im);
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

void detectAndDisplay(Mat frame,int im_width, int im_height);

string face_cascade_name = "/Users/gayatriprabhu/Downloads/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_alt.xml";

CascadeClassifier face_cascade;

Ptr<cv::face::FaceRecognizer> model;

int main(int argc, const char * argv[]) {
    
    vector<Mat> images;
    vector<int> labels;
    read_csv("/Users/gayatriprabhu/Desktop/dataset.csv",  images, labels);
    model = cv::face::createFisherFaceRecognizer();
    model->train(images,labels);
    
    int im_width = images[0].cols;
    int im_height = images[0].rows;

    
    VideoCapture cap(0);
    
    while (true) {
        
        
        Mat capt;
        
        cap.read(capt);
        
        if (!face_cascade.load(face_cascade_name)){
            printf("--(!)Error loading\n");
            return (-1);
        }
        
        if (!capt.empty()){
            detectAndDisplay(capt, im_width, im_height);
        }
        else{
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        
        namedWindow("video", WINDOW_NORMAL);
        
        imshow("video", capt);
        
        waitKey(1);
        
    }
}

    void detectAndDisplay(Mat frame, int im_width, int im_height)
    {
        
        
        Mat frame_gray;
        std::vector<Rect> faces;
        
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
        
        equalizeHist(frame_gray, frame_gray);
        
        // Detect faces
        face_cascade.detectMultiScale(frame_gray, faces, 1.1, 3, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        
        size_t ic = 0; // ic is index of current element
        
        for (ic = 0; ic < faces.size(); ic++) // Iterate through all current elements (detected faces)
            
        {
            
            Mat face;
            cvtColor(frame(faces[ic]), face, COLOR_BGR2GRAY);
            
            Mat face_resized;
            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
            
            int prediction = model->predict(face_resized);
            
            Point pt1(faces[ic].x, faces[ic].y);
            Point pt2((faces[ic].x + faces[ic].height), (faces[ic].y + faces[ic].width));
            rectangle(frame, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
            
            if(prediction==0)
            GaussianBlur(frame(faces[ic]), frame(faces[ic]), Size(0, 0), 20);
        }
}

