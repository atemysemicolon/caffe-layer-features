//
// Created by prassanna on 20/10/16.
//
#include <iostream>
#include "classifier.h"

using namespace std;
int main()
{

    int layer_nr = 2;
    std::string model_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/deploy_small.prototxt";
    std::string trained_file="/home/prassanna/Libraries/caffe-master/models/bvlc_alexnet/bvlc_alexnet.caffemodel";
    std::string mean_file = "";
    std::string label_file="/home/prassanna/Libraries/caffe-master/data/ilsvrc12/synset_words.txt";

    Mat img = cv::imread("/home/prassanna/Development/DataTest/Lenna.png");

    //std::vector<cv::Mat> mats;
    CaffeModel classifier(model_file, trained_file, mean_file, label_file);
    std::vector<cv::Mat> hc = classifier.forwardPass(img,1);



    std::cout<<"Bla"<<std::endl;

    return 0;
}