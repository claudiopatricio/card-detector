// Copyright (c) 2018 Cláudio Patrício
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace std;

bool is_file_exist(const char *fileName)
{
    ifstream infile(fileName);
    return infile.good();
}

int main( void )
{
    VideoCapture webcam(0);
    //cout << webcam.get(CV_CAP_PROP_POS_FRAMES) << endl;

    if(!webcam.isOpened())
    {
        cout << "Cam not live!!!" << endl;
        return 0;
    }
    int debug = 0;

    Mat cards[53];
    Mat cannys[52];
    Mat camFrame;
    Mat camFrameGray;
    Mat camFrameCanny;
    char buffer[100];
    cout << "Loading cards" << endl;
    for(int n = 1; n <= 52; n++)
    {
        sprintf(buffer, "canny/canny%d.jpg", n);
        cannys[n-1] = imread(buffer, CV_LOAD_IMAGE_UNCHANGED);
        sprintf(buffer, "cards/card%d.png", n);
        cards[n-1] = imread(buffer, CV_LOAD_IMAGE_UNCHANGED);
    }
    cards[52] = imread("cards/card-default.png", CV_LOAD_IMAGE_UNCHANGED);

    cout << "Loading cards done" << endl;
    int found = 52;
    namedWindow("Card Recognition", CV_WINDOW_AUTOSIZE);
    namedWindow("Webcam", CV_WINDOW_AUTOSIZE);
    imshow("Card Recognition", cards[found]);
    while(true)
    {
        webcam.read(camFrame);

        cvtColor(camFrame, camFrameGray, CV_BGR2GRAY);
        blur(camFrameGray, camFrameGray, Size(3,3));
        Canny(camFrameGray, camFrameCanny, 40, 120, 3, false);
        imshow("Webcam", camFrame);

        int pressed = waitKey(30);
        if(((char)pressed == 'Q') || ((char)pressed == 'q')) break;

        Mat result;
        //result.create(imagemCanny.rows - card.rows + 1, imagemCanny.cols - card.cols + 1, CV_32FC1);
        /*
        0: SQDIFF
        1: SQDIFF NORMED
        2: TM CCORR
        3: TM CCORR NORMED
        4: TM COEFF
        5: TM COEFF NORMED
        */
        int match_method = CV_TM_CCOEFF_NORMED;

        for(int i = 26; i < 39; i++)
        {
            Mat res_32f(camFrameCanny.rows - cannys[i].rows + 1, camFrameCanny.cols - cannys[i].cols + 1, CV_32FC2);
            matchTemplate(camFrameCanny, cannys[i], res_32f, match_method); // TM_SQDIFF_NORMED, TM_CCORR

            // Localizing the best match with minMaxLoc
            double minVal; double maxVal; Point minLoc; Point maxLoc;
            Point matchLoc;

            res_32f.convertTo(result, CV_8U, 255.0);
            adaptiveThreshold(result, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 1211, -150);

            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

            // For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
            if(match_method  == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) { matchLoc = minLoc; }
            else { matchLoc = maxLoc; }

            if(debug)
            {
                cout << "x: " << matchLoc.x << '|' << minLoc.x << '|' << maxLoc.x << '|' << minVal << endl;
                cout << "y: " << matchLoc.y << '|' << minLoc.y << '|' << maxLoc.y << '|' << maxVal << endl;
            }

            if(matchLoc.x > 0)
            {
                imshow("Card Recognition", cards[i]);
                break;
            }
        }
    }
    webcam.release();
    destroyAllWindows();
    return 0;
}
