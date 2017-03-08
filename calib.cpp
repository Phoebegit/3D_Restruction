#include <string>
#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;
/*
vector<vector<Point2f> >corners_l_array, corners_r_array;
int array_index = 0;
bool ChessboardStable(vector<Point2f>corners_l, vector<Point2f>corners_r){
  if (corners_l_array.size() < 10){
    corners_l_array.push_back(corners_l);
    corners_r_array.push_back(corners_r);
    return false;
  }
  else{
    corners_l_array[array_index % 10] = corners_l;
    corners_r_array[array_index % 10] = corners_r;
    array_index++;
    double error = 0.0;
    for (int i = 0; i < corners_l_array.size(); i++){
      for (int j = 0; j < corners_l_array[i].size(); j++){
        error += abs(corners_l[j].x - corners_l_array[i][j].x) + abs(corners_l[j].y - corners_l_array[i][j].y);
        error += abs(corners_r[j].x - corners_r_array[i][j].x) + abs(corners_r[j].y - corners_r_array[i][j].y);
      }
    }
    if (error < 1000)
    {
      corners_l_array.clear();
      corners_r_array.clear();
      array_index = 0;
      return true;
    }
    else
      return false;
  }
}
*/

int main(){
  cv::VideoCapture camera_l(1);
  cv::VideoCapture camera_r(0);

  camera_l.set(CAP_PROP_FRAME_WIDTH, 320);
  camera_l.set(CAP_PROP_FRAME_HEIGHT, 240);
  camera_r.set(CAP_PROP_FRAME_WIDTH, 320);
  camera_r.set(CAP_PROP_FRAME_HEIGHT, 240);

  if (!camera_l.isOpened()){ cout << "No left camera!" << endl; return -1; }
  if (!camera_r.isOpened()){ cout << "No right camera!" << endl; return -1; }

  cv::Mat frame_l, frame_r;

  Size boardSize(9, 5);
  const float squareSize = 26.1f;  // Set this to your actual square size

  vector<Mat> goodFrame_l;
  vector<Mat> goodFrame_r;

  vector<vector<Point2f> > imagePoints_l;
  vector<vector<Point2f> > imagePoints_r;

  vector<vector<Point3f> > objectPoints;

  int nimages = 0;

  while (1)
  {
    camera_l >> frame_l;
    camera_r >> frame_r;

    bool found_l = false, found_r = false;

    vector<Point2f>corners_l, corners_r;
    found_l = findChessboardCorners(frame_l, boardSize, corners_l,
      CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    found_r = findChessboardCorners(frame_r, boardSize, corners_r,
      CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);

    if (found_l && found_r)
    {
      goodFrame_l.push_back(frame_l);
      goodFrame_r.push_back(frame_r);

      Mat viewGray;
      cvtColor(frame_l, viewGray, COLOR_BGR2GRAY);
      cornerSubPix(viewGray, corners_l, Size(11, 11),
        Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));
      cvtColor(frame_r, viewGray, COLOR_BGR2GRAY);
      cornerSubPix(viewGray, corners_r, Size(11, 11),
        Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.1));

      imagePoints_l.push_back(corners_l);
      imagePoints_r.push_back(corners_r);
      ++nimages;
      frame_l += 100;
      frame_r += 100;

      drawChessboardCorners(frame_l, boardSize, corners_l, found_l);
      drawChessboardCorners(frame_r, boardSize, corners_r, found_r);

//      putText(frame_l, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));
//      putText(frame_r, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));
      imshow("Left Camera", frame_l);
      imshow("Right Camera", frame_r);
      char c = (char)waitKey(500);
      if (c == 27 || c == 'q' || c == 'Q') //Allow ESC to quit
        exit(-1);

      if (nimages >= 30)
        break;
    }
    else{
//      drawChessboardCorners(frame_l, boardSize, corners_l, found_l);
//      drawChessboardCorners(frame_r, boardSize, corners_r, found_r);

//      putText(frame_l, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));
//      putText(frame_r, to_string(nimages), Point(20, 20), 1, 1, Scalar(0, 0, 255));
      imshow("Left Camera", frame_l);
      imshow("Right Camera", frame_r);

      char key = waitKey(1);
      if (key == 27)
        break;
    }
  }
  if (nimages < 20){ cout << "Not enough" << endl; return -1; }

  vector<vector<Point2f> > imagePoints[2] = { imagePoints_l, imagePoints_r };

  objectPoints.resize(nimages);

  for (int i = 0; i < nimages; i++)
  {
    for (int j = 0; j < boardSize.height; j++)
    for (int k = 0; k < boardSize.width; k++)
      objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
  }

  cout << "Running stereo calibration ..." << endl;

  Size imageSize(320, 240);

  Mat cameraMatrix[2], distCoeffs[2];
  cameraMatrix[0] = initCameraMatrix2D(objectPoints, imagePoints_l, imageSize, 0);
  cameraMatrix[1] = initCameraMatrix2D(objectPoints, imagePoints_r, imageSize, 0);
  Mat R, T, E, F;

  double rms = stereoCalibrate(objectPoints, imagePoints_l, imagePoints_r,
    cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, E, F,
    CALIB_FIX_ASPECT_RATIO +
    CALIB_ZERO_TANGENT_DIST +
    CALIB_USE_INTRINSIC_GUESS +
    CALIB_SAME_FOCAL_LENGTH +
    CALIB_RATIONAL_MODEL +
    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
    TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5));
  cout << "done with RMS error=" << rms << endl;

  double err = 0;
  int npoints = 0;
  vector<Vec3f> lines[2];
  for (int i = 0; i < nimages; i++)
  {
    int npt = (int)imagePoints_l[i].size();
    Mat imgpt[2];
    imgpt[0] = Mat(imagePoints_l[i]);
    undistortPoints(imgpt[0], imgpt[0], cameraMatrix[0], distCoeffs[0], Mat(), cameraMatrix[0]);
    computeCorrespondEpilines(imgpt[0], 0 + 1, F, lines[0]);

    imgpt[1] = Mat(imagePoints_r[i]);
    undistortPoints(imgpt[1], imgpt[1], cameraMatrix[1], distCoeffs[1], Mat(), cameraMatrix[1]);
    computeCorrespondEpilines(imgpt[1], 1 + 1, F, lines[1]);

    for (int j = 0; j < npt; j++)
    {
      double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
        imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
        fabs(imagePoints[1][i][j].x*lines[0][j][0] +
        imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
      err += errij;
    }
    npoints += npt;
  }
  cout << "average epipolar err = " << err / npoints << endl;

  FileStorage fs("intrinsics.yml", FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
      "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
    fs.release();
  }
  else
    cout << "Error: can not save the intrinsic parameters\n";


  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];

  stereoRectify(cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

  fs.open("extrinsics.yml", FileStorage::WRITE);
  if (fs.isOpened())
  {
    fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
    fs.release();
  }
  else
    cout << "Error: can not save the extrinsic parameters\n";

  // OpenCV can handle left-right
  // or up-down camera arrangements
  bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

  // COMPUTE AND DISPLAY RECTIFICATION
  Mat rmap[2][2];
  // IF BY CALIBRATED (BOUGUET'S METHOD)

  //Precompute maps for cv::remap()
  initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
  initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

  Mat canvas;
  double sf;
  int w, h;
  if (!isVerticalStereo)
  {
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h, w * 2, CV_8UC3);
  }
  else
  {
    sf = 300. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width*sf);
    h = cvRound(imageSize.height*sf);
    canvas.create(h * 2, w, CV_8UC3);
  }

  destroyAllWindows();

  Mat imgLeft, imgRight;

  int ndisparities = 16 * 5;   /**< Range of disparity */
  int SADWindowSize = 31; /**< Size of the block window. Must be odd */
  Ptr<StereoBM> sbm = StereoBM::create(ndisparities, SADWindowSize);
  sbm->setMinDisparity(0);
  //sbm->setNumDisparities(64);
  sbm->setTextureThreshold(10);
  sbm->setDisp12MaxDiff(-1);
  sbm->setPreFilterCap(31);
  sbm->setUniquenessRatio(25);
  sbm->setSpeckleRange(32);
  sbm->setSpeckleWindowSize(100);

  Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 64, 7,
    10 * 7 * 7,
    40 * 7 * 7,
    1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

  Mat rimg, cimg;
  Mat Mask;
  while (1)
  {
    camera_l >> frame_l;
    camera_r >> frame_r;

    if (frame_l.empty() || frame_r.empty())
      continue;

    remap(frame_l, rimg, rmap[0][0], rmap[0][1], INTER_LINEAR);
    rimg.copyTo(cimg);
    Mat canvasPart1 = !isVerticalStereo ? canvas(Rect(w * 0, 0, w, h)) : canvas(Rect(0, h * 0, w, h));
    resize(cimg, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
    Rect vroi1(cvRound(validRoi[0].x*sf), cvRound(validRoi[0].y*sf),
      cvRound(validRoi[0].width*sf), cvRound(validRoi[0].height*sf));

    remap(frame_r, rimg, rmap[1][0], rmap[1][1], INTER_LINEAR);
    rimg.copyTo(cimg);
    Mat canvasPart2 = !isVerticalStereo ? canvas(Rect(w * 1, 0, w, h)) : canvas(Rect(0, h * 1, w, h));
    resize(cimg, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);
    Rect vroi2 = Rect(cvRound(validRoi[1].x*sf), cvRound(validRoi[1].y*sf),
      cvRound(validRoi[1].width*sf), cvRound(validRoi[1].height*sf));

    Rect vroi = vroi1&vroi2;

    imgLeft = canvasPart1(vroi).clone();
    imgRight = canvasPart2(vroi).clone();

    rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
    rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

    if (!isVerticalStereo)
    for (int j = 0; j < canvas.rows; j += 32)
      line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    else
    for (int j = 0; j < canvas.cols; j += 32)
      line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);

    cvtColor(imgLeft, imgLeft, CV_BGR2GRAY);
    cvtColor(imgRight, imgRight, CV_BGR2GRAY);

    //-- And create the image in which we will save our disparities
    Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
    Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
    Mat sgbmDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
    Mat sgbmDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

    if (imgLeft.empty() || imgRight.empty())
    {
      std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    sbm->compute(imgLeft, imgRight, imgDisparity16S);

    imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
    cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
    applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
    Mat disparityShow;
    imgDisparity8U.copyTo(disparityShow, Mask);

    sgbm->compute(imgLeft, imgRight, sgbmDisp16S);

    sgbmDisp16S.convertTo(sgbmDisp8U, CV_8UC1, 255.0 / 1000.0);
    cv::compare(sgbmDisp16S, 0, Mask, CMP_GE);
    applyColorMap(sgbmDisp8U, sgbmDisp8U, COLORMAP_HSV);
    Mat  sgbmDisparityShow;
    sgbmDisp8U.copyTo(sgbmDisparityShow, Mask);

    imshow("bmDisparity", disparityShow);
    imshow("sgbmDisparity", sgbmDisparityShow);
    imshow("rectified", canvas);
    char c = (char)waitKey(1);
    if (c == 27 || c == 'q' || c == 'Q')
      break;
  }
  return 0;
}
