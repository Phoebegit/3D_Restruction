#include <string>
#include <stdio.h>
#include <iostream>
#include <GL/freeglut.h>
#include <opencv2/opencv.hpp>
//#include "opencv2/calib3d/calib3d.hpp"
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui/highgui.hpp"

using namespace std;
using namespace cv;

void saveXYZ(const char* filename, const Mat& mat) ;
//distance
int XY[2] ;
static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/) ;
void detectDistance(Mat& pointCloud) ;
//GL
int rx = 0 , ry = 0 ;
int eyex = 115, eyez = 115, atx = 100, atz = 50;
float imgdata[500][500][3];
float texture[500][500][3] ; //存放纹理数据
int glWinWidth = 0 ;
int glWinHeight = 0;
float scalar=50;
bool leftClickHold = false, rightClickHold = false ;
void renderScene(void) ;
void reshape (int glWinWidth, int glWinHeight) ;
void load3dDataToGL(IplImage* img3d) ;
void loadTextureToGL(IplImage *img) ;  //载入左视图的纹理图
void special(int key, int x, int y) ;
void mouse(int button, int state, int x, int y) ;

int main(int argc , char* argv[])
{
  cv::VideoCapture camera_l(1);
  cv::VideoCapture camera_r(0);

  camera_l.set(CAP_PROP_FRAME_WIDTH, 320);
  camera_l.set(CAP_PROP_FRAME_HEIGHT, 240);
  camera_r.set(CAP_PROP_FRAME_WIDTH, 320);
  camera_r.set(CAP_PROP_FRAME_HEIGHT, 240);

  if (!camera_l.isOpened()){ cout << "No left camera!" << endl; return -1; }
  if (!camera_r.isOpened()){ cout << "No right camera!" << endl; return -1; }

  Mat cameraMatrix[2], distCoeffs[2];

  FileStorage fs("intrinsics.yml", FileStorage::READ);
  if (fs.isOpened())
  {
    fs["M1"] >> cameraMatrix[0];
    fs["D1"] >> distCoeffs[0];
    fs["M2"] >> cameraMatrix[1];
    fs["D2"] >> distCoeffs[1];
    fs.release();
  }
  else
    cout << "Error: can not save the intrinsic parameters\n";

  Mat R, T, E, F;
  Mat R1, R2, P1, P2, Q;
  Rect validRoi[2];
  Size imageSize(320, 240);

  fs.open("extrinsics.yml", FileStorage::READ);
  if (fs.isOpened())
  {
    fs["R"] >> R;
    fs["T"] >> T;
    fs["R1"] >> R1;
    fs["R2"] >> R2;
    fs["P1"] >> P1;
    fs["P2"] >> P2;
    fs["Q"] >> Q;
    fs.release();
  }
  else
    cout << "Error: can not save the extrinsic parameters\n";

  stereoRectify(cameraMatrix[0], distCoeffs[0],
    cameraMatrix[1], distCoeffs[1],
    imageSize, R, T, R1, R2, P1, P2, Q,
    CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

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
  cv::Mat frame_l, frame_r;
  Mat imgLeft, imgRight;

  int ndisparities = 16 * 5;   /**< Range of disparity */
  int SADWindowSize = 31; /**< Size of the block window. Must be odd */
  Ptr<StereoBM> BMState = StereoBM::create(ndisparities, SADWindowSize);
      BMState->setMinDisparity(0);
      BMState->setNumDisparities(64);
      BMState->setTextureThreshold(10);
      BMState->setDisp12MaxDiff(-1);
      BMState->setPreFilterCap(31);
      BMState->setUniquenessRatio(25);
      BMState->setSpeckleRange(32);
      BMState->setSpeckleWindowSize(100);

  Ptr<StereoSGBM> SGBM = StereoSGBM::create(0, 64, 7,
    10 * 7 * 7,
    40 * 7 * 7,
    1, 63, 10, 100, 32, StereoSGBM::MODE_SGBM);

    glutInit(&argc , argv);
    //glutInitDisplayMode(GLUT_DEPTH | GLUT_SINGLE | GLUT_RGBA);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowPosition(100,100);
    glutInitWindowSize(glWinWidth , glWinHeight);
    glutCreateWindow("3D Image");

  Mat rimg, cimg;
  Mat Mask;
  for(;;)
  {
    camera_l >> frame_l;
    camera_r >> frame_r;

    if (frame_l.empty() || frame_r.empty())
      break;

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
    //imshow("imgleft" , imgLeft) ;
    rectangle(canvasPart1, vroi1, Scalar(0, 0, 255), 3, 8);
    rectangle(canvasPart2, vroi2, Scalar(0, 0, 255), 3, 8);

    if (!isVerticalStereo)
    for (int j = 0; j < canvas.rows; j += 32)
      line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    else
    for (int j = 0; j < canvas.cols; j += 32)
      line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
    Mat grayImgLeft , grayImgRight ;
    cvtColor(imgLeft, grayImgLeft, CV_BGR2GRAY);
    cvtColor(imgRight, grayImgRight, CV_BGR2GRAY);

    //-- And create the image in which we will save our disparities
    Mat imgDisparity16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
    Mat imgDisparity8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);
    Mat SGBMDisp16S = Mat(imgLeft.rows, imgLeft.cols, CV_16S);
    Mat SGBMDisp8U = Mat(imgLeft.rows, imgLeft.cols, CV_8UC1);

    if (imgLeft.empty() || imgRight.empty())
    {
      std::cout << " --(!) Error reading images " << std::endl; return -1;
    }

    BMState->compute(grayImgLeft, grayImgRight, imgDisparity16S);

    imgDisparity16S.convertTo(imgDisparity8U, CV_8UC1, 255.0 / 1000.0);
    cv::compare(imgDisparity16S, 0, Mask, CMP_GE);
    applyColorMap(imgDisparity8U, imgDisparity8U, COLORMAP_HSV);
    Mat disparityShow;
    imgDisparity8U.copyTo(disparityShow, Mask);

    SGBM->compute(grayImgLeft, grayImgRight, SGBMDisp16S);

    SGBMDisp16S.convertTo(SGBMDisp8U, CV_8UC1, 255.0 / 1000.0);
    Mat Image3D ;
    reprojectImageTo3D(SGBMDisp8U , Image3D , Q , true , -1) ;
    for (int y = 0; y < Image3D.rows; ++y)
    {
        for (int x = 0; x < Image3D.cols; ++x)
        {
            cv::Point3f point = Image3D.at<cv::Point3f>(y, x);
            point.y = -point.y;
            Image3D.at<cv::Point3f>(y, x) = point;
        }
    }
    saveXYZ("Image3D.txt" , Image3D) ;

    cv::compare(SGBMDisp16S, 0, Mask, CMP_GE);
    applyColorMap(SGBMDisp8U, SGBMDisp8U, COLORMAP_HSV);
    Mat  SGBMDisparityShow;
    SGBMDisp8U.copyTo(SGBMDisparityShow, Mask);

    imshow("bmDisparity", disparityShow);
    imshow("SGBMDisparity", SGBMDisparityShow);
    imshow("rectified", canvas);

    setMouseCallback("SGBMDisparity", onMouse, 0);
    detectDistance(Image3D) ;
    //GL
    IplImage Img3DIpl = Image3D ;
    IplImage TextureImg = imgLeft ;
    glWinWidth = Img3DIpl.width ;
    glWinHeight = Img3DIpl.height ;
    loadTextureToGL(&TextureImg) ;
    load3dDataToGL(&Img3DIpl) ;
    glutDisplayFunc(renderScene);
    glutMouseFunc(mouse);                                // 鼠标按键响应
    glutReshapeFunc (reshape);
    glutSpecialFunc(special);
    glutPostRedisplay() ; //刷新画面
    char c = (char)waitKey(1);
    if (c == 27 || c == 'q' || c == 'Q')
          break;
    glutMainLoopEvent();
  }
  return 0;
}

void saveXYZ(const char* filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    FILE* fp = fopen(filename, "wt");
    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
            fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
        }
    }
    fclose(fp);
}

void renderScene(void)
{
    glClear (GL_COLOR_BUFFER_BIT);
    glLoadIdentity();// Reset the coordinate system before modifying
    gluLookAt (eyex-100, 0.0, eyez-100.0, atx-100.0, 0.0, atz-100.0, 0.0, 1.0, 0.0);    // 根据滑动块位置变换OpenGL摄像机视角
    glRotatef(ry, 0.0, 1.0, 0.0); //rotate about the z axis            // 根据键盘方向键按键消息变换摄像机视角
    glRotatef(rx-180, 1.0, 0.0, 0.0); //rotate about the y axis

    float x,y,z;

    glPointSize(1.0);
    glBegin(GL_POINTS);//GL_POINTS
    for (int i=0;i<glWinHeight;i++){
        for (int j=0;j<glWinWidth;j++){
            glColor3f(texture[i][j][0]/255, texture[i][j][1]/255, texture[i][j][2]/255);    // 将图像纹理赋值到点云上
            x=-imgdata[i][j][0]/scalar;        // 添加负号以获得正确的左右上下方位
            y=-imgdata[i][j][1]/scalar;
            z=imgdata[i][j][2]/scalar;
            glVertex3f(x,y,z);
        }
    }
    glEnd();
//    glFlush();
   glutSwapBuffers() ;
}

void reshape (int glWinWidth, int glWinHeight)
{
    glViewport (0, 0, (GLsizei)glWinWidth, (GLsizei)glWinHeight);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (60, (GLfloat)glWinWidth / (GLfloat)glWinHeight, 1.0, 500.0);
    glMatrixMode (GL_MODELVIEW);
}

void load3dDataToGL(IplImage* img3d)  //载入三维坐标数据
{
    CvScalar s ;
    for(int i = 0 ; i < glWinHeight ; i++)
    {
        for(int j = 0 ; j < glWinWidth ; j++)
        {
            s = cvGet2D(img3d , i , j) ;
            imgdata[i][j][0] = s.val[0] ;
            imgdata[i][j][1] = s.val[1] ;
            imgdata[i][j][2] = fabs(s.val[2]) ;
        }
    }
}

void loadTextureToGL(IplImage *img)  //载入左视图的纹理图
{
    CvScalar ss ;
    for(int i = 0 ; i < glWinHeight ; i++)
    {
        for(int j = 0 ; j < glWinWidth ; j++)
        {
         // opencv是BGR格式，opengl是RGB格式
            ss = cvGet2D(img , i , j) ;
            texture[i][j][2] = ss.val[0] ;
            texture[i][j][1] = ss.val[1] ;
            texture[i][j][0] = ss.val[2] ;
        }
    }
}

void special(int key, int x, int y)
{
    switch(key)
    {
    case GLUT_KEY_LEFT:
        ry-=5;
        glutPostRedisplay();
        break;
    case GLUT_KEY_RIGHT:
        ry+=5;
        glutPostRedisplay();
        break;
    case GLUT_KEY_UP:
        rx+=5;
        glutPostRedisplay();
        break;
    case GLUT_KEY_DOWN:
        rx-=5;
        glutPostRedisplay();
        break;
    }
}

// 鼠标按键响应函数
void mouse(int button, int state, int x, int y)
{
        if(button == GLUT_LEFT_BUTTON)
        {
                if(state == GLUT_DOWN)
                {
                        leftClickHold=true;
                }
                else
                {
                        leftClickHold=false;
                }
        }
        if (button== GLUT_RIGHT_BUTTON)
        {
                if(state == GLUT_DOWN)
                {
                        rightClickHold=true;
                }
                else
                {
                        rightClickHold=false;
                }
        }
}

static void onMouse(int event, int x, int y, int /*flags*/, void* /*param*/)
{
    if( event != EVENT_LBUTTONDOWN )
        return;
    if (event == CV_EVENT_LBUTTONDOWN)
    {
        XY[0] = x;
        XY[1] = y;
        cout << "x:" << XY[0] << "y:" << XY[1] << endl;
    }
}

void detectDistance(Mat& pointCloud)
{
    if (pointCloud.empty())
        return;
    // 提取深度图像
    vector<cv::Mat> xyzSet;
    split(pointCloud, xyzSet);
    cv::Mat depth;
    xyzSet[2].copyTo(depth);
    // 根据深度阈值进行二值化处理
    double maxVal = 0, minVal = 0;
    cv::Mat depthThresh = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
    cv::minMaxLoc(depth, &minVal, &maxVal);
    double thrVal = minVal * 1.5;
    threshold(depth, depthThresh, thrVal, 255, CV_THRESH_BINARY_INV);
    depthThresh.convertTo(depthThresh, CV_8UC1);

    double  distance = depth.at<float>(XY[0], XY[1]);
    cout << "distance:" << distance << endl;
}

