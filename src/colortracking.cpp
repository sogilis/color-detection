#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

#define AREA_THRESHOLD 200

using namespace cv;

void track(CvCapture *capture);
void applyThreshold(IplImage *src, IplImage *dst);
void estimatePosition(IplImage *image, int *pX, int *pY);
bool keepTracking();

/* Main function.
 * Initialize the video capture device and launch
 * the tracking task.
 */
int main(int argc, char* argv[])
{
  CvCapture* capture = cvCaptureFromCAM(0);

  if (capture == NULL) {
    std::cout << "Device capture initialization failed.";
    return EXIT_FAILURE;
  }

  track(capture);

  cvReleaseCapture(&capture);
  return EXIT_SUCCESS;
}

/* Tracks the position of a yellow object from the
 * captured video device. This algorithm works well
 * if the scene contains __only__ one yellow element,
 * otherwise it is very likely to show very bad
 * performance.
 *
 * Params:
 *    capture - The video device which can be queried
 *              for more frames.
 *
 * Returns when the user presses a key.
 */
void track(CvCapture *capture)
{
  int posX = -1, posY = -1;

  while (keepTracking()) {
    IplImage* frame = cvQueryFrame(capture);

    if (frame == NULL) {
      break;
    }

    cvFlip(frame, frame, 1);

    IplImage *threshed = cvCreateImage(cvGetSize(frame), 8, 1);
    applyThreshold(frame, threshed);

    estimatePosition(threshed, &posX, &posY);

    if (posX > 0 && posY > 0) {
      CvPoint point = cvPoint(posX, posY);
      cvLine(frame, point, point, cvScalar(0, 255, 255), 5);
    }

    cvShowImage("Threshold", threshed);
    cvShowImage("Video", frame);

    cvReleaseImage(&threshed);
  }
}

/* Applies an intensity threshold to the src image.
 * It basically converts the src image into a black
 * and white image, in which all the yellow shades
 * have been converted to plain white and all the
 * rest is set to black.
 *
 * Params:
 *    src - The original image.
 *    dst - The threshed image.
 */
void applyThreshold(IplImage *src, IplImage *dst)
{
  IplImage *hsvImage = cvCreateImage(cvGetSize(src), 8, 3);
  cvCvtColor(src, hsvImage, CV_BGR2HSV);
  cvInRangeS(hsvImage, cvScalar(20, 100, 100), cvScalar(30, 255, 255), dst);
  cvReleaseImage(&hsvImage);
}

/* Tries to estimate the position of the pixels'
 * intensities barycentre, using the "Moments method".
 * If the maximum intensity detected across the image
 * is not relevant, then both coordinates are set to
 * -1.
 *
 * Further readings:
 *    http://en.wikipedia.org/wiki/Image_moment
 *
 * Params:
 *    image - The original image.
 *    pX    - The barycentre's X coordinate.
 *    pY    - The barycentre's Y coordinate.
 */
void estimatePosition(IplImage *image, int *pX, int *pY)
{
  CvMoments moments = CvMoments();
  cvMoments(image, &moments, 1);

  double moment10 = cvGetSpatialMoment(&moments, 1, 0);
  double moment01 = cvGetSpatialMoment(&moments, 0, 1);
  double area     = cvGetCentralMoment(&moments, 0, 0);

  if (area >= AREA_THRESHOLD) {
    *pX = moment10 / area;
    *pY = moment01 / area;
  } else {
    *pX = -1;
    *pY = -1;
  }
}

/* Basic method to capture user keyboard interaction.
 * 
 * Returns:
 *    true  - if the user has done nothing.
 *    false - if the user has pressed any key.
 */
bool keepTracking()
{
  return cvWaitKey(10) == -1;
}
