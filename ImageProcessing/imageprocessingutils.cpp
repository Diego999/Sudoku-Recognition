#include "imageprocessingutils.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;

#define D_DEBUG 0

// from OpenCV Samples
// finds a cosine of angle between vectors from pt0->pt1 and from pt0->pt2
double image_processing_utils::findAngle(Point p1, Point p2, Point p0)
{
    double dx1 = p1.x - p0.x;
    double dy1 = p1.y - p0.y;
    double dx2 = p2.x - p0.x;
    double dy2 = p2.y - p0.y;
    return (dx1*dx2 + dy1*dy2)/sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
}

// from OpenCV Samples
// returns sequence of squares detected on the image, with a specific size in ratio.
std::vector<std::vector<cv::Point>> image_processing_utils::findSquares(const Mat& image, double minAreaPercentageFilter, double maxAreaPercentageFilter, double wholeArea, const int thresh, const int N)
{
	std::vector<std::vector<Point>> squares;

    Mat pyr, timg, gray0(image.size(), CV_8U), gray;

    // down-scale and upscale the image to filter out the noise
    pyrDown(image, pyr, Size(image.cols/2, image.rows/2));
    pyrUp(pyr, timg, image.size());
    vector<vector<Point> > contours;

    // find squares in every color plane of the image
    for (int c = 0; c < 3; ++c)
    {
        int ch[] = {c, 0};
        mixChannels(&timg, 1, &gray0, 1, ch, 1);

        // try several threshold levels
        for (int l = 0; l < N; ++l)
        {
            // hack: use Canny instead of zero threshold level.
            // Canny helps to catch squares with gradient shading
            if (l == 0)
            {
                // apply Canny. Take the upper threshold from slider and set the lower to 0 (which forces edges merging)
                Canny(gray0, gray, 0, thresh, 5);
                // dilate canny output to remove potential holes between edge segments
                dilate(gray, gray, Mat(), Point(-1,-1));
            }
            else
                // apply threshold if l!=0: tgray(x,y) = gray(x,y) < (l+1)*255/N ? 255 : 0
                gray = gray0 >= (l+1)*255/N;

            // find contours and store them all as a list
            findContours(gray, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            vector<Point> approx;
            for (auto& contour : contours)
            {
                // approximate contour with accuracy proportional to the contour perimeter
                approxPolyDP(Mat(contour), approx, arcLength(Mat(contour), true)*0.02, true);

                // square contours should have 4 vertices after approximation relatively large area (to filter out noisy contours) and be convex.
                double area = 100.0*fabs(contourArea(Mat(approx)))/wholeArea;
                if (approx.size() == 4 && area >= minAreaPercentageFilter && area <= maxAreaPercentageFilter && isContourConvex(Mat(approx)))
                {
                    double maxCosine = 0;
                    for ( int j = 2; j < 5; j++ )
                        // find the maximum cosine of the angle between joint edges
                        maxCosine = std::max(maxCosine, fabs(image_processing_utils::findAngle(approx[j%4], approx[j-2], approx[j-1])));

                    // if cosines of all angles are small (all angles are ~90 degree) then write quandrange vertices to resultant sequence
                    if (maxCosine < 0.3)
                        squares.push_back(approx);
                }
            }
        }
    }
    return squares;
}

void image_processing_utils::drawSquares(Mat& image, const std::vector<std::vector<Point>>& squares, bool R, bool G, bool B)
{
	for (auto& square : squares)
    {
        const Point* p = &square[0];
        int n = (int)square.size();
        char r,g,b;
        r = g = b = 0;
        if(R)
        	r = 255;
        else if(G)
        	g = 255;
        else if(B)
        	b = 255;
        else
        	r = g = b = 128;

        polylines(image, &p, &n, 1, true, Scalar(r,g,b), 3, CV_AA);
    }
}

Mat image_processing_utils::captureSoduku(const int minAreaPercentageFilter, const int maxAreaPercentageFilter, std::vector<Point>& square, const int nbIteration)
{
	VideoCapture capture(0);//0 = default and we assume it always exists
    
    int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    bool isASudokuCaptured = false;

    Mat sudoku;
    namedWindow(image_processing_utils::WEBCAM_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    int ithIteration = 1;
    while(waitKey(10) < 0 && !isASudokuCaptured)
    {
        Mat frame;
        capture >> frame;
        std::vector<std::vector<Point>> squares = image_processing_utils::findSquares(frame, minAreaPercentageFilter, maxAreaPercentageFilter, width*height);
        
        if(squares.size() > 0)
        {
            isASudokuCaptured = ++ithIteration >= nbIteration;
            sudoku = frame.clone();
            
            square = *std::min_element(squares.begin(), squares.end(), [](const vector<Point>& p1, const vector<Point>& p2) -> bool {return contourArea(p1) < contourArea(p2);});

            squares.clear();
        	squares.push_back(square);
        }

        drawSquares(frame, squares);
        imshow(image_processing_utils::WEBCAM_WINDOW_TITLE, frame);
    }

    return sudoku;
}

//Sort the points of a quadrilateral in the following order : TopL, TopR, BoR, BoL
void image_processing_utils::prepareQuadri(std::vector<Point>& quadri)
{
	std::sort(quadri.begin(), quadri.end(), [](const Point& a, const Point& b) -> bool {return a.x < b.x;});

	//Find topL and botL
	Point topL, botL;
	bool topLbotL= quadri[0].y > quadri[1].y;
	topL = quadri[topLbotL ? 1 : 0];
	botL = quadri[topLbotL ? 0 : 1];
	

	Point topR, botR;
	bool topRbotR = quadri[2].y > quadri[3].y;
	topR = quadri[topRbotR ? 3 : 2];
	botR = quadri[topRbotR ? 2 : 3];

	quadri.clear();
	quadri.push_back(topL);
	quadri.push_back(topR);
	quadri.push_back(botR);
	quadri.push_back(botL);
}

Mat image_processing_utils::cropPicture(const std::vector<Point>& srcQuadri, const Mat& srcImg, const int w, const int h)
{
	std::vector<Point> quadri(srcQuadri);
	image_processing_utils::prepareQuadri(quadri);

    Mat dstImg;
    dstImg.create(srcImg.size(), srcImg.type());
 
    Point2f srcPoints[4];
    Point2f dstPoints[4];

	dstPoints[0] = Point2f(0, 0);
    dstPoints[1] = Point2f(w, 0);
    dstPoints[2] = Point2f(w, h);
    dstPoints[3] = Point2f(0, h);
    
    for(int i = 0; i < quadri.size(); ++i)
        srcPoints[i] = quadri[i];

    warpPerspective(srcImg, dstImg, getPerspectiveTransform(srcPoints, dstPoints), Size(w, h));

    #if D_DEBUG
    	Mat img = dstImg.clone();
		namedWindow(image_processing_utils::CROP_SUDOKU_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
	    imshow(image_processing_utils::CROP_SUDOKU_WINDOW_TITLE, img);
    #endif

    return dstImg;
}

//Return true if we found the 81 squares of the sudoku
bool image_processing_utils::removeDuplicateSquares(std::vector<std::vector<Point>>& squares)
{
	std::vector<Rect> rectangles;
	for(auto& square : squares)
		rectangles.push_back(boundingRect(square));

	std::vector<Point> gCenters;
	for(auto& square : squares)
	{
		Point p(0,0);
		for(int i = 0; i < 4; ++i)
		{
			p.x += square[i].x;
			p.y += square[i].y;
		}

		p.x /= 4.0;
		p.y /= 4.0;
		gCenters.push_back(p);
	}

	std::vector<std::vector<Point>> squaresCopy(squares);
	std::vector<bool> visited(rectangles.size(), false);
	squares.clear();
	for(auto& g : gCenters)
	{
		std::vector<std::vector<Point>> rect;
		for(int j = 0; j < rectangles.size(); ++j)
			if(!visited[j] && rectangles[j].contains(g))
			{
				visited[j] = true;
				rect.push_back(squaresCopy[j]);
			}
		
		if(rect.size() > 0)
		{
			sort(rect.begin(), rect.end(), [](const vector<Point>& p1, const vector<Point>&p2) -> bool {return contourArea(p1) > contourArea(p2);});
			squares.push_back(rect[0]);
		}
	}
	
	return squares.size() == 81;
}

//Sort the squares by order where they appeared in the sudoku in row major order
std::vector<std::vector<Point>> image_processing_utils::labelling(const std::vector<std::vector<Point>>& squares)
{
	std::vector<std::vector<Point>> sortedSquares(squares);
	for(auto& square : sortedSquares)
		image_processing_utils::prepareQuadri(square);

	std::vector<std::vector<Point>> labels;
	std::vector<std::vector<Point>> rows;
	for(int i = 0; i < 9; ++i)
	{
		rows.clear();

		//Sort by row
		std::sort(sortedSquares.begin(), sortedSquares.end(), [](const vector<Point>& p1, const vector<Point>& p2) -> bool {return p1[0].y > p2[0].y;});

		//Then sort by col
		for(int j = 0; j < 9; ++j)
		{
			rows.push_back(sortedSquares.back());
			std::sort(rows.begin(), rows.end(), [](const vector<Point>& p1, const vector<Point>& p2) -> bool {return p1[0].x < p2[0].x;});
			sortedSquares.pop_back();
		}
		for(auto& square : rows)
			labels.push_back(square);
	}

	return labels;
}

std::vector<Mat> image_processing_utils::extractBlocks(const std::vector<std::vector<Point>>& unlabeledSquares, const Mat& image)
{
	std::vector<std::vector<Point>> squares = image_processing_utils::labelling(unlabeledSquares);

	#if D_DEBUG
		Mat img = image.clone();
		drawSquares(sudoku, squares, 0, true, 0);
		int i = 0;
		for(auto& square : squares)
		{
			string lbl = to_string(i++);
			putText(img, lbl, Point(15+square[0].x,15+square[0].y), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0));
		}
		namedWindow(image_processing_utils::LABELED_SUDOKU_WINDOW_TITLE, CV_WINDOW_AUTOSIZE);
    	imshow(image_processing_utils::LABELED_SUDOKU_WINDOW_TITLE, img);
	#endif

	vector<Mat> blocks;
	int w = 0;
	int h = 0;
	
	for(auto& square : squares)
	{
		w += abs(square[1].x-square[0].x);
		h += abs(square[3].y-square[0].y);
	}
	
	w /= squares.size();
	h /= squares.size();
	
	for(auto& square : squares)
		blocks.push_back(image_processing_utils::cropPicture(square, image, w, h));

    for(auto& b : blocks)
        b = b > 128;

	return blocks;
}
