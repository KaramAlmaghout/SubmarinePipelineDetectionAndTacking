/**
* pipelineIntersectLines - detect submarine pipeline 
* by generating lines using linear regression of the contours and check the intersection points among these lines  
*
* @author: Karam Almaghout
* 
*/

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>


using namespace std::chrono;
using namespace std;
using namespace cv;

Mat src_img, src_gray, edges, dst, dstV, dstH, polar_img, lab_img, image_clahe, cfar_img, gray_img, patches_img;
int vertical_size, horizontal_size;
int width, height;
int edgeThresh = 100;
int lowThreshold;
int highThreshold;
int const max_lowThreshold = 200, max_highThreshold = 255;
int kernel_size = 3;
const char* res_window = "Edge Map";
vector<Point> S;
RNG rng(12345);


bool intersection(Point2i o1, Point2i p1, Point2i o2, Point2i p2,Point2i &r)
{
    Point2i x = o2 - o1;
    Point2i d1 = p1 - o1;
    Point2i d2 = p2 - o2;

    float cross = d1.x*d2.y - d1.y*d2.x;
    if (abs(cross) < /*EPS*/1e-8)
        return false;

    double t1 = (x.x * d2.y - x.y * d2.x)/cross;
    r = o1 + d1 * t1;
    return true;
}

bool sortbysec(const Point &a,
              const Point &b)
{

    return (a.y > b.y);
}

void CACFAR()
{
    Mat src_img1;
    image_clahe.copyTo(src_img1);
    int h, w;
    h = src_img1.rows;
    w = src_img1.cols;
    src_img1.copyTo(cfar_img);
    Point cut;
    double Pfa = 7000; //false alaram probability 
    int i, j, step = 19; // reference cells (region n+9, including cut and guarding cells)
    int test_cells = 5, guardig_cells = 3; // guarding_cells is the boundray around the CUT
    int shift = (step-1)/2;
    // avoid boundary pixels 
    int n0 = step*step, n = 0, blue = 0, red = 0, green = 0;
    double  beta_r, beta_b, beta_g, T_b, T_r, T_g;
    long double alpha; 
    Vec3b pixelValue = src_img1.at<Vec3b>(0,0);
    Vec3b zeros(0,0,0);
    
    for(j=0; j<w-step; j++)
    {
        for(i=0; i<h-step; i++)
        {
            // set cell under test (cut)
            n = 0;
            cut.x = i+shift;
            cut.y = j+shift;
            for (int m = i; m<i+step ; m++)
            {
                for (int k=j; k<j+step; k++)
                {
                    // skip guarding pixels
                    if ((abs(k-cut.y) < test_cells && (abs(m-cut.x) < test_cells)))
                    {
                        
                        continue;
                    }

                    pixelValue = src_img1.at<Vec3b>(m,k);
                    // skip boundraies
                    if (pixelValue.val[0] >240 && pixelValue.val[1] >240 && pixelValue.val[2] >240)
                    {
                        continue;
                    }
                    else
                    {
                        n = n+1;
                        blue += pixelValue.val[0];
                        green += pixelValue.val[1];
                        red += pixelValue.val[2];
                    }
                }

            }
            if (n > 0)
            {
                beta_b = blue/n;
                beta_r = red/n;
                beta_g = green/n;
                alpha = 1.4;
                T_b = alpha*beta_b;
                T_g = alpha*beta_g;
                T_r = alpha*beta_r;
                for (int e=cut.x-2;e<cut.x+2;e++)
                {
                    for (int q=cut.y-2;q<cut.y+2;q++)
                    {
                        pixelValue = src_img1.at<Vec3b>(e,q);
                        if (pixelValue.val[0] < T_b && pixelValue.val[1] < T_g && pixelValue.val[2] < T_r)
                        {
                            cfar_img.at<Vec3b>(cut.x, cut.y) = zeros;
                        }

                    }
                    
                }
                
                
                n = 0;
            }
            else
            {
                cfar_img.at<Vec3b>(cut.x, cut.y) = zeros;
            }
            blue = 0;
            red = 0;
            green = 0;
            n = 0;

        }
        
    }
    Mat morphoKernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    dilate(cfar_img, cfar_img, morphoKernel, Point(-1, -1), 2);
    cv::cvtColor(cfar_img, gray_img, CV_BGR2GRAY);
}

void ExtractObjects()
{
    
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( gray_img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
    vector<Moments> mu(contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
        { mu[i] = moments( contours[i], false ); }
    vector<Point2f> mc( contours.size() );
    for( size_t i = 0; i < contours.size(); i++ )
        { mc[i] = Point2f( static_cast<float>(mu[i].m10/mu[i].m00) , static_cast<float>(mu[i].m01/mu[i].m00) ); }
    Mat drawing = Mat::zeros( gray_img.size(), CV_8UC3 );
    for( size_t i = 0; i< contours.size(); i++ )
        {
        Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        drawContours( drawing, contours, (int)i, color, 2, 8, hierarchy, 0, Point() );
        putText(drawing, //target image
            to_string(i), //text
            mc[i], //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            color, //font color
            1);
        }
    
    Mat Felong = Mat::zeros(1, contours.size(), CV_64FC1);

    double f, u1, u2, mu22 = 0, max_f = 0;
    int pipe_cnt = 0;
    for( size_t i = 0; i< contours.size(); i++ )
        {
        mu22 = mu[i].mu11;
        u1 = 0.5*(mu[i].mu20 + mu[i].mu02 + sqrt(pow(mu[i].mu20 - mu[i].mu02,2) + 4*pow(mu22,2)));
        u2 = 0.5*(mu[i].mu20 + mu[i].mu02 - sqrt(pow(mu[i].mu20 - mu[i].mu02,2) + 4*pow(mu22,2)));
        f = sqrt(u1/u2);
        Felong.at<double>(0, i) = f;
        }
    double min, max;
    minMaxLoc(Felong, &min, &max);
    Felong = Felong/max;
    patches_img = Mat::zeros( gray_img.size(), CV_8UC3 );
    cv::Vec4f line_;
    Scalar color = Scalar(255,255,255);
    vector<Point> linesPnts, linesPnts0;
    // vertical line
    Point ver_p1, ver_p2;
    ver_p1.x = (patches_img.cols)/2;
    ver_p2.x = (patches_img.cols)/2;
    ver_p1.y = (patches_img.rows-1);
    ver_p2.y = 0;
    linesPnts.push_back(ver_p1);
    linesPnts.push_back(ver_p2);
    linesPnts0.push_back(ver_p1);
    Point cnts_sort_p;
    vector<Point> cnts_sort;
    cnts_sort_p.x = 0;
    cnts_sort_p.y = ver_p1.y;
    cnts_sort.push_back(cnts_sort_p);
    for( size_t i = 0; i< contours.size(); i++ )
    {
        if (Felong.at<double>(0, i) > 0.35)
        {
            
            drawContours( patches_img, contours, i, color, -1, 8, hierarchy, 0, Point() );
            // cout << contours.size() << endl;
            fitLine(contours[i], line_, DIST_L2, 0, 0.01, 0.01);
            float vx = line_[0];
            float vy = line_[1];
            float x = line_[2];
            float y = line_[3];
            Point line0;
            line0.x = x;
            line0.y = y;
            // cout << cnts_sort << endl;
            linesPnts0.push_back(line0);
            
            cnts_sort_p.x++;
            cnts_sort_p.y = y;
            cnts_sort.push_back(cnts_sort_p);
            // cout << cnts_sort << endl;
            cnts_sort_p.y = 0;
            
            int lefty = (int) ((-x*vy/vx) + y);
            int righty = (int) (((patches_img.cols-x)*vy/vx)+y);
            Point p1(patches_img.cols-1,righty), p2(0,lefty);
            linesPnts.push_back(p1);
            linesPnts.push_back(p2);
            circle( patches_img, line0, 4, Scalar(127, 127, 127), -1, 8, 0 );
        }
    }

    Point pixel_(0,0);
    for (int i = 0; i<patches_img.rows ;i++)
    {
        for (int j = 0; j<patches_img.cols; j++)
        {
            if (patches_img.at<Vec3b>(i,j).val[0] == 255 && patches_img.at<Vec3b>(i,j).val[1] == 255 && patches_img.at<Vec3b>(i,j).val[2] == 255)
            {
                pixel_.x = j;
                pixel_.y = i;
                S.push_back(pixel_);

            }
        }
    }

    // imshow( "src_Contours", patches_img );
}

void PipelineSOM()
{
    Mat cfar_img1;
    patches_img.copyTo(cfar_img1);
    // decleare a set of points:
    int w = cfar_img1.cols;
    int h = cfar_img1.rows;
    int q_size = 100;
    vector< Point> q(q_size);
    int qh = (int)  h/(q_size+1);
    vector<int> index;
    // initialization
    Scalar color = Scalar( 127, 127, 127 );
    for (int i=0; i<q_size ; i++)
    {
        q[i].x = (int) w/2;
        q[i].y = h - qh*(i+1);
    }
    for (size_t n = 0; n<100; n++)
   {
    int random = rand() % S.size();
    int randomP, x = 0;
    circle( cfar_img1, S[random], 4, color, -1, 8, 0 );
    double dis = 1000000, min_dis = 10000;
    int winning_q;
    for (size_t i = 0; i<q_size; i++)
    {
        dis = sqrt(pow((q[i].x-S[random].x),2) + pow((q[i].y-S[random].y),2));
        if (dis < min_dis)
        {
            min_dis = dis;
            winning_q = i;
        }
    }
    
    // Excitation Propagation
    double sigma0 = 5, tau_s = 0.5, gamma0 = 4, tau_g = 0.5;
    double sigma, gamma, h_p;
    sigma = sigma0*exp(-1/tau_s);
    gamma = gamma0*exp(-1/tau_g);
    double A = 0, B = 2*pow(sigma, 2);
    int wx = q[winning_q].x;
    int wy = q[winning_q].y;
    for (size_t i = 0; i<q_size; i++)
    {
        if (find(index.begin(), index.end(), i) != index.end())
        continue;
        else
        {
            if ( i == winning_q)
            {
                q[i].x = S[random].x;
                q[i].y = S[random].y;
            }
            else
            {
            A = sqrt(pow(q[i].x -wx,2) + pow(q[i].y - wy,2));
            h_p =  exp(-A/B);
            q[i].x = wx + gamma*h_p*(S[random].x - q[i].x);
            q[i].y = wy + gamma*h_p*(S[random].y - q[i].y);
            }
        }
    }
    index.push_back(winning_q);
        

   }
    vector<Point> q_final, q_final1;
    q_final.push_back(q[0]);
    for (size_t i = 1; i<q.size() ; i++)
    {
        if (find(q_final.begin(), q_final.end(), q[i]) != q_final.end())
        {
            continue;
        }
        else
        {
            q_final.push_back(q[i]);

        }
    }
    sort(q_final.begin(), q_final.end(), sortbysec);
    double dis = 10000;
    int ind, ind_p;
    vector<int> index1;
    q_final1.push_back(q_final[0]);
    index1.push_back(0);
    for (size_t i = 0; i<q_final.size()-1; i++)
    {
        for (size_t j = 1; j<q_final.size();j++)
        {
            if (find(index1.begin(), index1.end(), j) != index1.end())
        {
            continue;
        }
        else
        {
            if (sqrt(pow((q_final1[i].x - q_final[j].x),2) + pow((q_final1[i].y - q_final[j].y),2))<dis && sqrt(pow((q_final1[i].x - q_final[j].x),2) + pow((q_final1[i].y - q_final[j].y),2)) > 0)
            {
                
                dis = sqrt(pow(q_final1[i].x - q_final[j].x,2) + pow(q_final1[i].y - q_final[j].y,2));
                ind_p = ind;
                ind = j;
            }
        }
         
            
        }
        index1.push_back(ind);
        dis = 10000;
        q_final1.push_back(q_final[ind]);
        
    }
    
    color = Scalar( 0, 0, 255);
    const Point *pts = (const cv::Point*) Mat(q_final1).data;
    int npts = q_final1.size();
    color = Scalar( 255, 255, 0 );
    
    polylines(src_img, &pts, &npts, 1, false, color);
        for (int i=0; i<q_size ; i++)
    {
 
        circle( src_img, q[i], 3, color, -1, 8, 0 );
    }
    
    imshow( "piplineSOM_output", src_img );
    
}

int main(int argc, char** argv )
{
    auto start = high_resolution_clock::now();
    
    if ( argc < 2 )
    {
        src_img = imread("imgs/sonar3.png", IMREAD_COLOR  );
    }
    else
    {
        src_img = imread( argv[1], IMREAD_COLOR  );
    }
    
    if ( !src_img.data )
    {
        printf("No image data \n");
        return -1;
    }

    // scale down the image size to reduce computation cost
    resize(src_img, src_img, Size(), 0.5, 0.5, INTER_LINEAR);
    height = src_img.rows;
    width = src_img.cols;
    imshow( "src_img", src_img );
    
    // convert the image color space from 'BGR' to 'Lab' to adjust the lightness histogram of the image (L channel)
    cv::cvtColor(src_img, lab_img, CV_BGR2Lab);

    // Extract the L channel
    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_img, lab_planes);  // now we have the L image in lab_planes[0]

    // apply the CLAHE algorithm to the L channel
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
    clahe->setClipLimit(4);
    cv::Mat res;
    clahe->apply(lab_planes[0], res);

    // Merge the the color planes back into an Lab image
    res.copyTo(lab_planes[0]);
    cv::merge(lab_planes, lab_img);

    // convert back to RGB
    cv::cvtColor(lab_img, image_clahe, CV_Lab2BGR);
    // imshow("lab_img", image_clahe);

    CACFAR();

    ExtractObjects();

    PipelineSOM();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << "exec_time [ms]:" << duration.count()/1000 << endl;

    waitKey(0);
    
    return 0;
}