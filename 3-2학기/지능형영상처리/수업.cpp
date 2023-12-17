#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include "my.h"
#include <opencv2/opencv.hpp>

using namespace cv;

typedef struct {
	int r, g, b;
}int_rgb;


int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i < height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i < height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i < height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); // 0        Ե        

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y)
{
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx) * (1.0 - dy) * A + dx * (1.0 - dy) * B
		+ (1.0 - dx) * dy * C + dx * dy * D;

	return((int)(value + 0.5));
}


void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
						// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);
}

//2주차 픽셀다루기
void Ex0907_0(
	int** img, // 입력영상
	int height, int width, // 영상 크기
	int value, //밝기
	int y0, int y1, // y의 영역
	int x0, // x의 영역
	int x1) {

	for (int i = y0; i < y1; i++) {
		for (int j = x0; j < x1; j++) {
			img[i][j] = value;
		}
	}
}

struct PARAMETER {
	int threshold;
	int y0;
	int size_y;
	int x0;
	int size_x; // 범위
	int** img_in;
	int height;
	int width;
	int** img_out;
};

//이진화
void Thresholding_Part(
	int threshold,
	int y0, int size_y, int x0, int size_x, // 범위
	int** img_in,
	int height, int width,
	int** img_out) {

	for (int y = y0; y < y0 + size_y; y++) {
		for (int x = x0; x < x0 + size_x; x++) {
			if (img_in[y][x] >= threshold) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}
}

//이진화+구조체
void Thresholding_Part2(PARAMETER para) {
	for (int y = para.y0; y < para.y0 + para.size_y; y++) {
		for (int x = para.x0; x < para.x0 + para.size_x; x++) {

			if (para.img_in[y][x] >= para.threshold) para.img_out[y][x] = 255;
			else para.img_out[y][x] = 0;
		}
	}
}

int Ex0911_0() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	int y0, x0, size_y = height / 2, size_x = width / 2;

	y0 = 0; x0 = 0;
	Thresholding_Part(50, y0, size_y, x0, size_x, img, height, width, img_out);

	y0 = 0; x0 = width / 2;
	Thresholding_Part(100, y0, size_y, x0, size_x, img, height, width, img_out);

	y0 = height / 2; x0 = 0;
	Thresholding_Part(150, y0, size_y, x0, size_x, img, height, width, img_out);

	y0 = height / 2; x0 = width / 2;
	Thresholding_Part(200, y0, size_y, x0, size_x, img, height, width, img_out);

	return 0;
}

int Ex0911_1() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	for (int y = 0; y < height / 2; y++) {
		for (int x = 0; x < width / 2; x++) {
			if (img[y][x] >= 50) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}

	for (int y = 0; y < height / 2; y++) {
		for (int x = width / 2; x < width; x++) {
			if (img[y][x] >= 100) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}

	for (int y = height / 2; y < height; y++) {
		for (int x = 0; x < width / 2; x++) {
			if (img[y][x] >= 150) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}

	for (int y = height / 2; y < height; y++) {
		for (int x = width / 2; x < width; x++) {
			if (img[y][x] >= 200) img_out[y][x] = 255;
			else img_out[y][x] = 0;
		}
	}
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//임계값 입력 후 이진화
void Thresholding2(int threshold, IMAGE input, IMAGE output) {
	for (int y = 0; y < input.height; y++) {
		for (int x = 0; x < input.width; x++) {

			if (input.data[y][x] >= threshold) output.data[y][x] = 255;
			else output.data[y][x] = 0;
		}
	}
}

int Ex0911_2() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	IMAGE input, output;

	input.data = img;
	input.height = height;
	input.width = width;

	output.data = img_out;
	output.height = height;
	output.width = width;

	Thresholding2(128, input, output);

	ImageShow((char*)"output", output.data, output.height, output.width);

	return 0;
}


//3주차 클리핑과 영상혼합
#define IMAX(x, y) ((x>y) ? x : y)
#define IMIN(x, y) ((x<y) ? x : y)

//밝기값 표현 범위 벗어나는 영상 만들기
void AddValue(int** img, int height, int width, int** img_out, int value) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x] + value;

			// if (img_out[y][x] < 0) img_out[y][x] = 0;
			// img_out[y][x] = (img_out[y][x] < 0) ? 0 : img_out[y][x];
			// img_out[y][x] = IMAX(img_out[y][x], 0);

			// if (img_out[y][x] > 255) img_out[y][x] = 255;
			// img_out[y][x] = (img_out[y][x] >255) ? 255 : img_out[y][x];
			// img_out[y][x] = IMIN(img_out[y][x], 255);

			// img_out[y][x] = IMIN(IMAX(img_out[y][x], 0), 255);
			img_out[y][x] = IMIN(IMAX(img[y][x] + value, 0), 255);
		}
	}
}

int Ex0915() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	AddValue(img, height, width, img_out, 50);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

int Ex0918_0() {
	int a[7] = { 1, 2, -3, -10, 5, 9, -1 };
	int maxValue = -INT_MAX; // INT_MIN

#if 0
	maxValue = a[0];
	maxValue = IMAX(maxValue, a[1]);
	maxValue = IMAX(maxValue, a[2]);
	maxValue = IMAX(maxValue, a[3]);
	maxValue = IMAX(maxValue, a[4]);
	maxValue = IMAX(maxValue, a[5]);
	maxValue = IMAX(maxValue, a[6]);


#else
	maxValue = a[0];
	for (int i = 1; i < 7; i++) {
		maxValue = IMAX(maxValue, a[i]);
	}
#endif

	return 0;
}

int Ex0918_1() { // 반올림
	float a = 1.5, b = 1.4;
	int a_int, b_int;

	a_int = (int)(a + 0.5);
	b_int = (int)(b + 0.5);

	return 0;
}

void ImageMixing(int** img0, int** img1, int height, int width, int** img_out, float alpha) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = (int)(alpha * img0[y][x] + (1 - alpha) * img1[y][x] + 0.5);
		}
	}
}

int Ex0918_2() {
	int height, width;
	int** img_0 = ReadImage((char*)"barbara.png", &height, &width);
	int** img_1 = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	/*for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = (int)(alpha * img_0[y][x] + (1 - alpha) * img_1[y][x] + 0.5);
		}
	}*/
	float alpha = 0.5;

	for (float alpha = 0.1; alpha < 1.0; alpha += 0.1) {
		ImageMixing(img_0, img_1, height, width, img_out, alpha);
		ImageShow((char*)"output", img_out, height, width);

	}

	return 0;
}

int Ex0918_3() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] <= 128) {
				img_out[y][x] = 255.0 / 128 * img[y][x] + 0.5; // float형으로 255 -> 255.0
			}
			else if (img[y][x] > 128) {
				img_out[y][x] = 255;
			}
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//4주차 스트레칭과히스토그램평활화

//스트레칭
void Stretching_1(int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] <= 128) {
				img_out[y][x] = 255.0 / 128 * img[y][x] + 0.5;
			}
			else if (img[y][x] > 128) {
				img_out[y][x] = 255;
			}
		}
	}
}

//감마보정
void GammaCorrection(int** img, int** img_out, int height, int width, double gamma) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = pow(img[y][x] / 255.0, gamma) * 255 + 0.5;
		}
	}
}

int Ex0918_4() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	double gamma[8] = { 0.1, 0.2, 0.4, 0.67, 1.0, 1.5, 2.0, 2.5 };

	for (int i = 0; i < 8; i++) {
		GammaCorrection(img, img_out, height, width, gamma[i]);
		ImageShow((char*)"output", img_out, height, width);
	}

	return 0;
}

//선형 스트레칭 변환
int Stransform(int input, int a, int b, int c, int d) { // 결과값이 하나인 경우 용이
	int output;

	if (input <= a) {
		output = (float)c / a * input + 0.5;
	}
	else if (input <= b) { // <- a < img[y][x] && img[y][x] <= b 

		// img_out[y][x] = (float)(d - c) / (b - a) * (img[y][x] - a) + c + 0.5; 
		// 문제점: 계산 순서로 인해 예상치 못한 결과가 나올 수 있음. 바로 변수 앞에 써줘야 함.

		output = ((float)d - c) / (b - a) * (input - a) + c + 0.5;
	}
	else {
		output = (255.0 - d) / (255.0 - b) * (input - b) + d + 0.5;
	}

	return output;
}

//선형 스트레칭 변환+포인터
void StransformPtr(int input, int a, int b, int c, int d, int* output) { // 포인터 사용, 결과값이 여러 개인 경우 용이
	if (input <= a) {
		*output = (float)c / a * input + 0.5;
	}
	else if (input <= b) { // <- a < img[y][x] && img[y][x] <= b 

		// img_out[y][x] = (float)(d - c) / (b - a) * (img[y][x] - a) + c + 0.5; 
		// 문제점: 계산 순서로 인해 예상치 못한 결과가 나올 수 있음. 바로 변수 앞에 써줘야 함.

		*output = ((float)d - c) / (b - a) * (input - a) + c + 0.5;
	}
	else {
		*output = (255.0 - d) / (255.0 - b) * (input - b) + d + 0.5;
	}

}

//선형 스트레칭 변환 + 참조변수
void StransformRef(int input, int a, int b, int c, int d, int& output) { // 레퍼런스 변수(output) 사용
	if (input <= a) {
		output = (float)c / a * input + 0.5; //반올림
	}
	else if (input <= b) { // <- a < img[y][x] && img[y][x] <= b 
		// img_out[y][x] = (float)(d - c) / (b - a) * (img[y][x] - a) + c + 0.5; 
		// 문제점: 계산 순서로 인해 예상치 못한 결과가 나올 수 있음. 바로 변수 앞에 써줘야 함.

		output = ((float)d - c) / (b - a) * (input - a) + c + 0.5;
	}
	else {
		output = (255.0 - d) / (255.0 - b) * (input - b) + d + 0.5;
	}
}

int Ex0921_0() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	int a = 100, b = 150, c = 50, d = 200;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// img_out[y][x]=Stransform(img[y][x], a, b, c, d);
			// StransformPtr(img[y][x], a, b, c, d, &img_out[y][x]);
			StransformRef(img[y][x], a, b, c, d, img_out[y][x]);
		}
	}
	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//히스토그램 count
int Counting(int** img, int height, int width, int value) {
	int count = 0;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] == value) count++;
		}
	}
	return count;
}

//히스토그램 그리기
int Ex0921_1() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	int histogram[256] = { 0, };

	for (int value = 0; value < 256; value++) {
		histogram[value] = Counting(img, height, width, value);
	}

	DrawHistogram((char*)"output", histogram);

	return 0;
}

void GetHistogram(int** img, int height, int width, int* histogram) { // histogram[] 도 가능
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			histogram[img[y][x]]++;
		}
	}
}

int* GetHistogram2(int** img, int height, int width) {

	int hist[256] = { 0, };

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			hist[img[y][x]]++;
		}
	}

	return hist; // 주소 반환 -> 반환형 == int*, 해당 함수가 끝나는 순간 할당된 메모리가 삭제됨. 주소를 전달하더라도 해제되기 때문에 실행되지 않음
}

/*(참고용) 함수 내 선언된 배열이나 변수의 메모리는 그 함수를 벗어나는 순간 해제되어 사용 불가 */
int Ex0921_2() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	int histogram[256] = { 0, };

	// GetHistogram(img, height, width, histogram);
	int* hist = GetHistogram2(img, height, width);
	DrawHistogram((char*)"output", hist);

	return 0;
}

void Get_C_Histogram(int** img, int height, int width, int* chist) {
	int histogram[256] = { 0, };
	GetHistogram(img, height, width, histogram);

	chist[0] = histogram[0];
	for (int k = 1; k < 256; k++) { // 적분, 누적히스토그램 계산
		chist[k] = histogram[k] + chist[k - 1]; // 255 인덱스에는 영상의 전체 픽셀값이 있음
	}
}

void HistogramEqualization(int** img, int height, int width, int* chist, int** img_out) {
	Get_C_Histogram(img, height, width, chist);

# if 0
	int T[256] = { 0, };
	for (int k = 0; k < 256; k++) {
		T[k] = chist[k] * 255.0 / (512 * 512) + 0.5;
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = T[img[y][x]];
		}
	}

#else
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = chist[img[y][x]] * 255.0 / (512 * 512) + 0.5;
		}
	}
#endif
}

/*누적히스토그램을 평활화하는 데 사용, uniform한 분포를 갖도록 만들기 위함*/
//평활화 메인문
int EX0916() {
	int height, width;
	int** img = ReadImage((char*)"lenax0.5.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	int chist[256]{ 0, }; // 누적 히스토그램 배열
	Get_C_Histogram(img, height, width, chist);

	// chist[k]*255/(512*512); -> 커브 곡선
	// 그래프 형태는 유지하면서 255로 스케일링(높이를 최대 255로 설정)

	HistogramEqualization(img, height, width, chist, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);

	return 0;
}

//5주차 평균필터
void MeanFiltering3x3(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1)
				img_out[y][x] = img[y][x];
			else {
				img_out[y][x] = (int)(img[y - 1][x - 1] + img[y - 1][x] + img[y - 1][x + 1]
					+ img[y][x - 1] + img[y][x] + img[y][x + 1]
					+ img[y + 1][x - 1] + img[y + 1][x] + img[y + 1][x + 1]) / 9.0 + 0.5; //반올림처리 0.5
			}
		}
	}
}

void MeanFiltering5x5(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == 1 || x == width - 1 || x == width - 2 || y == 0 || y == 1 || y == height - 1 || y == height - 2)
				img_out[y][x] = img[y][x];
			else {
				int sum = 0;
				for (int y0 = -2; y0 <= 2; y0++) {
					for (int x0 = -2; x0 <= 2; x0++) {
						sum += img[y + y0][x + x0];
					}
				}
				img_out[y][x] = (int)(sum / 25.0 + 0.5);
			}
		}
	}
}

void MeanFiltering7x7(int** img, int height, int width, int** img_out) {
	int N = 7;
	int delta = (N - 1) / 2;

	for (int y = 0; y < height - 3; y++) {
		for (int x = 0; x < width - 3; x++) {
			//if (x == 0 || x == 1 || x == width - 1 || x == width - 2 || y == 0 || y == 1 || y == height - 1 || y == height - 2)
			//	img_out[y][x] = img[y][x];
			if (x < delta || x >= width - 2 || y < delta || y >= width - 2)
				img_out[y][x] = img[y][x];
			else {
				int sum = 0;
				for (int y0 = -delta; y0 <= delta; y0++) {
					for (int x0 = -delta; x0 <= delta; x0++) {
						sum += img[y + y0][x + x0];
					}
				}
				img_out[y][x] = (int)(sum / 49.0 + 0.5);
			}
		}
	}
}

void MeanFilteringNxN(int N, int** img, int height, int width, int** img_out) {
	int delta = (N - 1) / 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x < delta || x >= width - delta || y < delta || y >= width - delta)
				img_out[y][x] = img[y][x];
			else {
				int sum = 0;
				for (int y0 = -delta; y0 <= delta; y0++) {
					for (int x0 = -delta; x0 <= delta; x0++) {
						sum += img[y + y0][x + x0];
					}
				}
				img_out[y][x] = (int)(sum / (double)(N * N) + 0.5); //반올림
			}
		}
	}
}


//평균필터 메인문
int EX0926() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_out3 = (int**)IntAlloc2(height, width);

	//MeanFiltering3x3(img, height, width, img_out);
	MeanFilteringNxN(3, img, height, width, img_out);
	//MeanFiltering5x5(img, height, width, img_out2);
	MeanFilteringNxN(5, img, height, width, img_out2);
	//MeanFiltering7x7(img, height, width, img_out3);
	MeanFilteringNxN(7, img, height, width, img_out3);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
	ImageShow((char*)"output2", img_out2, height, width);
	ImageShow((char*)"output3", img_out3, height, width);
	return 0;
}

//6주차 마스킹과콘볼루션
int Masking(int y, int x, float** mask, int** img) {
	float sum = 0.0;
	for (int y0 = -1; y0 <= 1; y0++) {
		for (int x0 = -1; x0 <= 1; x0++) {
			sum += mask[y0 + 1][x0 + 1] * img[y + y0][x + x0];

			/*
			3*3 마스크
			mask[0][0] mask[0][1] ... mask[2][2]
			*/
		}
	}

	return (int)(sum + 0.5); //반올림을 통한 부정확성 보완
}

/*
	mask[0][0] * img[y - 1][x - 1] +
	mask[0][1] * img[y - 1][x] +
	mask[0][2] * img[y - 1][x + 1] +
	mask[1][0] * img[y][x - 1] +
	mask[1][1] * img[y][x] +
	mask[1][2] * img[y][x + 1] +
	mask[2][0] * img[y + 1][x - 1] +
	mask[2][1] * img[y + 1][x] +
	mask[2][2] * img[y + 1][x + 1];
*/

void Masking3x3(float** mask, int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == width - 1 || y == 0 || y == width - 1)
				img_out[y][x] = img[y][x]; //가장자리 처리
			else
				img_out[y][x] = Masking(y, x, mask, img);
		}
	}
}

int Masking_TWO(int y, int x, float** mask, int** img, int height, int width) {
	float sum = 0.0;
	for (int y0 = -1; y0 < 1; y0++) {
		for (int x0 = -1; x0 < 1; x0++) {
			//int y_new = (y + y0 < 0) ? 0 : y + y0;
			//int x_new = (x + x0 < 0) ? 0 : x + x0;

			//height-1 을 초과하면 height로 바꾼다.
			//y_new = (y_new >= height) ? height - 1 : y_new;
			//x_new = (x_new >= width) ? width - 1 : x_new;

			int y_new = imax(0, y + y0); //가장자리 처리 코드 없이 마스크 생성 
			int x_new = imax(0, x + x0);

			sum += mask[y0 + 1][x0 + 1] * img[y_new][x_new]; //x+x0에서 음수가 생기는 문제 발생 이를 해결하기 위해 new_y
		}
	}

	return (int)(sum + 0.5);
}

void Masking3x3_TWO(float** mask, int** img, int** img_out, int height, int width) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = Masking_TWO(y, x, mask, img, height, width);
		}
	}
}

void EX1012() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];
	float** mask = (float**)FloatAlloc2(3, 3);

	//마스크 계수 설정
	mask[0][0] = 1 / 9.0;	mask[0][1] = 1 / 9.0;	mask[0][2] = 1 / 9.0;
	mask[1][0] = 1 / 9.0;	mask[1][1] = 1 / 9.0;	mask[1][2] = 1 / 9.0;
	mask[2][0] = 1 / 9.0;	mask[2][1] = 1 / 9.0;	mask[2][2] = 1 / 9.0;

	Masking3x3(mask, img, img_out, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
}

#define FXFY 0 
#define FX 1
#define FY 2

// 7주차 에지 검출
void DetectEdgeByDerivative(
	int mode,
	int height, int width, int** img, int** img_out) {
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int fx = img[y][x + 1] - img[y][x];
			int fy = img[y + 1][x] - img[y][x];

			if (mode == FXFY)
				img_out[y][x] = imin(abs(fx) + abs(fy), 255);
			else if (mode == FX)
				img_out[y][x] = abs(fx);
			else if (mode == FY)
				img_out[y][x] = abs(fy);
			else {
				printf("\n Mode Error !!!");
				return;
			}
		}
	}
}

int EX1016() {
	int height, width;
	int** img = ReadImage((char*)"barbara.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	DetectEdgeByDerivative(FXFY, height, width, img, img_out);
	ImageShow((char*)"output0", img_out, height, width);

	DetectEdgeByDerivative(FX, height, width, img, img_out);
	ImageShow((char*)"output1", img_out, height, width);

	DetectEdgeByDerivative(FY, height, width, img, img_out);
	ImageShow((char*)"output2", img_out, height, width);

	return 0;
}

int FindMaxValue(int** img, int height, int width) {
	int maxvalue = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			maxvalue = imax(maxvalue, img[y][x]);
		}
	}

	return maxvalue;
}

int FindMinValue(int** img, int height, int width) {
	int minvalue = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			minvalue = imin(minvalue, img[y][x]);
		}
	}

	return minvalue;
}

int EX1016_1() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);

	int maxvalue = FindMaxValue(img, height, width);
	int minvalue = FindMinValue(img, height, width);

	printf("\n maxvalue = %d, minvalue = %d", maxvalue, minvalue);
	return 0;
}

void NormalizeImageByMaxvalue(int** img_out, int height, int width) {
	int maxvalue = FindMaxValue(img_out, height, width); //그리고자 하는 영상은 img_out
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = (float)img_out[y][x] / maxvalue * 255;
		}
	}
}

int EX1016_2() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	DetectEdgeByDerivative(FXFY, height, width, img, img_out);
	NormalizeImageByMaxvalue(img_out, height, width);
	ImageShow((char*)"output0", img_out, height, width);

	DetectEdgeByDerivative(FX, height, width, img, img_out);
	NormalizeImageByMaxvalue(img_out, height, width);
	ImageShow((char*)"output1", img_out, height, width);

	DetectEdgeByDerivative(FY, height, width, img, img_out);
	NormalizeImageByMaxvalue(img_out, height, width);
	ImageShow((char*)"output2", img_out, height, width);

	return 0;
}

void AbsImage(int** img_out, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = abs(img_out[y][x]);
		}
	}
}

int EX1019() {
	int height, width;
	int** img = ReadImage((char*)"lenaGN10.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	float** mask0 = (float**)FloatAlloc2(3, 3);
	mask0[0][0] = 0;  mask0[0][1] = -1; mask0[0][2] = 0;
	mask0[1][0] = -1; mask0[1][1] = 4;  mask0[1][2] = -1;
	mask0[2][0] = 0;  mask0[2][1] = -1; mask0[2][2] = 0;

	float** mask1 = (float**)FloatAlloc2(3, 3);
	mask1[0][0] = -1;  mask1[0][1] = -1; mask1[0][2] = -1;
	mask1[1][0] = -1; mask1[1][1] = 8;  mask1[1][2] = -1;
	mask1[2][0] = -1;  mask1[2][1] = -1; mask1[2][2] = -1;

	float** mask2 = (float**)FloatAlloc2(3, 3);
	mask2[0][0] = 1 / 9.0;  mask2[0][1] = 1 / 9.0; mask2[0][2] = 1 / 9.0;
	mask2[1][0] = 1 / 9.0;  mask2[1][1] = 1 / 9.0; mask2[1][2] = 1 / 9.0;
	mask2[2][0] = 1 / 9.0;  mask2[2][1] = 1 / 9.0; mask2[2][2] = 1 / 9.0;

	Masking3x3(mask0, img, img_out, height, width);
	AbsImage(img_out, height, width);
	NormalizeImageByMaxvalue(img_out, height, width);

	ImageShow((char*)"output0", img_out, height, width);

	Masking3x3(mask1, img, img_out, height, width);
	AbsImage(img_out, height, width);
	NormalizeImageByMaxvalue(img_out, height, width);

	ImageShow((char*)"output1", img_out, height, width);

	Masking3x3(mask2, img, img_out, height, width);
	//AbsImage(img_out, height, width);
	//NormalizeImageByMaxvalue(img_out, height, width);

	ImageShow((char*)"output2", img_out, height, width);

	return 0;
}

//고대역통과 필터의 특성! : 급격히 변하는 것이 경계선(?)
//잡음이 심함-> 고대역필터 통과 시키면 거의 안 나옴.
//잡음은 랜덤으로 값이 변하기 때문에, 이 값의 변화는 고주파에 해당함
//**고대역통과 필터는 잡음이 있는 영상에 적용하면 효과가 높다., 잡음을 증폭시키는 역할
//**저대역통과 필터는? -> 잡음이 줄어드는 효과가 있음.

int EX1019_2() {
	int height, width;
	int** img = ReadImage((char*)"lenaGN10.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	//수평선
	float** mask0 = (float**)FloatAlloc2(3, 3);
	mask0[0][0] = -1;  mask0[0][1] = -2; mask0[0][2] = -1;
	mask0[1][0] = -1; mask0[1][1] = 4;  mask0[1][2] = -1;
	mask0[2][0] = 1;  mask0[2][1] = 2; mask0[2][2] = 1;

	//수직선
	float** mask1 = (float**)FloatAlloc2(3, 3);
	mask1[0][0] = -1;  mask1[0][1] = 0;  mask1[0][2] = 1;
	mask1[1][0] = -2;  mask1[1][1] = 0;  mask1[1][2] = 2;
	mask1[2][0] = -1;  mask1[2][1] = 0;  mask1[2][2] = 1;

	Masking3x3(mask0, img, img_out, height, width); //y방향으로 미분
	AbsImage(img_out, height, width); //y방향 미분의 절댓값
	NormalizeImageByMaxvalue(img_out, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output0", img_out, height, width);

	Masking3x3(mask1, img, img_out, height, width); //x방향으로 미분
	AbsImage(img_out, height, width); //x방향으로 미분의 절댓값
	NormalizeImageByMaxvalue(img_out, height, width);

	//ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output1", img_out, height, width);

	return 0;
}

void SobelEdgeDetct(int** img, int height, int width,
	int** img_out0, int** img_out1, int** img_out2)
{
	//수평선
	float** mask0 = (float**)FloatAlloc2(3, 3);
	mask0[0][0] = -1;  mask0[0][1] = -2; mask0[0][2] = -1;
	mask0[1][0] = 0; mask0[1][1] = 0;  mask0[1][2] = 0;
	mask0[2][0] = 1;  mask0[2][1] = 2; mask0[2][2] = 1;

	//수직선
	float** mask1 = (float**)FloatAlloc2(3, 3);
	mask1[0][0] = -1;  mask1[0][1] = 0;  mask1[0][2] = 1;
	mask1[1][0] = -2;  mask1[1][1] = 0;  mask1[1][2] = 2;
	mask1[2][0] = -1;  mask1[2][1] = 0;  mask1[2][2] = 1;

	Masking3x3(mask0, img, img_out0, height, width); //y방향으로 미분
	AbsImage(img_out0, height, width); //y방향 미분의 절댓값

	Masking3x3(mask1, img, img_out1, height, width); //x방향으로 미분
	AbsImage(img_out1, height, width); //x방향으로 미분의 절댓값

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out2[y][x] = img_out0[y][x] + img_out1[y][x];
		}
	}
}

void EX1019_3() {
	int height, width;
	int** img = ReadImage((char*)"lenaGN10.png", &height, &width);
	int** img_out0 = (int**)IntAlloc2(height, width); // int img_out[512]][512];
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	SobelEdgeDetct(img, height, width, img_out0, img_out1, img_out2);

	NormalizeImageByMaxvalue(img_out0, height, width);
	NormalizeImageByMaxvalue(img_out1, height, width);
	NormalizeImageByMaxvalue(img_out2, height, width);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output0", img_out0, height, width);
	ImageShow((char*)"output1", img_out1, height, width);
	ImageShow((char*)"output2", img_out2, height, width);
}


//255 넘으면 255로 설정, 255 이하면 그대로 설정
void ClippingImage(int** img_in, int height, int width, int** img_out2) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out2[y][x] = imin(255, imax(img_in[y][x], 0));
		}
	}
}

void EdgeEnhanceImage(float alpha, int** img, int height, int width, int** img_out0, int** img_out1) {
	float** mask = (float**)FloatAlloc2(3, 3);

	mask[0][0] = -alpha;	mask[0][1] = -alpha;		 mask[0][2] = -alpha;
	mask[1][0] = -alpha;	mask[1][1] = 1 + 8 * alpha;  mask[1][2] = -alpha;
	mask[2][0] = -alpha;	mask[2][1] = -alpha;		 mask[2][2] = -alpha;

	Masking3x3(mask, img, img_out0, height, width);
	ClippingImage(img_out0, height, width, img_out1);
}

void EdgeEnhanceImage_TWO(float alpha, int** img, int height, int width, int** img_out) {
	float** mask = (float**)FloatAlloc2(3, 3);

	mask[0][0] = -alpha;	mask[0][1] = -alpha;		 mask[0][2] = -alpha;
	mask[1][0] = -alpha;	mask[1][1] = 1 + 8 * alpha;  mask[1][2] = -alpha;
	mask[2][0] = -alpha;	mask[2][1] = -alpha;		 mask[2][2] = -alpha;

	Masking3x3(mask, img, img_out, height, width);
	ClippingImage(img_out, height, width, img_out);
}

//스펙트럼이 고주파까지 뻗어있으면 더 또렷하게 보임
void EX1023() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];
	int** img_out1 = (int**)IntAlloc2(height, width); // int img_out[512]][512];

	float alpha = 0.1;

	ImageShow((char*)"input", img, height, width);

	for (alpha = 0.1; alpha < 1.0; alpha += 0.1) {
		EdgeEnhanceImage_TWO(alpha, img, height, width, img_out);
		ImageShow((char*)"output", img_out, height, width);
	}


}

void Swap(int* A, int* B) {
	int tmp;

	tmp = *A;
	*A = *B;
	*B = tmp;
}

//8주차 선명화 처리 및 중간값 필터
void Bubbling(int* data, int num)
{
	for (int i = 0; i < num - 1; i++) {
		if (data[i] > data[i + 1]) Swap(&data[i], &data[i + 1]);
	}
}

void BubbleSorting(int* data, int num) {
	for (int i = num; i > 0; i--) {
		Bubbling(data, i);
	}
}


void EX1030_0() {
#define N 9
	//중앙값, swap 함수로 정렬
	int data[N] = { 7, 1, 3, 5, 2, 7, 1, 3, 5 };

	BubbleSorting(data, N);
	printf("\n 중앙값 = %d", data[(N - 1) / 2]);
}

void GetBlock3x3(int y, int x, int** img, int* data)
{
	int count = 0;

	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			data[count] = img[y + i][x + j];
			count++;
		}
	}

}

void MedianFiltering(int** img, int height, int width, int** img_out) {
	int data[9];

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			GetBlock3x3(y, x, img, data);
			BubbleSorting(data, 9);
			img_out[y][x] = data[4];
		}
	}

	int x = 0;
	for (int y = 0; y < height; y++) //세로선
		img_out[y][x] = img[y][x];

	x = width - 1;
	for (int y = 0; y < height; y++) //세로선
		img_out[y][x] = img[y][x];

	int y = 0;
	for (int x = 0; x < width; x++) //가로선
		img_out[y][x] = img[y][x];

	y = height - 1;
	for (int x = 0; x < width; x++) //가로선
		img_out[y][x] = img[y][x];
}

void EX1030() {
	int height, width;
	int** img = ReadImage((char*)"lenaSP10.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width); // int img_out[512]][512];
	int** img_out2 = (int**)IntAlloc2(height, width);

	MeanFiltering3x3(img, height, width, img_out);
	MeanFiltering3x3(img_out, height, width, img_out2);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, height, width);
	ImageShow((char*)"output2", img_out2, height, width);
}

// 4개도 가능하다.
void ZeroOrderInterpolation(int** img, int height, int width, int** img_out) {

	//zero-order interpolation(0차 보간)
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[2 * y][2 * x] = img[y][x];
			img_out[2 * y][2 * x + 1] = img[y][x];
			img_out[2 * y + 1][2 * x] = img[y][x];
			img_out[2 * y + 1][2 * x + 1] = img[y][x];
		}
	}
}

void Bi_LinearInterpolation(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height - 1; y++) {
		for (int x = 0; x < width - 1; x++) {
			int A = img[y][x];
			int B = img[y][x + 1];
			int C = img[y + 1][x];
			int D = img[y + 1][x + 1];

			img_out[2 * y][2 * x] = A;
			img_out[2 * y][2 * x + 1] = (A + B) / 2;
			img_out[2 * y + 1][2 * x] = (A + C) / 2;
			img_out[2 * y + 1][2 * x + 1] = (A + B + C + D) / 4;
		}
	}
}

int Bi_LinearInterpolation_1pixel(float dy, float dx,
	int A, int B, int C, int D) {
	int output = (1 - dx) * (1 - dy) * A + dx * (1 - dy) * B
		+ (1 - dx) * dy * C + dx * dy * D;
	return output;
}

int EX1102() {
	int height, width;
	int** img = ReadImage((char*)"s_lena.png", &height, &width);

	int heightx2 = 2 * height, widthx2 = 2 * width;
	int** img_out = (int**)IntAlloc2(heightx2, heightx2); // int img_out[512]][512];
	int** img_out2 = (int**)IntAlloc2(heightx2, heightx2);

	ZeroOrderInterpolation(img, height, width, img_out);
	Bi_LinearInterpolation(img, height, width, img_out2);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"output", img_out, heightx2, heightx2);
	ImageShow((char*)"output2", img_out2, heightx2, heightx2);


	return 0;
}

int BilinearInterpolation(
	double y, double x, //ex) y=100.2, x = 150.3
	int** img, int height, int width)
{
	//A 좌표
	int y_int = (int)y;
	int x_int = (int)x;
	double dy = y - y_int;
	double dx = x - x_int;

	int A = img[y_int][x_int];
	int B = img[y_int][x_int + 1];
	int C = img[y_int + 1][x_int];
	int D = img[y_int + 1][x_int + 1];

	return Bi_LinearInterpolation_1pixel((float)dy, (float)dx, A, B, C, D);
}

int EX102_2(double dy, double dx, int** img, int height, int width) {
	float y0 = 100.2;
	float x0 = 150.3;
	//A = (y=100, x=150)
	int y = (int)y0; //소숫점 삭제
	int x = (int)x0;

	int A = img[y][x];
	int B = img[y][x + 1];
	int C = img[y + 1][x];
	int D = img[y + 1][x + 1];

	return Bi_LinearInterpolation_1pixel(dy, dx, A, B, C, D);
}

//img의 중심을 좌표로 하는 코드
void Translation(int ty, int tx,
	int** img, int height, int width, int** img_out)
{
	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			int y = yp - ty;
			int x = xp - tx;
			if (y<0 || y> height - 1 || x<0 || x>width - 1)
				img[yp][xp] = 0;
			else
				img[yp][xp] = img[y][x];
		}
	}
}

//img의 중심을 좌표로 하는 코드
void TranslationF(float ty, float tx,
	int** img, int height, int width, int** img_out)
{
	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			float y = yp - ty;
			float x = xp - tx;
			if (yp<0 || y> height - 1 || xp<0 || xp>width - 1)
				img[yp][xp] = 0;
			else
				img[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}

void Rotation(float theta, int y0, int x0,
	int** img, int height, int width, int** img_out)
{
	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			//주요코드
			float x = (xp - x0) * cos(theta) + (yp - y0) * sin(theta) + x0;
			float y = -(xp - x0) * sin(theta) + (yp - y0) * cos(theta) + y0;

			if (y < 0 || y >= height - 1 || x <= 0 || x >= width - 1)
				img_out[yp][xp] = 0;
			else
				img_out[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}

//영상이동 코드
//img_out의 좌표를 중심으로
void Translation_TWO(int ty, int tx,
	int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int yp = y + ty;
			int xp = x + tx;
			if (yp<0 || y> height - 1 || xp<0 || xp>width - 1)
				continue;
			else
				img[yp][xp] = img[y][x];
		}
	}
}

void EX1106() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	int tx = -100.7, ty = -50.5;

	TranslationF(ty, tx, img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
}

//1. 사진의 중앙을(0, 0)으로 이동
//사진의 왼쪽 좌표(-256, -256)
//[x'] [cos -sin] [x-256]
//[y'] [sin  cos] [x-256]
//
//x' = (x-256)cos-ysin
//y' = (x-256)sin+ycos
//
//2. 회전시키기
//다시 원위치로
//[x'-256] = [cos -sin] [x- 256] + [256]
//[y'-256]   [sin  cos] [y- 256] + [256]
//
//3. x' y'을 중심으로
//[x - 256] = [cos  sin][x'-256]
//[y - 256] = [-sin cos][y'-256]
//
//[x] = [cos  sin][x'-256] + [256]
//[y] = [-sin cos][y'-256] + [256]

#define PI 3.14
void EX1106_2() {
	int height, width;
	int** img = ReadImage((char*)"lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float theta = 15.0 * (PI / 180.0); //라디안 바꾸기
	int y0 = 256, x0 = 256;
	Rotation(theta, y0, x0, img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"img_out", img_out, height, width);
}

//사진의 중앙을 (0,0)으로 맞추고 싶다면?

//어파인변환
/*
x = a'*(x'-tx) + b;*(y'-ty)
y = a'*(x'-tx) + b'*(y'-ty)
*/
struct AFFINE {
	float a, b, c, d, tx, ty;
};

void AffineTransform(AFFINE para,
	int** img, int height, int width, int** img_out)
{
	float D = para.a * para.d - para.b * para.c;
	float ap = para.d / D;
	float bp = -para.b / D;
	float cp = -para.c / D;
	float dp = para.a / D;

	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			//주요코드
			float x = ap * (xp - para.tx) + bp * (yp - para.ty);
			float y = cp * (xp - para.tx) + dp * (yp - para.ty);

			if (y < 0 || y >= height - 1 || x <= 0 || x >= width - 1)
				img_out[yp][xp] = 0;
			else
				img_out[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}


void main()
{
	int height, width;
	int** img = ReadImage((char*)"s_barbara_4affine.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	float theta = 15.0 * (PI / 180.0); //라디안 바꾸기

	AFFINE para;
	//para.a = cos(theta);	para.b = -sin(theta);
	//para.c = sin(theta);	para.d = cos(theta);
	//para.tx = 0.0;			para.ty = 0.0;

	//원점을 중심으로 두 배
	para.a = 2;		para.b = 0;
	para.c = 0;		para.d = 2;
	para.tx = 0.0;	para.ty = 0.0;


	para.a = 2;		para.b = 0;
	para.c = 0;		para.d = 2;
	para.tx = 0.0;	para.ty = 0.0;

	AffineTransform(para, img, height, width, img_out);

	ImageShow((char*)"input", img, height, width);
	ImageShow((char*)"img_out", img_out, height, width);
}

/*
-------------------------------------------------------------------------------
2020 중간고사
*/

void MakeCheckBoard(int ny, int nx, int** img_out, int height, int width) {
	// Implement checkerboard pattern logic here
	for (int i = 0; i < height; ++i) {
		for (int j = 0; j < width; ++j) {
			int color = ((i / (height / ny)) + (j / (width / nx))) % 2 == 0 ? 255 : 0;
			img_out[i][j] = color;
		}
	}
}

void Prob1_2020()
{
	int height = 512, width = 512;
	int** img_out = (int**)IntAlloc2(height, width);
	int ny = 11, nx = 15;

	MakeCheckBoard(ny, nx, img_out, height, width);
	ImageShow((char*)"출력영상", img_out, height, width);
}


/*
--------------------------------------------------------------------------------------------------------
2022년 중간고사
*/
void sStretching(int** img, int height, int width, int** img_out) { // Prob1
	int max_value = img[0][0];			//Imax
	int min_value = img[0][0];			//Imin
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			max_value = imax(max_value, img[y][x]);
			min_value = imin(min_value, img[y][x]);
		}
	}

	//픽셀값 정규화
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] < min_value)
				img_out[y][x] = 0;
			else if (img[y][x] < max_value)
				img_out[y][x] = 255.0 / (max_value - min_value) * (img[y][x] - min_value);
			else
				img_out[y][x] = 0;
		}
	}
}

void Prob1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"test1.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	sStretching(img, height, width, img_out);

	ImageShow((char*)"영상보기", img, height, width);
	ImageShow((char*)"출력영상", img_out, height, width);
}

void DetectBBox(int** img, int height, int width, int** img_out) { // Prob2
	int Xmax = 0, Xmin = width - 1, Ymax = 0, Ymin = height - 1;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] == 255) {
				Xmax = imax(Xmax, x);
				Xmin = imin(Xmin, x);
				Ymax = imax(Ymax, y);
				Ymin = imin(Ymin, y);
			}
		}
	}
	printf("%d %d %d %d", Xmax, Xmin, Ymax, Ymin);
	//좌상단 : 흰부분 중 x값 가장작고, y값 가장작은곳 (Xmin, Ymin)
	//우하단 : 흰부분 중 x값 가장크고, y값 가장큰곳	   (Xmax, Ymax)
	for (int y = Ymin; y <= Ymax; y++) {
		for (int x = Xmin; x <= Xmax; x++) {
			if ((x == Xmin || x == Xmax) || (y == Ymin || y == Ymax)) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
	}
}

void Prob2()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"test2.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	DetectBBox(img, height, width, img_out);

	ImageShow((char*)"영상보기", img, height, width);
	ImageShow((char*)"출력영상", img_out, height, width);
}


void DetectBBox2(int** img, int height, int width, int** img_out) { // Prob3_1
	int Xmax1 = 0, Xmin1 = width - 1, Xmax2 = 0, Xmin2 = width - 1, Xmax3 = 0, Xmin3 = width - 1, Xmax4 = 0, Xmin4 = width - 1, Xmax5 = 0, Xmin5 = width - 1, Ymax = 0, Ymin = height - 1;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] == 255) { //흰색이면
				Ymax = imax(Ymax, y); Ymin = imin(Ymin, y);
				if (x <= 0.2 * width) { Xmax1 = imax(Xmax1, x); Xmin1 = imin(Xmin1, x); }
				else if (x <= 0.4 * width) { Xmax2 = imax(Xmax2, x); Xmin2 = imin(Xmin2, x); }
				else if (x <= 0.6 * width) { Xmax3 = imax(Xmax3, x); Xmin3 = imin(Xmin3, x); }
				else if (x <= 0.8 * width) { Xmax4 = imax(Xmax4, x); Xmin4 = imin(Xmin4, x); }
				else { Xmax5 = imax(Xmax5, x); Xmin5 = imin(Xmin5, x); }
			}
		}
	}

	for (int y = Ymin; y <= Ymax; y++) {
		for (int x = Xmin1; x <= Xmax1; x++) {
			if (x == Xmin1 || x == Xmax1 || y == Ymin || y == Ymax) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
		for (int x = Xmin2; x <= Xmax2; x++) {
			if (x == Xmin2 || x == Xmax2 || y == Ymin || y == Ymax) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
		for (int x = Xmin3; x <= Xmax3; x++) {
			if (x == Xmin3 || x == Xmax3 || y == Ymin || y == Ymax) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
		for (int x = Xmin4; x <= Xmax4; x++) {
			if (x == Xmin4 || x == Xmax4 || y == Ymin || y == Ymax) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
		for (int x = Xmin5; x <= Xmax5; x++) {
			if (x == Xmin5 || x == Xmax5 || y == Ymin || y == Ymax) img_out[y][x] = 255;
			else img_out[y][x] = img[y][x];
		}
	}
}


void Prob3_1()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"test3.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	DetectBBox2(img, height, width, img_out);

	ImageShow((char*)"영상보기", img, height, width);
	ImageShow((char*)"출력영상", img_out, height, width);
}

void Prob3_2()
{

}

void Avg3x3_test(int** img, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == (width - 1) || y == 0 || y == (height - 1))
				img_out[y][x] = img[y][x];	// 가장자리 입력값 복사

			else {
				//img_out[y][x] = img[y][x];  // 4번문제 그림의 P값 처음에 집어넣어줌
				for (int i = -1; i <= 1; i++) {
					for (int j = -1; j <= 1; j++) {
						img_out[y][x] = imax(img_out[y][x], img[y + i][x + j]);
					}
				}
			}
		}
	}
}

void Prob4()
{
	int height, width;
	int** img = (int**)ReadImage((char*)"test4.bmp", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);

	Avg3x3_test(img, height, width, img_out);

	ImageShow((char*)"영상보기", img, height, width);
	ImageShow((char*)"출력영상", img_out, height, width);
}

/*
-------------------------------------------------------------------------------
중간고사
*/

void DrawCircle(int r, int y0, int x0, int value, int** img, int height, int width)
{
	//y0, x0 : 원의 중심좌표, r : 반경, value : 원의 밝기값
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if ((double)(x - x0) * (x - x0) + (double)(y - y0) * (y - y0) <= r * r)
				img[y][x] = value;
		}
	}
}

void Probb1() {
	int height = 512, width = 512;
	int** img_out = (int**)IntAlloc2(height, width);

	DrawCircle(200, 256, 256, 180, img_out, height, width);

	ImageShow((char*)"output", img_out, height, width);
}

void DrawCircle3(int r, int** img, int height, int width) {
	int	x1 = 175, y1 = 220, x2 = 325, y2 = 200, x3 = 325, y3 = 350;
	int value = 100;

	for (int x = 0; x < height; x++) {
		for (int y = 0; y < width; y++) {
			if ((double)(x - x1) * (x - x1) + (double)(y - y2) * (y - y2) <= r * r) {
				img[y][x] = imin(img[y][x] + value, 255); //스트레칭
			}
			if ((double)(x - x2) * (x - x2) + (double)(y - y2) * (y - y2) <= r * r) {
				img[y][x] = imin(img[y][x] + value, 255); //스트레칭
			}
			if ((double)(x - x3) * (x - x3) + (double)(y - y3) * (y - y3) <= r * r) {
				img[y][x] = imin(img[y][x] + value, 255); //스트레칭
			}
		}
	}
}

void Probb2() {
	int height = 512, width = 512;
	int** img_out = (int**)IntAlloc2(height, width);

	DrawCircle3(150, img_out, height, width);

	ImageShow((char*)"output", img_out, height, width);
}


struct KernelSize {
	int x, y;
};

void MaxFiltering(KernelSize size, int** img, int height, int width, int** img_out)
{
	int* block = (int*)malloc(size.y * size.x * sizeof(int)); // 1차원 배열 메모리 할당
	int delta = (size.x - 1) / 2;


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) {
				img[y][x] = img[y][x];
				continue;
			}
			int maxValue = 0, k = 0;

			for (int y0 = -delta; y0 <= delta; y0++) {
				for (int x0 = -delta; x0 <= delta; x0++) {
					block[k] = img[y + y0][x + x0];
					maxValue = imax(maxValue, block[k]);
					k++;
				}
			}
			img_out[y][x] = maxValue;
		}
	}
	free(block); // 1차원 배열 메모리 할당
}

void DiffImage(int** A, int** B, int height, int width, int** diff_img) // diff_mg = A – B 
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			diff_img[y][x] = abs(A[y][x] - B[y][x]);
		}
	}
}

void Probb3() {
	int height, width;
	int** img = (int**)ReadImage((char*)"img1-1.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	KernelSize size;
	size.x = 3; size.y = 3;

	MaxFiltering(size, img, height, width, img_out);
	ImageShow((char*)"output", img_out, height, width);
	DiffImage(img, img_out, height, width, img_out2);
	ImageShow((char*)"output", img_out2, height, width);
}



#define PI 3.14
void PaddingImage(int** img, int height, int width, int** img_out, int height_out, int width_out) {
	
	int tx = (width_out - width) / 2;
	int ty = (height_out - height) / 2;

	//검정색으로 칠하기
	for (int y = 0; y < height_out; y++) {
		for (int x = 0; x < width_out; x++) {
			img_out[y][x] = 0;
		}
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int y_prime = y + ty;
			int x_prime = x + tx;

			img_out[y_prime][x_prime] = img[y][x];
		}
	}
}

void Rrotation(float theta, int** img, int height, int width, int** img_out) {
	double rad = theta / 180.0 * PI;
	int y0 = height / 2, x0 = width / 2;

	for (int yp = 0; yp < height; yp++) {
		for (int xp = 0; xp < width; xp++) {
			//주요코드
			float x = (xp - x0) * cos(theta) + (yp - y0) * sin(theta) + x0;
			float y = -(xp - x0) * sin(theta) + (yp - y0) * cos(theta) + y0;

			if (y < 0 || y >= height - 1 || x <= 0 || x >= width - 1)
				img_out[yp][xp] = 0;
			else
				img_out[yp][xp] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}

void MergeImage(float alpha, int** img, int** img_char, int height, int width, int** img_out) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = IMIN(IMAX((int)(img[y][x] + alpha * img_char[y][x] + 0.5), 0), 255);
		}
	}
}

void Probb4() {
	int height, width;
	int** img = (int**)ReadImage((char*)"lena.png", &height, &width);
	int h_char, w_char;
	int** img_char = ReadImage((char*)"num0123.png", &h_char, &w_char);
	int** img_char_pad = (int**)IntAlloc2(height, width);
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	// num0123.png 영상의 크기를 lena.png 크기와 동일하게 하기 위해 주변의 값을 0으로 채움
	PaddingImage(img_char, h_char, w_char, img_char_pad, height, width);
	// num0123.png 영상을 회전시킴
	Rrotation(45, img_char_pad, height, width, img_out);

	// 회전된 num0123.png 영상과 lena.png 영상을 lena + alpha*num0123 로 만듦 (clipping적용할 것)
	float alpha = 1.0;
	MergeImage(alpha, img, img_out, height, width, img_out2);
	ImageShow((char*)"output", img_out2, height, width);

	alpha = 0.3;
	MergeImage(alpha, img, img_out, height, width, img_out2);
	ImageShow((char*)"output", img_out2, height, width);
	
	ImageShow((char*)"output", img_char_pad, height, width);
	ImageShow((char*)"output", img_out, height, width);

}

void __main() {
	Prob1_2020();
	//Probb1();
	//Probb2();
	//Probb3();
	//Probb4();
}