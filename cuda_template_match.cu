//Author: Dongwei Shi
//Created: 06/15/2016
//Description: this program is for template matching with cuda. The program is expected to template match several template simutaneously

#include <stdio.h>
#include <iostream>
#include <math.h>
#include <vector>
#include <unistd.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include </usr/local/cuda-8.0/include/cuda.h>
#include </usr/local/cuda-8.0/include/cufft.h>
#include </usr/local/cuda-8.0/include/cufft.h>



#define KERNEL_WIDTH 31
#define KERNEL_RADIUS (KERNEL_WIDTH/2)
#define TILE_WIDTH (33-KERNEL_WIDTH)
#define BLK_SIZE (TILE_WIDTH+KERNEL_WIDTH-1)
#define TMP_NUM 8


#define ACCURATE_MODE KERNEL_WIDTH
#define SPEED_MODE 1
#define RECORD 0
#define CROP_PARAM 2.2
using namespace std;
using namespace cv;

//global image and templates
Mat img, img_cmp, gray_img, gray_img_cmp;
Mat templs[TMP_NUM];
Mat img_vec[TMP_NUM];
Point kpt_vec[TMP_NUM];
Point ext_vec[TMP_NUM];
vector<Point2f > corners;
int dis[TMP_NUM];

//deviceKernel for storing the templates 
__constant__ float deviceKernel[TMP_NUM*KERNEL_WIDTH*KERNEL_WIDTH];
///////////////////////////////////////////////////////////////////
/* conv2d
 *      Description: This funtion is CUDA kernel. Where perform the 2D convolution of the images and templates.
 *                   Using CV_TM_CCOEFF_NORMED method for template matching. Simutaneously perform 2D convolution
 *                   on several images with specific templates.
 *      Input: A -- the input data of images
 *             x_size -- the image width
 *             y_size -- the image height
 *             template_num -- the total templates need to be matched.
 *      Output: B -- the convolution results of the images.
 *      
 * 
*/
///////////////////////////////////////////////////////////////////
__global__ void conv2d(float* A, float* B, const int x_size, const int y_size, const int template_num)
{
   //allocated shared memory for storing the image
    __shared__ float Nds[BLK_SIZE][BLK_SIZE];
    int tx = threadIdx.x;
    int ty = threadIdx.y;


    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int x_out = bx*TILE_WIDTH + tx;
    int y_out = by*TILE_WIDTH + ty;
    
    int x_in = x_out - KERNEL_RADIUS;
    int y_in = y_out - KERNEL_RADIUS;
    float res = 0.0;
    float templ_res = 0.0;
    float img_res = 0.0;
    //copy the image to the shared memeory 
    if((x_in>=0) && (x_in<x_size) && (y_in>=0) && (y_in<y_size) && (bz>=0) && (bz<template_num) )
    {
        Nds[ty][tx] = A[bz*x_size*y_size + y_in*x_size + x_in];
    }
    else
    {
        Nds[ty][tx] = 0.0;
    }
    __syncthreads();

    //perform convolution below using CV_TM_CCOEFF_NORMED method for template matching
    if( (tx<TILE_WIDTH) && (ty<TILE_WIDTH) && (x_out<x_size) && (y_out<y_size) && (bz>=0) && (bz<template_num))
    {
            res = 0.0;
            templ_res = 0.0;
            img_res = 0.0;
            for( int idx_y=0; idx_y<KERNEL_WIDTH; idx_y++ )
            {
                for( int idx_x=0; idx_x<SPEED_MODE; idx_x++ )
                {
                    
                    templ_res += pow(deviceKernel[bz*KERNEL_WIDTH*KERNEL_WIDTH+idx_y*KERNEL_WIDTH+idx_x],2);
                    img_res += pow(Nds[ty+idx_y][tx+idx_x],2);
                    res += Nds[ty+idx_y][tx+idx_x] * deviceKernel[bz*KERNEL_WIDTH*KERNEL_WIDTH+idx_y*KERNEL_WIDTH+idx_x];
                    
    
                }
            }
            //copy the result into the output data
            __syncthreads();
            if((x_out<x_size) && (y_out<y_size) && (bz<template_num))
            {
                B[bz*x_size*y_size + y_out*x_size + x_out] = res/sqrt(templ_res*img_res);
            }
            __syncthreads();
        
    }
   
}
///////////////////////////////////////////////////////////////////
/* cuda_tp_img
 *      Description: This function use for preparation step for the 
 *                   cuda kernel. It is allocate several memory space
 *                   on both GPU and CPU. It also be used to select the
 *                   peak value of the convolution results  
 *      Input: templates number -- the total number of templates that need to
 *                                 be matched.
 *      Output: 0 -- success, -1 -- failure
 *      
 * 
*/
///////////////////////////////////////////////////////////////////

int cuda_tp_img(int template_num)
{
   
    //get size of templates and images.
    int x_size = img_vec[0].cols;
    int y_size = img_vec[0].rows;
    int tmp_x_size = KERNEL_WIDTH;//templs[0].cols;
    int tmp_y_size = KERNEL_WIDTH;//templs[0].rows;
    int img_size = x_size * y_size;
    int tmpl_size = tmp_x_size * tmp_y_size;
    
    //allocate a space to store the image intensity
    float* host_img = (float*) malloc(sizeof(float)*img_size*template_num);
    float* host_templ = (float*) malloc(sizeof(float)*tmpl_size*template_num);
    float* gpu_out = (float*) malloc(sizeof(float)*img_size*template_num);

    float* device_img_input;
    float* device_img_output;
  
    //copy the intensity value from image
    for(int img_idx=0; img_idx<template_num; img_idx++)
    {
        for(int y=0; y<y_size; y++)
        {
            for(int x=0; x<x_size; x++)
            {
                Scalar intensity = img_vec[img_idx].at<uchar>(y,x);
                host_img[y*x_size+x + img_idx*img_size] = intensity.val[0];
            }   
         } 
         
    }
    //copy the intensity value from templates
    for(int tmpl_idx=0; tmpl_idx<template_num; tmpl_idx++)
    {
        for(int y=0; y<tmp_y_size; y++)
        {
            for(int x=0; x<tmp_x_size; x++)
            {
                Scalar intensity = templs[tmpl_idx].at<uchar>(y,x);
                host_templ[y*tmp_x_size+x+tmpl_idx*tmpl_size] = intensity.val[0];
            }        
        }
    }
    //allocate memory in cuda global memory
    cudaMalloc( (void**)&device_img_input, img_size*sizeof(float)*template_num  );
    cudaMalloc( (void**)&device_img_output, img_size*sizeof(float)*template_num );

    cudaMemcpy( device_img_input, host_img, img_size*sizeof(float)*template_num, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol( deviceKernel, host_templ, tmpl_size*sizeof(float)*template_num);

    //assign blocks and threads
    dim3 Dimblock(BLK_SIZE, BLK_SIZE, 1);
    dim3 DimGrid(((TILE_WIDTH+x_size)-1/TILE_WIDTH), ((TILE_WIDTH+y_size)-1/TILE_WIDTH),template_num);
    //calling the convolution gpu function
    conv2d <<< DimGrid, Dimblock >>>( device_img_input, device_img_output, x_size, y_size, template_num);
    cudaDeviceSynchronize();
    
    cudaMemcpy( gpu_out, device_img_output, img_size*sizeof(float)*template_num, cudaMemcpyDeviceToHost);
    //Selecting peak value of each image's convolution result and label out on the image.
    float res = 0;
    int y_pos;
    for(int idx=0; idx<template_num; idx++)
    {
        y_pos = 0;
        res = 0;
        for(int y=0; y<y_size; y++)
        {
            for(int x=0; x<x_size; x++)
            {
                
                if(gpu_out[idx*img_size+y*x_size+x]>res)
                {
                    res = gpu_out[idx*img_size+y*x_size+x];
                    y_pos = y;
                }
            }  
        }
        ext_vec[idx].x = kpt_vec[idx].x;
        ext_vec[idx].y = (img.rows/CROP_PARAM)+dis[idx]+y_pos;
        rectangle(img_cmp, Point(kpt_vec[idx].x-KERNEL_RADIUS,(img.rows/CROP_PARAM)+dis[idx]+y_pos-KERNEL_RADIUS), Point(kpt_vec[idx].x+KERNEL_RADIUS,(img.rows/CROP_PARAM)+dis[idx]+y_pos+KERNEL_RADIUS), Scalar(0,255,0 ), 1, 4);
        line(img,kpt_vec[idx],Point(kpt_vec[idx].x,(img.rows/CROP_PARAM)+dis[idx]+y_pos),Scalar(0,0,255),1,8,0);
    }

    //Free the allocated memory before    
    cudaFree(device_img_input);
    cudaFree(device_img_output);
    free(host_img);
    free(host_templ);
    free(gpu_out);
    return 0;
}
/////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char*argv[])
{
    
    //declear varible here
    int template_num;
    vector<Point2f > pred_vec;
    vector<Point2f > ref_pred_vec;
    Mat status;
    Mat ref_status;
    Mat err;
    Mat ref_err;
    //VideoWriter video("reflection_matching.avi", CV_FOURCC('M','J','P','G'), 10, Size(800, 600),true);
     
    char filename[256];
    char filename_comp[256];
    while(fscanf(stdin, "%s", filename)!=EOF)
    {
        
        cout << filename << endl;
        template_num = TMP_NUM;
        img = imread(filename, -1);
        //Reading the Image to be compared
        if(fscanf(stdin, "%s", filename_comp)!=EOF)
        {
            cout << filename_comp << endl;
            img_cmp = imread(filename_comp, -1);      
        }

        if(!img.data || !img_cmp.data)
        {
            cout << "Problem loading image !!!" << endl;
            return -1;
        }
        //convert the image to gray scale in order to only have one pointer
        cvtColor(img, gray_img, CV_BGR2GRAY);
        cvtColor(img_cmp, gray_img_cmp, CV_BGR2GRAY);
        
        
        Mat mask;
        bool useHarrisDetector = false;
        
        goodFeaturesToTrack(gray_img, corners, TMP_NUM, 0.01, 20.0, mask, 3, useHarrisDetector, 0.04);
        //imshow("L", img);
        //imshow("R", img_cmp);
        waitKey(0);
        if(corners.size() == 0)
        {
            cout << "bad frame" << endl;
            continue;
        }
        Point kpt;
 
        for(int temp_generate_idx = 0; temp_generate_idx<template_num; temp_generate_idx++)
        {   
            kpt = corners[temp_generate_idx];
            //get the predict distance
            dis[temp_generate_idx] = 0;//gray_img.rows/CROP_PARAM-kpt.y;

            //boundary check for the images
            if( kpt.x < KERNEL_RADIUS)  
                kpt.x = KERNEL_RADIUS;
            if( kpt.x > (img.cols-KERNEL_WIDTH) )
                kpt.x = img.cols-KERNEL_WIDTH;
            if( kpt.y < KERNEL_RADIUS)
                kpt.y = KERNEL_RADIUS;
            if( kpt.y > ((img.rows/CROP_PARAM+dis[temp_generate_idx])-KERNEL_WIDTH) )
                kpt.y = (img.rows/CROP_PARAM+dis[temp_generate_idx])-KERNEL_WIDTH;

            //label the original feature point of the image
            rectangle(img, Point(kpt.x-KERNEL_RADIUS,kpt.y-KERNEL_RADIUS), Point(kpt.x+KERNEL_RADIUS,kpt.y+KERNEL_RADIUS), Scalar(255,0,0 ), 1, 4);
            templs[temp_generate_idx] = gray_img(Rect(kpt.x-KERNEL_RADIUS,kpt.y-KERNEL_RADIUS,KERNEL_WIDTH,KERNEL_WIDTH));
            //flip the template in order to find the reflections
            //flip(curr_tmpl,templs[temp_generate_idx],0);

            /*
            imshow("img", img);
            waitKey(0);
            printf("%d:%d\n", temp_generate_idx,dis[temp_generate_idx]);
            */
           
            //cropping the image and store
            //img_vec[temp_generate_idx] = gray_img(Rect(kpt.x-KERNEL_RADIUS,gray_img.rows/CROP_PARAM+dis[temp_generate_idx],KERNEL_WIDTH,gray_img.rows-(gray_img.rows/CROP_PARAM+dis[temp_generate_idx])));
            img_vec[temp_generate_idx] = gray_img_cmp(Rect(0, kpt.y-KERNEL_RADIUS, gray_img_cmp.cols, KERNEL_WIDTH));
            /*
            imshow("temp_img",img_vec[temp_generate_idx]);
            waitKey(0);
            */
            kpt_vec[temp_generate_idx] = kpt;
            
        }
          
        cuda_tp_img(template_num);
        //video.write(img);
        //line(img, Point(0,img.rows/CROP_PARAM), Point(img.cols,img.rows/CROP_PARAM), Scalar(110,220,0));
        imshow("L", img);
        imshow("R",img_cmp);
        waitKey(1);
    }

}
