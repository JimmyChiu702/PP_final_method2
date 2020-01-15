#include "MLP_Layer.h"

#include <string.h>

#define NUM_THREADS 8

void MLP_Layer::Allocate(int previous_num, int current_num)
{
    this->nPreviousUnit   =  previous_num;
    this->nCurrentUnit    =  current_num;
    
    weight          = new float[nPreviousUnit * nCurrentUnit];
    gradient       = new float[nPreviousUnit * nCurrentUnit];
    inputLayer     = new float[nPreviousUnit];
    outputLayer    = new float[nCurrentUnit];
    delta          = new float[nCurrentUnit];
    biasWeight    = new float[nCurrentUnit]; 
    biasGradient  = new float[nCurrentUnit];
    
    srand(1);
    for (int j = 0; j < nCurrentUnit; j++)
    {
        outputLayer[j]=0.0;
        delta[j]=0.0;
        for (int i = 0; i < nPreviousUnit; i++)
        {
            weight[j*nPreviousUnit+i]   = 0.2 * rand() / RAND_MAX - 0.1;
            gradient[j*nPreviousUnit+i]= 0.0;
        }
        biasWeight[j] = 0.2 * rand() / RAND_MAX - 0.1;                             
        biasGradient[j] = 0;
    }
}



void MLP_Layer::Delete(){
    delete [] weight;
    delete [] gradient;
    delete [] delta;
    delete [] outputLayer;
    delete [] biasGradient;
    delete [] biasWeight;
}

float* MLP_Layer::ForwardPropagate(float* inputLayers)      // f( sigma(weights * inputs) + bias )
{
    int size[NUM_THREADS], offset[NUM_THREADS];
    for (int i=0; i<NUM_THREADS; i++) {
        size[i] = nCurrentUnit/NUM_THREADS + (i<nCurrentUnit%NUM_THREADS ? 1 : 0);
        if (i==0)
            offset[i] = 0;
        else
            offset[i] = size[i-1] + offset[i-1];
    }
    
    this->inputLayer=inputLayers;
    
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadID = omp_get_thread_num();
        for(int j = 0 ; j < size[threadID]; j++)
        {
            float net = 0;
            for(int i = 0 ; i < nPreviousUnit ; i++)
            {
                net += inputLayer[i] * weight[(offset[threadID]+j)*nPreviousUnit+i];
            }
            net += biasWeight[offset[threadID]+j];
            
            outputLayer[offset[threadID]+j] = ActivationFunction(net);
        }
    }
    return outputLayer;
}

void MLP_Layer::BackwardPropagateOutputLayer(float* desiredValues)
{   
    int size[NUM_THREADS], offset[NUM_THREADS];
    for (int i=0; i<NUM_THREADS; i++) {
        size[i] = nCurrentUnit/NUM_THREADS + (i<nCurrentUnit%NUM_THREADS ? 1 : 0);
        if (i==0)
            offset[i] = 0;
        else
            offset[i] = size[i-1] + offset[i-1];
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadID = omp_get_thread_num();
        for (int k = 0; k < size[threadID]; k++){
            float fnet_derivative = outputLayer[offset[threadID]+k] * (1 - outputLayer[offset[threadID]+k]);
            delta[offset[threadID]+k] = fnet_derivative * (desiredValues[offset[threadID]+k] - outputLayer[offset[threadID]+k]);
            for (int j = 0 ; j < nPreviousUnit; j++)
                gradient[(offset[threadID]+k)*nPreviousUnit+j] += - (delta[offset[threadID]+k] * inputLayer[j]);
            biasGradient[offset[threadID]+k] += - delta[offset[threadID]+k] ;
        }
    }
}

void MLP_Layer::BackwardPropagateHiddenLayer(MLP_Layer* previousLayer)
{
    int size[NUM_THREADS], offset[NUM_THREADS];
    for (int i=0; i<NUM_THREADS; i++) {
        size[i] = nCurrentUnit/NUM_THREADS + (i<nCurrentUnit%NUM_THREADS ? 1 : 0);
        if (i==0)
            offset[i] = 0;
        else
            offset[i] = size[i-1] + offset[i-1];
    }

    float* previousLayer_weight = previousLayer->GetWeight();
    float* previousLayer_delta = previousLayer->GetDelta();
    int previousLayer_node_num = previousLayer->GetNumCurrent();

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadID = omp_get_thread_num();
        for (int j = 0; j < size[threadID]; j++)
        {
            float previous_sum=0;
            for (int k = 0; k < previousLayer_node_num; k++)
            {
                previous_sum += previousLayer_delta[k] * previousLayer_weight[k*nCurrentUnit + offset[threadID] + j];
            }
            delta[offset[threadID]+j] =  outputLayer[offset[threadID]+j] * (1 - outputLayer[offset[threadID]+j])* previous_sum;
            for (int i = 0; i < nPreviousUnit ; i++) 
                gradient[(offset[threadID]+j)*nPreviousUnit + i] +=  -delta[offset[threadID]+j] * inputLayer[i];
            biasGradient[offset[threadID]+j] += -delta[offset[threadID]+j];
        }
    }
}

void MLP_Layer::UpdateWeight(float learningRate)
{
    int size[NUM_THREADS], offset[NUM_THREADS];
    for (int i=0; i<NUM_THREADS; i++) {
        size[i] = nCurrentUnit/NUM_THREADS + (i<nCurrentUnit%NUM_THREADS ? 1 : 0);
        if (i==0)
            offset[i] = 0;
        else
            offset[i] = size[i-1] + offset[i-1];
    }

    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int threadID = omp_get_thread_num();
        for (int j = 0; j < size[threadID]; j++) {
            for (int i = 0; i < nPreviousUnit; i++) {
                weight[(offset[threadID]+j)*nPreviousUnit + i] +=  -learningRate *gradient[(offset[threadID]+j)*nPreviousUnit + i];
            }
            biasWeight[offset[threadID]+j] += -biasGradient[offset[threadID]+j];
        }
    }

    memset(gradient, 0, sizeof(float)*nCurrentUnit*nPreviousUnit);
    memset(biasGradient, 0, sizeof(float)*nCurrentUnit);
}

int MLP_Layer::GetMaxOutputIndex()
{
    int maxIdx = 0;
    for(int o = 1; o < nCurrentUnit; o++){
        if(outputLayer[o] > outputLayer[maxIdx])
            maxIdx = o;
    }
    
    return maxIdx;
}