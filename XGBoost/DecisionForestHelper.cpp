#include "stdafx.h"
#include "DecisionForestHelper.h"
#include <random>


DecisionForestHelper::DecisionForestHelper()
{
}


DecisionForestHelper::~DecisionForestHelper()
{
}

int DecisionForestHelper::ReturnMostLikelyClass(DecisionTreeNode** forest, DataPoint* dataPoint, Config* config)
{
	int numberOfTrees = config->NumberOfTrees;
	int numberOfClasses = config->NumberOfClasses;

	// create new weight vector
	double* weights = new double[numberOfClasses];
	for (int k = 0; k < numberOfClasses; k++) weights[k] = 0;

	// populate weight vector from forest
	for (int j = 0; j < numberOfTrees; j++)
	{
		double* newWeights = ClassifyPoint(forest[j], dataPoint);
		for (int k = 0; k < numberOfClasses; k++) weights[k] += newWeights[k];
	}

	// find most likely weight
	int mostLikelyClass = 0;
	double bestClassWeight = 0;
	for (int k = 0; k < numberOfClasses; k++)
	{
		if (weights[k] > bestClassWeight)
		{
			bestClassWeight = weights[k];
			mostLikelyClass = k;
		}
	}

	delete[] weights;

	return mostLikelyClass;
}

DecisionTreeNode** DecisionForestHelper::TrainForest(DataHandler* data, Config* config)
{
	int numberOfTrees = config->NumberOfTrees;
	DecisionTreeHelper* treeEngine = new DecisionTreeHelper();
	DecisionTreeNode** forest = new DecisionTreeNode*[numberOfTrees];
	srand(0);

	for (int treeIndex = 0; treeIndex < numberOfTrees; treeIndex++)
	{
		// set up training data
		DataHandler* trainingData = new DataHandler();
		trainingData->NumberOfDataPoints = config->NumberOfTrainingPoints / 2;
		trainingData->DataType = data->DataType;
		trainingData->DataPoints = new DataPoint*[trainingData->NumberOfDataPoints];

		for (int j = 0; j < trainingData->NumberOfDataPoints; j++)
		{
			// pick random training point
			int index = rand() % config->NumberOfTrainingPoints;
			trainingData->DataPoints[j] = data->DataPoints[index];
		}

		// train tree
		//DataHandler* copy = data;
		forest[treeIndex] = treeEngine->TrainTree(config, trainingData);

		// update weights for all data
		for (int dataIndex = 0; dataIndex < config->NumberOfTrainingPoints; dataIndex++)
		{
			double* newWeights = ClassifyPoint(forest[treeIndex], data->DataPoints[dataIndex]);
			for (int j = 0; j < config->NumberOfClasses; j++)
			{
				data->DataPoints[dataIndex]->Weights[j] += newWeights[j];
			}
			//delete[] newWeights;
		}
	}

	delete treeEngine;

	return forest;
}

double* DecisionForestHelper::ClassifyPoint(DecisionTreeNode* node, DataPoint* dataPoint)
{
	if (node->isLeaf)
	{
		return node->Weights;
	}
	else
	{
		DataHandler* handle = new DataHandler();
		if (handle->AskQuestion(node->Question, dataPoint))
		{
			return ClassifyPoint(node->RightChild, dataPoint);
		}
		else
		{
			return ClassifyPoint(node->LeftChild, dataPoint);
		}
	}
}