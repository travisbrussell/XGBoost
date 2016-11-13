#include "stdafx.h"
#include "DecisionTreeHelper.h"
#include <math.h>


DecisionTreeHelper::DecisionTreeHelper()
{
}

DecisionTreeHelper::~DecisionTreeHelper()
{
}

DecisionTreeNode* DecisionTreeHelper::TrainTree(Config* config, DataHandler* data)
{
	DecisionTreeNode* root = new DecisionTreeNode();

	if (config->IsDAG)
	{
		TrainDAG(config, data, root);
	}
	else
	{
		// Train tree
		// compute weights for node
		double* weights = ComputeNodeWeights(config, data);
		root = ReturnOrSplit(config, data, 1, weights);
	}

	return root;
}

DecisionTreeNode* DecisionTreeHelper::ReturnOrSplit(Config* config, DataHandler* data, int level, double* weights)
{
	DecisionTreeNode* node = new DecisionTreeNode();
	node->Level = level;

	if (level < config->MaxNumberOfLevels)
	{
		// Find best splitting question
		Question** questionsToTest = data->ReturnQuestions(data->DataType, config);
		int numberOfQuestions = data->ReturnNumberOfQuestions(data->DataType, config);
		Question* bestQuestion = FindBestSplittingQuestion(config, data, questionsToTest, numberOfQuestions);
		delete[] questionsToTest;

		// split datapoints into left and right partitions
		DataHandler* leftData = PartitionData(data, bestQuestion, false);
		DataHandler* rightData = PartitionData(data, bestQuestion, true);

		// compute left and right weights
		double* leftWeights = ComputeNodeWeights(config, leftData);
		double* rightWeights = ComputeNodeWeights(config, rightData);

		// compute split error
		double leftError = ComputeError(leftWeights, leftData, config);
		double rightError = ComputeError(rightWeights, rightData, config);
		double splitError = leftData->NumberOfDataPoints * leftError + rightData->NumberOfDataPoints * rightError;
		splitError = splitError / (leftData->NumberOfDataPoints + rightData->NumberOfDataPoints);
		splitError += ComputeRegularizationPenalty(config, leftWeights) + ComputeRegularizationPenalty(config, rightWeights);

		// compare with initial error
		double error = ComputeError(weights, data, config) + ComputeRegularizationPenalty(config, weights);

		if (splitError < error)
		{
			node->isLeaf = false;
			node->Question = bestQuestion;
			delete[] weights;
			delete data;
			node->LeftChild = ReturnOrSplit(config, leftData, level + 1, leftWeights);
			node->RightChild = ReturnOrSplit(config, rightData, level + 1, rightWeights);
		}
		else
		{
			node->isLeaf = true;
			node->Weights = new double[config->NumberOfClasses];
			for (int j = 0; j < config->NumberOfClasses; j++) node->Weights[j] = weights[j];
			delete[] weights;
			delete data;
		}
	}
	else
	{
		node->isLeaf = true;
		node->Weights = new double[config->NumberOfClasses];
		for (int j = 0; j < config->NumberOfClasses; j++) node->Weights[j] = weights[j];
		delete[] weights;
		delete data;
	}

	return node;
}

double DecisionTreeHelper::ComputeRegularizationPenalty(Config* config, double* weights)
{
	double penality = 0;
	for (int classIndex = 0; classIndex < config->NumberOfClasses; classIndex++)
	{
		double weight = weights[classIndex];
		penality += weight * weight;
	}

	return config->eta * penality;
}

double* DecisionTreeHelper::ComputeNodeWeights(Config* config, DataHandler* data)
{
	double* weights = new double[config->NumberOfClasses];
	for (int j = 0; j < config->NumberOfClasses; j++) weights[j] = 0;

	for (int dataIndex = 0; dataIndex < data->NumberOfDataPoints; dataIndex++)
	{
		DataPoint* point = data->DataPoints[dataIndex];
		for (int classIndex = 0; classIndex < config->NumberOfClasses; classIndex++)
		{
			if (classIndex == point->ClassName)
			{
				weights[classIndex] += (1 - point->Weights[classIndex]) / data->NumberOfDataPoints;
			}
			else
			{
				weights[classIndex] -= point->Weights[classIndex] / data->NumberOfDataPoints;
			}
		}
	}

	return weights;
}

Question* DecisionTreeHelper::FindBestSplittingQuestion(Config* config, DataHandler* data, Question** questions, int numberOfQuestions)
{
	int bestQuestion = 0;
	double bestError = 1;

	for (int questionIndex = 0; questionIndex < numberOfQuestions; questionIndex++)
	{
		Question* question = questions[questionIndex];
		// split datapoints into left and right partitions
		DataHandler* leftData = PartitionData(data, question, false);
		DataHandler* rightData = PartitionData(data, question, true);

		// compute left and right weights
		double* leftWeights = ComputeNodeWeights(config, leftData);
		double* rightWeights = ComputeNodeWeights(config, rightData);

		// compute error
		double leftError = ComputeError(leftWeights, leftData, config);
		double rightError = ComputeError(rightWeights, rightData, config);
		double error = leftData->NumberOfDataPoints * leftError + rightData->NumberOfDataPoints * rightError;
		error = error / (leftData->NumberOfDataPoints + rightData->NumberOfDataPoints);
		if (error < bestError)
		{
			bestError = error;
			bestQuestion = questionIndex;
		}

		delete[] leftWeights;
		delete[] rightWeights;
	}

	return questions[bestQuestion];
}

double DecisionTreeHelper::ComputeError(double* weights, DataHandler* data, Config* config)
{
	int numberOfDataPoints = data->NumberOfDataPoints;
	int numberOfClasses = config->NumberOfClasses;
	double error = 0;

	for (int dataIndex = 0; dataIndex < numberOfDataPoints; dataIndex++)
	{
		DataPoint* point = data->DataPoints[dataIndex];
		for (int classIndex = 0; classIndex < numberOfClasses; classIndex++)
		{
			double value = Sigmoid(point->ClassName - weights[classIndex]);
			error += value*value;
		}
	}

	return error / (numberOfClasses + numberOfDataPoints);
}

double DecisionTreeHelper::Sigmoid(double input)
{
	return 1.0 / (1 + exp(-input));
}

DataHandler* DecisionTreeHelper::PartitionData(DataHandler* data, Question* question, bool right)
{
	int numberOfDataPoints = 0;
	bool* isInPartition = new bool[data->NumberOfDataPoints];
	for (int dataIndex = 0; dataIndex < data->NumberOfDataPoints; dataIndex++)
	{
		DataPoint* point = data->DataPoints[dataIndex];
		if (data->AskQuestion(question, point) == right)
		{
			numberOfDataPoints++;
			isInPartition[dataIndex] = true;
		}
		else
		{
			isInPartition[dataIndex] = false;
		}
	}

	// create partition
	DataHandler* partition = new DataHandler();
	partition->NumberOfDataPoints = numberOfDataPoints;
	partition->DataType = data->DataType;
	partition->NumberOfAttributes = data->NumberOfAttributes;
	partition->NumberOfAttributeValues = data->NumberOfAttributeValues;
	partition->DataPoints = new DataPoint*[numberOfDataPoints];
	int partitionIndex = 0;
	for (int dataIndex = 0; dataIndex < data->NumberOfDataPoints; dataIndex++)
	{
		if (isInPartition[dataIndex])
		{
			partition->DataPoints[partitionIndex] = data->DataPoints[dataIndex];
			partitionIndex++;
		}
	}
	delete[] isInPartition;
	
	return partition;
}

void DecisionTreeHelper::TrainDAG(Config* config, DataHandler* data, DecisionTreeNode* root)
{
	// Train DAG
}