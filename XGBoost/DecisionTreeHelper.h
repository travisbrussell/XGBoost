#pragma once

#include "DecisionTreeNode.h"
#include "Config.h"
#include "DataHandler.h"
#include "DataPoint.h"

class DecisionTreeHelper
{
public:
	DecisionTreeHelper();
	~DecisionTreeHelper();

	DecisionTreeNode* TrainTree(Config* conifg, DataHandler* data);
	void TrainDAG(Config* config, DataHandler* data, DecisionTreeNode* root);
	DecisionTreeNode* ReturnOrSplit(Config* config, DataHandler* data, int level, double* weights);
	double ComputeRegularizationPenalty(Config* config, double* weights);
	double* ComputeNodeWeights(Config* config, DataHandler* data);
	Question* FindBestSplittingQuestion(Config* config, DataHandler* data, Question** questions, int numberOfQuestions);
	double ComputeError(double* weights, DataHandler* data, Config* config);
	double Sigmoid(double input);
	DataHandler* PartitionData(DataHandler* data, Question* question, bool right);
};

