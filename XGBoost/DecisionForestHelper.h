#pragma once
#include "DecisionTreeHelper.h"

class DecisionForestHelper
{
public:
	DecisionForestHelper();
	~DecisionForestHelper();
	int ReturnMostLikelyClass(DecisionTreeNode** forest, DataPoint* dataPoint, Config* config);
	DecisionTreeNode** TrainForest(DataHandler* data, Config* config);
	double* ClassifyPoint(DecisionTreeNode* node, DataPoint* dataPoint);
};

