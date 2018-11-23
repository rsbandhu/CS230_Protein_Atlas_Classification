import numpy as np

class EvaluateMetrics():
    def metrics(model, output_batch, labels_batch, params):
        """
        Args: output_batch = predicted output for current batch
              labels_batch = ground truth labels for current batch
              params = hyperparameters
        Output: Accuracy, Precision, Recallm F1Score for current batch
        """
        #set model to training mode
        model.eval()

        #variables for eval
        threshold = 0.5
        num_batches = params.batch_size

        # reset TP, TN, FP, FN to zero for each batch
        True_Positive = 0
        True_Negative = 0
        False_Positive = 0
        False_Negative = 0
        # reset all metrics to zero for each batch
        Accuracy = 0
        Precision = 0
        Recall = 0
        F1Score = 0

        #converting output_batch and labels_batch to numpy array
        output_batch_np = output_batch.detach().numpy()
        labels_batch_np = labels_batch.detach().numpy()
        for i in range(num_batches):
            for j in range(params.classes):
                # True positive (predicted class correctly)
                if ((output_batch_np[i, j] >= threshold) & (labels_batch_np[i,j] == 1)):
                    True_Positive += 1
                # True Negative (predicted absence of class correctly)
                if ((output_batch_np[i, j] < threshold) & (labels_batch_np[i,j] == 0)):
                    True_Negative += 1
                # False positive (When actual class not present predicted class)
                if ((output_batch_np[i, j] >= threshold) & (labels_batch_np[i, j] == 0)):
                    False_Positive += 1
                # False Negative (When actual class present predicted incorrectly)
                if ((output_batch_np[i, j] < threshold) & (labels_batch_np[i, j] == 1)):
                    False_Negative += 1
        # print(True_Positive)
        # print(True_Negative)
        # print(False_Positive)
        # print(False_Negative)
        # Accuracy = (True_Positive+True_Negative)/(True_Positive+True_Negative+False_Positive+False_Negative)
        # if ((True_Positive + False_Positive)!= 0):
        #     Precision = (True_Positive)/(True_Positive+False_Positive)
        # if ((True_Positive+False_Negative)!= 0):
        #     Recall = (True_Positive)/(True_Positive+False_Negative)
        # if ((Precision+Recall)!=0):
        #     F1Score = 2 * (Recall*Precision)/(Recall+Precision)
        # return Accuracy,Precision,Recall,F1Score

        return True_Positive,True_Negative,False_Positive,False_Negative