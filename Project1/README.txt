My code is basically has a flow of:
1. Loading the data
2. Training and validating the model
3. Saving the model weights in every 20 epoch
4. Testing the model
5. Plotting the tSNE map of the embeddings
6. Testing the model with three given sequences: 'city of new', 'life in the', 'he is the'

You can change the batch size with variable "batch_size", epoch number with variable "epoch_num" and learning rate with initialization variable of the Network class "learning_rate". 

The code itself, naturally starts training with the for loop in the line 51. If you want to directly test the model, please comment out the lines between 51-74. You can define the file that you will load the weights from, in testing part, tSNE part and small testing part differently (lines 76,81,84). I determined the last epoch's weights' file (model200.pk) as default in the code (and I also gave it in the directory). There are functions for saving and loading the weights, in the Network class.