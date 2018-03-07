*how to reproduce the results*	
						
In summary, the steps are shown below. please refer to refer to "reproduce_results.pptx" for details
								
	1. "make_split.py"
        Split the dataset into train and validation set							
								
	2. "train_<network>.py"
        Train the 7 base nets :							
			xception 						
			resnet101						
			dpnet92						
			se-inceptionV3 						
			se-resnext101/se-resnext101a						
			se-resnet50 						
								
								
	3. "fuse_extract.py"
        Extract the features base nets  							
								
	4. "fuse_train_fcnet<0,1,3>.py"
       Train the 6 fuse nets on combined extracted features from step 3.							
		fcnet3  =	se-inceptionV3 + se-resnext101 + se-resnet50					
		fcnet3  =	se-inceptionV3 + se-resnext101 + se-resnet50					
		fcnet3  =	xception + resnet101 					
		fcnet3  =	dpnet92 + se-resnext101a					
		fcnet1  =	dpnet92 + se-resnext101a					
		fcnet0  =	dpnet92 + se-resnext101a					
								
	5. "fuse_train_submit_cvs.py"
        Make submit cvs file for each of the model in step.4. 
        A memmap file of the test results is also produced for blending in the next step.
							
	6. "blend.py"
        Blend the results of the 6 fuse nets from step 5 to make an ensembled submit cvs file.							

