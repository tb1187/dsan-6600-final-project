# Purpose

The purpose of this project is to provide users with a simple graphical user interface that will give personalized dietary advice based on the meals they eat. This is achieved by first predicting the macros associated with an image of food.

# Running The Code

To access the graphical user interface, the user simply has to run the command 'python src/gui.py' from the project root directory. Since we are leveraging OpenAI's LLM, the user will need to have an API key in order to access the personalized dietary advice. Additionally, all of the required dependencies can be found in the 'requirements.txt' file.

# Deeper Dive

For those wanting to look deeper, you can find all of our code under the 'src/' folder. Here, you will see how we coded our preprocessing pipeline, data loader, model specifications, optimizer, and training runs. 

The .ipynb files found under the 'notebooks/' folder will explain how to implement the loader, model, and optimizer classes. Here, you can also find some EDA we conducted prior to model building.

Once trained, model weights are stored in the 'models/' folder and referenced in the GUI. 3 models were tested under different conditions, and the model yielding the best validation loss was used in the end.

Previous works that inspired this project can be found in the 'resources/' folder.

# Use of AI

Artificial intelligence tools (specifically OpenAI’s ChatGPT-5.1) were used throughout the development of this project in the following ways:

• Problem-Solving & Debugging  
  AI assistance was used to help identify and resolve errors in Python scripts, PyTorch model definitions, dataset handling, data loaders, normalization routines, and GUI integration. This included debugging mis-indexing issues, incorrect layer freezing, misapplied transforms, early-stopping logic, and denormalization errors.

• Code Architecture & Refactoring  
  ChatGPT assisted in guiding portions of the project architecture, model-training scripts, evaluation utilities, and the PyQt5 GUI layout. All AI-generated code was manually reviewed, tested, and modified by the project authors.

• Conceptual Clarification  
  Explanations provided by ChatGPT supported understanding of convolutional backbones, transfer learning, normalization, overfitting behavior, and dataset biases. These explanations informed decisions about model selection, training strategy, and preprocessing.

• Report Writing Assistance  
  AI was used to help edit, restructure, and improve clarity in sections such as Methods, Related Work, Training Procedure, Results, and Discussion.
