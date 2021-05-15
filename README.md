# Benchmarking MTL for Dravidian Languages 

This is the code for the paper "Benchmarking Multi-Task Learning for Sentiment Analysis and Offensive Language Identification in Under-Resourced Dravidian Languages"

## [Steps to run Single Task Models:](https://github.com/SiddhanthHegde/Dravidian-MTL-Benchmarking/tree/main/Single%20Tasks)
1) For character BERT, XLM and XLNet run the specific task file and find the string 'read_csv'. Change the path to the dataset where you have stored and run the program on the terminal

2) For other BERT versions and XLMr go to BERT versions and XLMr folder and use train_task1.py for sentiment classification and offensive language detection. Find the string 'read_csv'. Change the path to the dataset where you have stored and run the program on the terminal.

Steps given above can be used for Kannada, Malayalam and Tamil. For custom datasets make sure you have a csv file with 'comment', 'sent', 'off' as the column names and they can be used as well.

## Steps to run Multi Task Models:

For Hard Parameter sharing and Soft parameter sharing use the train.py file. Find the string 'read_csv'. Change the path to the dataset where you have stored and run the program on the terminal.


Message: Please cite the following when using this code
``` *
@article{Hande-etal-Multitask,
    title = "Benchmarking Multi-Task Learning for Sentiment Analysis and Offensive Language Identification in Under-Resourced Dravidian Languages",
    author = "Hande, Adeep  and
      U Hegde, Siddhanth  and
      Priyadharshini, Ruba  and
      Ponnusamy, Rahul  and
      Kumaresan, Prasanna Kumar and
      Thavareesan, Sajeetha and
      Chakravarthi, Bharathi Raja ",
      journal={Cognitive Computation},
      publisher={Springer}
    }
*```
