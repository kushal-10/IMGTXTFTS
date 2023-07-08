# PROJECT MODULE : EXPLAINABLE AI TASK 3

## Overview
Welcome to the repository for the Explainable AI course taught as a Project Module as a part of Cognitive Systems master's program at the University of Potsdam! Artificial Intelligence has witnessed remarkable progress in recent years, permeating every aspect of our lives. As these AI systems become more sophisticated, understanding the reasoning behind their decisions has become increasingly crucial. The importance of transparency, fairness, and accountability in AI systems has led to the emergence of Explainable AI. 

In this course, our primary objective is to equip the knowledge and skills required to design, implement, and evaluate AI models that are not only accurate but also interpretable. We delve into various domains, including language, vision, and multimodal tasks, to explore different approaches and techniques for achieving explainability in AI systems. The course is divided into tasks, each focusing on a specific aspect of Explainable AI:

## Task: Textual and Visual Feature extraction
For this task, the goal is to implement textual and visual feature extraction for the FLICKR30K Dataset.

### Dataset
The [FLICKR30K](https://shannon.cs.illinois.edu/DenotationGraph/) Dataset is modified to create a binary classification task. The dataset iniitally consists of 30k images and their respective captions. There are 5 captions for a single image having different levels of description from brief to extensive. The dataset is modified by assigning images with random captions to create a dataset having an image and a caption which may or may not match the image. The caption is assigned by calculating a similarity score, so that the randomly assigned captions does not match the image.

### Model
To extract visual features [CLIP](https://github.com/openai/CLIP) is used. To extract textual features [spacy](https://spacy.io/usage/linguistic-features) linguistics features are used.
### Usage

1) After cloning the repository, the directory tree should look like this:

---data.txt

---get_features.py

---img_features.py

---labled_instances.jsonl

---saves.py

---text_features.py

---token_frequencies.txt

 2) Visual features are stored in a dataframe 'image_data.csv'. Generate Visual features by running:
    

```python
python3 get_features.py --visual
```

3) Textual features are stored in a dataframe 'image_text_df.csv' combined with visual features. Generate Textual features by running::

```python
python3 get_features.py --textual
```











 
