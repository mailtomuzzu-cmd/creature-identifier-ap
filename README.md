# 🦟 Creature Identifier — Andhra Pradesh Edition

An AI-powered app that identifies creatures found in hostels and campuses 
in Andhra Pradesh and tells you if they are dangerous.

## Live Demo
[Click here to try the app](https://muzakkir-cmd-creature-identifier-ap.hf.space/?__theme=system&deep_link=bVRMoav2LXY)

## Problem
Students living in hostels in Andhra Pradesh regularly encounter unknown 
insects, lizards and spiders in their rooms and campus. This app helps 
identify them instantly and understand if they pose any danger.

## Dataset
- Combined dataset from multiple sources
- Kaggle insect datasets for mosquito, grasshopper, beetle, aphids, armyworm
- Self scraped images using DuckDuckGo for spider, lizard, ant
- 1745 total images across 8 classes

## Classes
1. Ant — low danger, useful for soil
2. Aphids — harmful to crops
3. Armyworm — high danger to crops
4. Beetle — mostly harmless
5. Grasshopper — medium danger to crops
6. Lizard — no danger, very useful
7. Mosquito — high danger, spreads Dengue and Malaria
8. Spider — low danger, controls mosquito population

## Approach
- Transfer learning with ResNet34 using fastai
- Learning rate finder to optimize training
- 8 epochs of fine tuning
- Confusion matrix analysis to identify weak spots
- Rich output layer — danger level, usefulness, interesting facts

## Results
- Accuracy: 98.28%
- Error Rate: 1.72%
- Almost zero confusion between classes

## Key Learning
Model achieved high accuracy because creature classes are visually 
distinct. This contrasts with my skin type classifier where accuracy 
was lower because skin types look visually similar — showing that 
problem difficulty depends on visual separability of classes.

## Technologies Used
- Python
- fastai
- PyTorch
- Gradio
- Hugging Face Spaces

## Future Work
- Add species level classification for dangerous spiders
- Include termites and cockroaches
- Add Telugu language support for farmers
