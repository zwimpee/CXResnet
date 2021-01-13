# CXResnet
This project is an attempt to produce an image classification model that can distinguish between chest X-rays labeled as either normal, pneumonia, or COVID-19. Although performance would likely be optimal using a pretrained model, the model to be trained here will have an architecture built from custom components created through subclassing PyTorch's nn.Module. Note that the data preprocessing script can be tailored to future use cases involving Kaggle datasets.

#### Requirements
- [ ] *determine complete set of dependencies*
#### Usage
- [ ] *create useage guide for reproducing and modifying project* 
## Project procedures and results
### Motivation and Background
  The COVID-19 pandemic has presented seemingly all areas of society with a host of new and unprecedented challenges, many of which are faced in the healthcare field. While many people living in developed countries are fortunate enough to have relatively easy access to testing, those living in developing or impoverished countries may not be afforded this same luxury.
  
  The virus, however, does not discriminate, and therefore additional tools are needed to aid in diagnosing infected patients when the number of tests available are limited (or perhaps even nonexistent). This project explores the use of neural network image classification models as a potential tool in such situations.
  
  With that being said it is important to acknowledge that a predictive model by itself might not be the most useful tool for healthcare professionals, even for a model with excellent predictive power. This is especially true for a neural network model by itself, which can leave much to be desired with respect to the interpretability of the results. Therefore the primary goals for this project is to provide both a predictive model with a nontrivial degree of predictive power, and an intuitive communication of what the model "sees" when making its predictions. The latter will be achieved using the Captum library for PyTorch model interpretability. 
  
  For those without any prior training in medicine such as myself, it might be useful to know some [basics on reading chest X-rays](https://iem-student.org/how-to-read-chest-x-rays/):
  

- Motivation
  - [x] Explain the problem 
  - [x] Propose solution(s)
  - [x] Describe the value of project deliverables w.r.t. proposed solution 
- Background
  - [ ] Summarize how chest X-rays are read
  - [ ] Relate this to the expected end results
### Data wrangling
- [ ] Summarize data sources, and processing
### Model Architecture
- [ ] Get simple visual representation of model architecture
### Training 
- [ ] Describe the structure of the training code
- [ ] Show plots of training history
### Testing
- [ ] Discuss model performance on holdout test data
### Insight and Interpretability
- [ ] Show and discuss results of input attribution obtained using Captum

![alt text](figures/occlusion_attribution_COVID-19_(284).png)
![alt text](figures/occlusion_attribution_Normal_(429).png)


## Credits
```
SAIT, UNAIS; k v, Gokul Lal; Prajapati, Sunny; Bhaumik, Rahul; Kumar, Tarun; S, Sanjana; Bhalla , Kriti
(2020), “Curated Dataset for COVID-19 Posterior-Anterior Chest Radiography Images (X-Rays).”,
Mendeley Data, V1, doi: 10.17632/9xkhgts2s6.1
```
```
Cleverley Joanne, Piper James, Jones Melvyn M.
The role of chest radiography in confirming covid-19 pneumonia
BMJ 2020; 370 :m2426
```

The files in the original dataset are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
