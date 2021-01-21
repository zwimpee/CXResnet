# CXResnet
This project is an attempt to produce an image classification model that can distinguish between chest X-rays labeled as either normal, pneumonia, or COVID-19. Although performance would likely be optimal using a pretrained model, the model to be trained here will have an architecture built from custom components created through subclassing PyTorch's nn.Module.

#### Requirements
- [ ] *determine complete set of dependencies*
#### Usage
- [ ] *create useage guide for reproducing and modifying project* 

## Project procedures and results
- - -
### Motivation and Background
  The COVID-19 pandemic has presented seemingly all areas of society with a host of new and unprecedented challenges, many of which are faced in the healthcare field. While many people living in developed countries are fortunate enough to have relatively easy access to testing, those living in developing or impoverished countries may not be afforded this same luxury.
  
  The virus, however, does not discriminate, and therefore additional tools are needed to aid in diagnosing infected patients when the number of tests available are limited (or perhaps even nonexistent). This project explores the use of neural network image classification models as a potential tool in such situations.
  
  With that being said it is important to acknowledge that a predictive model by itself might not be the most useful tool for healthcare professionals, even for a model with excellent predictive power. This is especially true for a neural network model by itself, which can leave much to be desired with respect to the interpretability of the results. Therefore the primary goals for this project is to provide both a predictive model with a nontrivial degree of predictive power, and an intuitive communication of what the model "sees" when making its predictions. The latter will be achieved using the Captum library for PyTorch model interpretability. 
  
  For those without any prior medical training such as myself, it might be useful to know some basics on reading chest X-rays. Here are some resources I used while working on this project for those who might be interested.
  
- <ins>Some basic and relatively non-technical resources</ins>  
  - [Blog post on the very basics of reading chest X-Rays](https://iem-student.org/how-to-read-chest-x-rays/)
  - [A more in-depth tutorial on reading and interpreting chest X-Rays](https://www.med-ed.virginia.edu/courses/rad/cxr/index.html)
  - [Wiki article on ground-glass opacification](https://radiopaedia.org/articles/ground-glass-opacification-3)
  
- <ins>More technical/academic resources</ins> <sub>(citations can be found below in _References_ section)</sub>
  - https://www.bmj.com/content/370/bmj.m2426
  - https://pubs.rsna.org/doi/10.1148/ryct.2020200280 
  - https://rdcu.be/cdvSJ
- - - 
### Data wrangling
- [ ] Summarize data sources, and processing
- - -
### Model Architecture
- [ ] Get simple visual representation of model architecture
- - -
### Training 
- [ ] Describe the structure of the training code
- [ ] Show plots of training history
- - -
### Testing
- [ ] Discuss model performance on holdout test data
- - -
### Insight and Interpretability
- [ ] Show and discuss results of input attribution obtained using Captum

![alt text](figures/occlusion_attribution_COVID-19_(542).png)
![alt text](figures/occlusion_attribution_COVID-19_(1267).png)

- - -
## References
This reference's dataset files are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)
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
```
A Characteristic Chest Radiographic Pattern in the Setting of COVID-19 Pandemic
David L. Smith, John-Paul Grenier, Catherine Batte, and Bradley Spieler
Radiology: Cardiothoracic Imaging 2020 2:5
```
```
Rousan, L.A., Elobeid, E., Karrar, M. et al.
Chest x-ray findings and temporal lung changes in patients with COVID-19 pneumonia.
BMC Pulm Med 20, 245 (2020). https://doi.org/10.1186/s12890-020-01286-5
```


