# Dementia-using-machine-learning
Detection of Dementia using machine learning and python 

✅Abstract :
Dementia is a neurodegenerative disorder, which is one of the most common among older
people, it constitutes one of the diseases with great social impact in Europe and America. The
progress of medical diagnosis using magnetic resonance imaging (MRI) is widely used for
the treatment of neurological diseases; it allows obtaining, increasingly, more functional and
anatomical information from the brain of the patients, with a great precision in time and
space. This model proposes a methodology for the diagnosis of dementia based on
Alzheimer’s disease combining imaging processing and artificial intelligence techniques by
creating an artificial neural network (ANN) of classification based on architecture Multilayer
Perceptron. In order to construct a complete dataset for training and testing of the network,
initial inputs-target variables were obtained from the database OASIS (Open Access Series of
imaging studies) with a total of instances equal to 235. The variables were classified into 3
groups: demographic, clinical and morphometric data. Task of training and testing were
applied on initial data which would result in 48% confusion error. For minimizing this
percentages of error, image processing and Voxel based morphometry (VBM) techniques
were implemented to obtain new morphometric variables of three areas of the brain: white
matter (WM), gray matter (GM) and fluid (CSF) cerebro-espinal. In this way, by reducing the
percentage of confusion to 17%, the results obtained with the ANN, demonstrates that the
demographic and clinical information from patients, combined with morphometric
information of areas of the brain, are input variables useful to train an ANN of diagnosis of
dementia with 83 % of reliability, and in this way, help to the early diagnosis of Dementia.

✅Introduction:
Dementia is a global health issue. Dementia is a state describing some group of symptoms which
can occur when few groups of cells of brain stop working in an appropriate manner. Dementia
normally occurs inside the particular regions of the brain that influences the way one can think,
recall and convey. Dementia usually occurs due to Alzheimer’s disease. Dementia occurring
before 65 years of age is said to be as early-onset dementia whereas after 65 years is said as lateonset dementia. 
The World Alzheimer Report of the year 2015 shows the global number of
people suffering from dementia are 46.8 million and an increase of 74.7 million is estimated by
2030 and 131.5 million by 2050. The types of Dementia are Alzheimer’s Disease,
Frontotemporal Dementia, Dementia with Lewy Bodies and Vascular Dementia. Alzheimer’s
Disease is responsible for almost 75% of the overall cases. A state when both Alzheimer’s and
Vascular Dementia occur simultaneously is called as Mixed Dementia. Although there are a
number of medical areas to which Machine Learning ( ML ) systems have been applied,
dementia has not been one of them. Yet dementia is a complex problem with correct diagnosis
requiring historical data, physical exam, cognitive testing, laboratory studies and imaging.
Although diagnostic accuracy using clinical NINCDS-ADRDA criteria for probable AD 1 is
about 88% compared to post-mortem diagnosis, most demented patients are seen by community
physicians who often do not detect dementia or misdiagnosis it. This problem is compounded by
the average delay of 4 years between symptom onset and physician contact, which usually relates
to the patient's social embarrassment about having a memory problem. At this point of the
disease, physicians are less able to slow the progression and minimize debilitating behavioral
changes of the dementia. A simple, method for detecting dementia early in the disease's course
would help get patients to seek early evaluation and treatment.
 
 ✅Tools Used :
 Jupyter Notebook,
 Pycharm
 
 ✅Purpose of the Project:
Early detection and correct diagnosis can lead to disease-retarding therapies which can slow
disease progression and reduce patient and caregiver stress and morbidities. After excluding
delirium and depression, two forms, the Mini mental state examination( MIMSE ) can be used to
determine if a person has dementia, which is defined as multiple cognitive impairments with loss
of functional skills related to those cognitive impairments without an altered level of
consciousness. The ANN and J48 methods can simplify the task of interpreting test results by
constructing a set of criteria to classify the patient. In our project we report the use of the
MIMSE , eTIV tests in conjunction with ML methods. The goal was to determine whether ML
methods improve the accuracy of diagnosis of these dementia screening tests for classifying a
subject as either demented or not demented. By feeding data with machine learning techniques,
we can significantly cut down on the amount of people suffering from dementia that are
undiagnosed So having model that can take a vast amount of data, and automatically identify
patients with possible dementia, to facilitate targeted screening, could potentially be very useful
and help improve diagnosis rates.

✅Scope:
The model proposes a methodology for the diagnosis of dementia based on Alzheimer’s disease
combining imaging processing, machine learning(j48),artificial intelligence techniques by
creating an artificial neural network (ANN) of classification based on architecture Multilayer
Perceptron. In order to construct a complete dataset for training and testing of the network, initial
inputs-target variables were obtained from the database OASIS (Open Access Series of imaging
studies) with a total of instances equal to 235. The variables were classified into 3 groups:
demographic, clinical and morphometric data. Task of training and testing were applied on initial
data which would result in 48% confusion error. For minimizing this percentages of error, image
processing and Voxel based morphometry (VBM) techniques are being implemented to obtain
new morphometric variables of three areas of the brain: white matter (WM), gray matter (GM)
and fluid (CSF) cerebroespinal. In this way, by reducing the percentage of confusion to 17%, the
results obtained with the ANN, would be able to demonstrate that the demographic and clinical
information from patients, combined with morphometric information of areas of the brain, are
input variables useful to train an ANN of diagnosis of dementia with 83 % of reliability, and in
this way, help to the early diagnosis of Dementia.

✅Existing System:
With the expected growth in dementia prevalence, the number of specialist memory clinics may
be insufficient to meet the expected demand for diagnosis. Furthermore, although current ’gold
standards’ in dementia diagnosis may be effective, they involve the use of expensive
neuroimaging (for example, positron emission tomography scans) and time-consuming
neuropsychological assessments which is not ideal for routine screening of dementia.
In the proposed model, the dataset is collected from the OASIS-Brains.org . The Open Access
Series of Imaging Studies (OASIS) which allows free availability of the brain MRI data sets. The
OASIS dataset contains Longitudinal MRI data of non-demented and demented older adults. We
have applied the following classification techniques: J48 and Multilayer Perceptron to the
dataset. And further, Attribute Selection technique is also applied using CFSSubsetEval. The
attributes included in the OASIS dataset are age, sex, education, socioeconomic status, minimental state examination, 
clinical dementia rating , atlas scaling factor, estimated total
intracranial volume, and normalized whole-brain volume.

✅Problem Statement:
The aim of this study is to develop a machine learning-based model that could be used in general
practice to detect dementia from collected data and brain MRI images. The model would be a
useful tool for identifying people who may be living with dementia but have not been formally
diagnosed or have a proclivity towards the disease.

✅Summary:
So having model that can take a vast amount of data, and automatically identify patients with
possible dementia, to facilitate targeted screening, could potentially be very useful and help
improve diagnosis rates. Hence with machine learning techniques like Multilayer perceptron and
j48 along with Mini mental state examination helps in achieving the task. Thus, we can
significantly cut down on the amount of people suffering from dementia that are undiagnosed.


