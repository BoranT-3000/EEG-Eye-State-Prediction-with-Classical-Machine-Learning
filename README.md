# EEG Eye State Prediction with Classical Machine Learning

**Suggested GitHub description:** Classical ML pipeline for EEG eye-state prediction using LazyPredict screening, Optuna tuning, and literature comparison on the EEG Eye State dataset.

## Türkçe Açıklama

Bu proje, Brain-Computer Interfaces dersi kapsamında EEG sinyallerinden göz durumunu tahmin etmek için hazırlanmıştır. Amaç, tek bir kişiden kaydedilmiş 14 kanallı EEG verisini kullanarak gözün açık mı kapalı mı olduğunu klasik makine öğrenmesi yöntemleriyle sınıflandırmaktır. Projede yapay sinir ağı veya derin öğrenme modeli kullanılmamış; bunun yerine KNN, Label Propagation, Label Spreading, Extra Trees, Random Forest, XGBoost, LightGBM, SVC ve benzeri klasik ML modelleri denenmiştir.

Bu çalışmayı yapmamın nedeni, EEG tabanlı BCI problemlerinde basit görünen bir ikili sınıflandırma görevinin bile model seçimi, veri temsili, ölçekleme, outlier temizliği, çapraz doğrulama stratejisi ve literatür karşılaştırması açısından dikkatli tasarlanması gerektiğini göstermektir. Literatürde bu veri setinde KStar, instance-based learning ve ensemble tree yaklaşımlarının güçlü sonuçlar verdiği görülmektedir. Bu nedenle proje önce geniş bir model taraması yapmış, ardından en başarılı modeller Optuna ile optimize edilmiştir.

## English Description

This project was developed for the Brain-Computer Interfaces course assignment on EEG-based eye-state prediction. The objective is to classify whether the subject's eyes are open or closed using 14-channel EEG data from a single subject. The project intentionally uses classical machine learning only; no ANN or deep learning model is used.

The motivation is to show that even a compact EEG binary classification task requires careful decisions about dataset representation, outlier handling, scaling, validation strategy, model selection, and comparison with previously reported literature values. Since previous studies on the same dataset reported strong results with KStar, instance-based learning, and tree ensembles, this project evaluates similar classical ML families and tunes the best candidates with Optuna.

## Dataset

The project uses the public **EEG Eye State** dataset originally introduced by Rösler and Suendermann. The dataset contains 14 EEG channels and one target column named `eyeDetection`.

- `0`: eye open
- `1`: eye closed
- Final selected representation: `raw_sample`
- Final feature matrix shape after robust outlier removal: `(14974, 14)`

## Method Summary

1. Downloaded and loaded the original `.arff.gz` dataset.
2. Converted the target label into integer form.
3. Saved a CSV copy for easier inspection during notebook development.
4. Applied robust MAD-based outlier detection.
5. Tested two representations:
   - raw sample classification
   - window-based statistical feature classification
6. Used LazyPredict and manual cross-validation for model screening.
7. Tuned the best candidates with Optuna.
8. Compared the final result with reported values from the literature.

## Why Google Colab?

Google Colab is suitable for this project because it provides a reproducible environment with easy package installation. The workflow uses packages such as `scipy`, `scikit-learn`, `LazyPredict`, `Optuna`, `XGBoost`, `LightGBM`, and optionally `SHAP` / `LIME`. Using Colab reduces local dependency issues and makes it easier for the instructor or other users to reproduce the notebook from the public repository.

## Models Tested

The following model families were evaluated:

- Logistic Regression
- Gaussian Naive Bayes
- LDA / QDA
- KNN Uniform / KNN Distance
- KNN Distance + RobustScaler
- Label Propagation RBF / KNN
- Label Spreading RBF / KNN
- SVC RBF
- Decision Tree
- Random Forest
- Extra Trees
- Bagging Trees
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM

## Main Results

### Raw-sample screening

| Model | CV Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Label Spreading RBF | 0.971484 | 0.971024 | 0.971175 |
| Label Propagation RBF | 0.971417 | 0.970977 | 0.971109 |
| KNN Distance + RobustScaler | 0.964138 | 0.963339 | 0.963728 |
| KNN Distance | 0.961332 | 0.960683 | 0.960908 |
| Extra Trees | 0.952317 | 0.949966 | 0.951620 |

### LazyPredict screening

| Model | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
|---|---:|---:|---:|---:|
| LabelPropagation | 0.968948 | 0.968791 | 0.996258 | 0.968954 |
| LabelSpreading | 0.968614 | 0.968489 | 0.995956 | 0.968621 |
| KNeighborsClassifier | 0.962938 | 0.962026 | 0.991546 | 0.962918 |
| ExtraTreesClassifier | 0.944908 | 0.942282 | 0.990361 | 0.944774 |
| XGBClassifier | 0.931553 | 0.930376 | 0.981957 | 0.931520 |

### Optuna tuning

| Model | Best CV Accuracy | Best Parameters Summary |
|---|---:|---|
| Label Propagation RBF | 0.974823 | `gamma≈9.805`, `max_iter≈962` |
| Label Spreading RBF | 0.970883 | `gamma≈28.376`, `alpha≈0.252` |
| KNN Distance + RobustScaler | 0.971951 | `n_neighbors=1`, `p=1` |

### Final hold-out result

| Metric | Value |
|---|---:|
| Dataset | raw_sample |
| Final model | Label Propagation RBF |
| Accuracy | 0.973957 |
| Balanced accuracy | 0.973611 |
| Macro F1-score | 0.973676 |
| Macro precision | 0.973743 |
| Macro recall | 0.973611 |

## Literature Comparison

| Study | Method | Reported Accuracy | Note |
|---|---|---:|---|
| Our experiment | Label Propagation RBF on raw_sample | 0.973957 | Hold-out test result after Optuna tuning |
| Rösler & Suendermann, 2013 | KStar / Weka | 0.973000 | Original reference study; 10-fold CV |
| Al-Taei, 2017 | Vote(KStar + Random Forest) | 0.972700 | Voting ensemble |
| Sahu et al., 2015 | IB1 | 0.945100 | Instance-based learning |
| Nilashi et al., 2023 | LVQ + Bagged Trees | 0.943100 | Hybrid clustering + ensemble trees |
| Sahu et al., 2015 | IBK-3 | 0.934200 | KNN-like instance-based method |
| Sahu et al., 2015 | Random Forest | 0.892700 | Tree ensemble |

## Notes on Label Propagation and Label Spreading

Although Label Propagation and Label Spreading are commonly introduced as semi-supervised graph-based algorithms, they were evaluated here under the same supervised train/test and cross-validation protocol as the other classifiers. No test labels were used during training.

## License

This project is intended to be released under the **MIT License**.

The MIT License applies to the authored code and project documentation. It allows reuse, modification, distribution, and publication of the software as long as the copyright and license notice are preserved. The dataset and cited papers keep their own original licenses and citation requirements.

## References

1. O. Rösler and D. Suendermann, “A First Step towards Eye State Prediction Using EEG,” 2013. https://suendermann.com/su/pdf/aihls2013.pdf
2. UCI Machine Learning Repository, “EEG Eye State” dataset. https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
3. M. Sahu et al., “Performance Evaluation of Different Classifier for Eye State Prediction Using EEG Signal,” 2015. https://www.ijke.org/vol1/24-E002.pdf
4. A. Al-Taei, “Ensemble Classifier for Eye State Classification using EEG Signals,” 2017. https://arxiv.org/abs/1709.08590
5. M. Nilashi et al., “Electroencephalography (EEG) eye state classification using learning vector quantization and bagged trees,” Heliyon, 2023. https://doi.org/10.1016/j.heliyon.2023.e15258
