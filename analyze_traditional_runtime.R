# Runtime analysis of the traditional analysis pipeline
# Setting working directory
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))



fnames <- list.files(pattern = "timing.txt$", recursive = TRUE)

run_times <- c()
for(filenumber in 1:length(fnames)){
  run_time <- read.table(fnames[filenumber])
  if(any(!is.na(run_time))){
    run_times <- c(run_times, as.vector(t(run_time)))
  } else {
    next
  }
}
run_times
experiments <- gsub("_timing.txt", "", gsub("results/.*/", "", fnames))
experiments <- rep(experiments, each=3)

execution_type <- rep(c("FeatureExtraction", "ModelTraining", "ModelPrediction"), 90)


dataset <- c()
extraction_method <- c()
model <- c()
Parameters <- c()
for(experiment in experiments){
  
  # Dataset
  if(grepl("clothes", experiment, fixed = TRUE)){
    dataset <- c(dataset, "FashionMnist")
  } else {
    dataset <- c(dataset, "LabelledFacesInTheWild")
  }
  # Extraction method
  if(grepl("CH", experiment, fixed = TRUE)){
    extraction_method <- c(extraction_method, "ColorHistogram")
  } else {
    extraction_method <- c(extraction_method, "SIFT")
  }
  # Model
  if(grepl("SupportVectorMachine", experiment, fixed = TRUE)){
    model <- c(model, "SVM")
  } else if(grepl("RandomForrest", experiment, fixed = TRUE)) {
    model <- c(model, "RandomForrest")
  } else if(grepl("QuadraticDiscriminantAnalysis", experiment, fixed = TRUE)) {
    model <- c(model, "QDA")
  } else if(grepl("MLPClassifier", experiment, fixed = TRUE)) {
    model <- c(model, "MLP")
  } else if(grepl("NaiveBayes", experiment, fixed = TRUE)) {
    model <- c(model, "NaiveBayes")
  } else if(grepl("DecisionTree", experiment, fixed = TRUE)) {
    model <- c(model, "DecisionTree")
  }
  # Parameters
  parameter_settings <- strsplit(experiment, "_")[[1]]
  parameter_settings <- paste0(parameter_settings[4:length(parameter_settings)], collapse="_")
  Parameters <- c(Parameters, parameter_settings)
}

performance <- data.frame(execution_type= rep(c("FeatureExtraction", "ModelTraining", "ModelPrediction"), 90),
                          dataset=dataset, extraction_method=extraction_method, model=model, Parameters=Parameters, runtime=run_times)
performance


library(ggplot2)
p_feature_extraction <- ggplot(data=performance[which(performance$execution_type == "FeatureExtraction"),], aes(x=model, y=runtime, fill=Parameters)) +
  geom_bar(position="dodge", stat="identity") + facet_wrap(~ dataset + extraction_method + execution_type, ncol=2) + 
  guides(fill=guide_legend(ncol=1)) + ggtitle("Feature extraction runtime in seconds") + xlab("Model") + ylab("Runtime in seconds")
p_feature_extraction

p_model_training <- ggplot(data=performance[which(performance$execution_type == "ModelTraining"),], aes(x=model, y=runtime, fill=Parameters)) +
  geom_bar(position="dodge", stat="identity") + facet_wrap(~ dataset + extraction_method, ncol=2) + 
  guides(fill=guide_legend(ncol=1)) + ggtitle("Model training runtime in seconds") + xlab("Model") + ylab("Runtime in seconds")
p_model_training

p_model_prediction <- ggplot(data=performance[which(performance$execution_type == "ModelPrediction"),], aes(x=model, y=runtime, fill=Parameters)) +
  geom_bar(position="dodge", stat="identity") + facet_wrap(~ dataset + extraction_method + execution_type, ncol=2) + 
  guides(fill=guide_legend(ncol=1)) + ggtitle("Model prediction runtime in seconds") + xlab("Model") + ylab("Runtime in seconds")
p_model_prediction


ggsave("results/Feature_Extraction_Runtime.png", p_feature_extraction, width = 30, height = 20, units = "cm")
ggsave("results/Model_Runtime_Runtime.png", p_model_training, width = 30, height = 20, units = "cm")
ggsave("results/Model_Prediction_Runtime.png", p_model_prediction, width = 30, height = 20, units = "cm")

