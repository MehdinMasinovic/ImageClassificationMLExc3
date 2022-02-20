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
parameters <- c()
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
    model <- c(model, "ColorHistogram")
  } else if(grepl("RandomForrest", experiment, fixed = TRUE)) {
    model <- c(model, "RandomForrest")
  } else if(grepl("QuadraticDiscriminantAnalysis", experiment, fixed = TRUE)) {
    model <- c(model, "QuadraticDiscriminantAnalysis")
  } else if(grepl("MLPClassifier", experiment, fixed = TRUE)) {
    model <- c(model, "MLPClassifier")
  } else if(grepl("NaiveBayes", experiment, fixed = TRUE)) {
    model <- c(model, "NaiveBayes")
  } else if(grepl("DecisionTree", experiment, fixed = TRUE)) {
    model <- c(model, "DecisionTree")
  }
  # Parameters
  parameter_settings <- strsplit(experiment, "_")[[1]]
  parameter_settings <- paste0(parameter_settings[4:length(parameter_settings)], collapse="_")
  parameters <- c(parameters, parameter_settings)
}

performance <- data.frame(execution_type= rep(c("FeatureExtraction", "ModelTraining", "ModelPrediction"), 90),
                          dataset=dataset, extraction_method=extraction_method, model=model, parameters=parameters, runtime=run_times)
performance


library(ggplot2)
ggplot(data=performance, aes(x=execution_type, y=runtime, fill=model)) +
  geom_bar(position="dodge", stat="identity") + facet_wrap(~  extraction_method)


  # scale_color_viridis(discrete = TRUE) +
  # ggtitle("Popularity of American names in the previous 30 years") +
  # theme_ipsum() +
  # ylab("Number of babies born")

