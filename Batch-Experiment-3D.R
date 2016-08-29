#install.packages("devtools")
#library(devtools)
#install_github("mllg/batchtools")
#library(batchtools)
library(mlr)
library(BatchJobs)
library(BatchExperiments)
library(ROCR)
library(mboost)
library(caret)
library(pROC)
library(dplyr)

loadConfig()
getConfig()
#name ID an file direction
reg = makeExperimentRegistry(id = "mytest3D", packages = c("mlr", "BatchJobs", "BatchExperiments",
                                                           "ROCR", "mboost", "caret", "pROC", "dplyr"))

data <- function(simulations, 
                 samples=200,
                 predictors,
                 SNRx=2,
                 SNRy=10,
                 kappa=5){
  
 
  

  
  cubeDim <- round(predictors^(1/3))
 
  
  #initialise variables
  X <- array(0, c(simulations, samples, predictors))
  Xnoise <- array(0, c(simulations, samples, predictors))
  Y <- matrix(0, nrow=samples, ncol=simulations)
  Ynoise <- matrix(0, nrow=samples, ncol=simulations)
  yBin <- matrix(0, nrow=samples, ncol=simulations)
  yBinNoise <- matrix(0, nrow=samples, ncol=simulations)
  
  #simply use a gaussian spherical symmetry for the coefficient values
  #co-centred with the cube and SD=1/8 the cube dimension
  coeffs <- array(0, dim=c(cubeDim, cubeDim, cubeDim))
  volCentre <- c(cubeDim/2+.5, cubeDim/2+.5, cubeDim/2+.5)
  coeffSD <- as.integer(cubeDim/8)
  for (coefX in 1:cubeDim){
    for (coefY in 1:cubeDim){
      for (coefZ in 1:cubeDim){
        distance <- as.numeric(dist(rbind(volCentre, c(coefX, coefY, coefZ)), method = "euclidean"))
        coeffs[coefX, coefY, coefZ] <- dnorm(distance, mean=0, sd=coeffSD)
      }
    }
  }
  
  
  #generate dataset for each simulation
  for (simulation in 1:simulations) {
    #new seeds for reproducibility
    set.seed(1234 + simulation*samples*4) #multiply by four because we sample four times per iteration
    #generate the samples
    for (predictor in 1:predictors) {
      curDim <- rnorm(samples, 0, 1)
      X[simulation, , predictor] <- curDim
    
      noise <- rnorm(samples)
      #calculate the adj coefficient from variance of the predictor signal and the desired SNRx
      noiseCoeffX <- sqrt(var(curDim)/(SNRx * var(noise)))
      Xnoise[1, , predictor] <- X[1, , predictor] + noiseCoeffX*noise
    }
    #generate the response
    Y[,simulation] <- X[simulation, ,] %*% coeffs
    #generate some noise
    noise <- rnorm(samples)
    #calculate the adj coefficient from variance of the signal and the desired SNRy
    #varY<-varY+var(Y[,simulation])
    noiseCoeffY <- sqrt(var(Y[,simulation])/(SNRy * var(noise)))
    #generate the response with noise
    Ynoise[,simulation] <- Y[,simulation] + noiseCoeffY*noise
    #generate two-group response (kappa regulates the steepness and largely the overlap)
    yLogit <- 1/(1+exp(-kappa*Y[,simulation]))
    yBin[,simulation] <- rbinom(samples,1,yLogit)
    yLogitNoise <- 1/(1+exp(-kappa*Ynoise[,simulation]))
    yBinNoise[,simulation] <- rbinom(samples,1,yLogitNoise)
    
  }
  
  #save file containing continuous and binary responses, together with simulation parameters
  list (simulations = simulations, samples = samples, predictors = predictors, SNRx = SNRx,
        SNRy = SNRy, kappa = kappa, coeffs = coeffs,X = X, Xnoise = Xnoise, Y = Y, Ynoise = Ynoise,
        yBin = yBin, yBinNoise = yBinNoise)
 
  
  
}






#Add the problem
addProblem(reg, id = "mytest3D", dynamic = data, seed = 123)





################################################################################
################################################################################
################################################################################
################################################################################

sampledboosting.wrapper <- function(dynamic, sampleRatio ){
  
  
  
  
  
  if (length(commandArgs(trailingOnly = TRUE))>0) {
    #####read the sample number to test upon from the call
    args <- commandArgs(trailingOnly = TRUE)
    
    nOuterFolds <- as.numeric(args[1])  #integer number of outer folds to use
    redSteps <- as.numeric(args[2])     #integer number of iterations to run for the sampled boosting
    sampleRatio <- as.numeric(args[3])  #ratio of voxels TO REMOVE
    fixedMstop <- as.numeric(args[4])   #mstop to be used when not implementing cv early stopping
    fixedNu <- args[5]                  #shrinkage coefficient for boosting
    dynamic <- args[6]                 #dynamic to load
    localRun <- FALSE
  } else {
    ###### local testing
    nOuterFolds <- 10 #number of folds
    redSteps <- 100 #number of steps to run for the reduction in voxels
    sampleRatio = sampleRatio #ratio of voxels to REMOVE
    fixedMstop <- 100
    fixedNu <- 0.1
    dynamic <- dynamic
    localRun <- TRUE
  }
  
  ###########file types 
  fileTypePlain <- 1
  fileTypeVols <- 2
  
  #label for output
  labelOut <- ""
  addLabelOut <- ""
  

  n <- dynamic$samples  #number of cases (samples)
  # simulations is already called 'simulations' in the dynamic
  nVariables <- dynamic$predictors 
  dataX <- dynamic$X # X[simulation, sample, x]
  dataXnoise <- dynamic$Xnoise # Xnoise[simulation, sample, x]
  #make a copy of the matrix, to keep for the final test (untouched predictors)
  originalX <- dynamic$X
  originalXnoise <- dynamic$Xnoise
  simulations <- dynamic$simulations
  #### other variables that get loaded:
  #response variables:
  # Y
  # Ynoise
  # yBin
  # yBinNoise
  #...
  #####################################################################
  
  predModelList <- array(0,c(simulations, nOuterFolds, redSteps, nVariables))  #this keeps the list of unaltered models to produce intermediate predictions
  #for monitoring
  predModelListNoise <- array(0,c(simulations, nOuterFolds, redSteps, nVariables))  
  predModelListClass <- array(0,c(simulations, nOuterFolds, redSteps, nVariables))  #this keeps the list of unaltered models to produce intermediate predictions
  #for monitoring
  predModelListClassNoise <- array(0,c(simulations, nOuterFolds, redSteps, nVariables))
  
  # predModelListAdj <- as.array(0,c(simulations, redSteps, nVariables))  #the same, but adjusted (number of folds)
  offsetFinal <- array(0,c(simulations, nOuterFolds, redSteps))
  offsetFinalNoise <- array(0,c(simulations, nOuterFolds, redSteps))
  offsetFinalClass <- array(0,c(simulations, nOuterFolds, redSteps))
  offsetFinalClassNoise <- array(0,c(simulations, nOuterFolds, redSteps))
  
  removedVoxels <- vector("list", redSteps)  #this keeps track of the voxels set to zero at each reduction
  removedVoxelsNoise <- vector("list", redSteps)
  
  
  predictionVector <- array(0, c(simulations, redSteps, n))
  predictionVectorNoise <- array(0, c(simulations, redSteps, n))
  predictionVectorClass <- array(0, c(simulations, redSteps, n))
  predictionVectorClassNoise <- array(0, c(simulations, redSteps, n))
  
  coefNumbers <- rep(0, redSteps) #the number of coefficient in the final linear model at each reduction (for monitoring)
  coefNumbersNoise <- rep(0, redSteps)
  
  set.seed(1234) #for reproducibility
  
  for (simulation in 1:simulations){
    
    print(paste("Simulation n.", simulation))   #DEBUG
    
    X <- as.matrix(dataX[simulation, , ])
    Xnoise <- as.matrix(dataXnoise[simulation, , ])
    #training is always performed on noisy response variables
    y <- dynamic$Ynoise[, simulation]
    yClass <- dynamic$yBinNoise[, simulation]
    
    #create Outer folds (list of indices, one list per fold, which specify the test sets)
    indexOuterList <- createFolds(dynamic$Ynoise[, simulation], nOuterFolds) #sample(n)  
    
  
    for (kkk in 1:nOuterFolds){
      
      #print(paste("Fold n.", kkk))   #DEBUG
      
      indvecOuter <- indexOuterList[[kkk]]  #indices for the test set of this Outer fold
      trainXOuter <- X[-indvecOuter,]
      trainXnoiseOuter <- Xnoise[-indvecOuter,]
      trainyOuter <- y[-indvecOuter]
      trainyOuterClass <- yClass[-indvecOuter]
      testXOuter <- X[indvecOuter,]
      testXnoiseOuter <- Xnoise[indvecOuter,]
      testyOuter <- y[indvecOuter]
      testyOuterClass <- yClass[indvecOuter]
      
      modelList <- matrix(0,nrow = nOuterFolds, ncol = nVariables) #this is the matrix with model coefficients
      modelListNoise <- matrix(0,nrow = nOuterFolds, ncol = nVariables) 
      modelListClass <- matrix(0,nrow = nOuterFolds, ncol = nVariables) #this is the matrix with model coefficients
      modelListClassNoise <- matrix(0,nrow = nOuterFolds, ncol = nVariables) 
     
      
      #start with all predictors
      trainX <- trainXOuter
      trainXnoise <- trainXnoiseOuter
      trainXClass <- trainXOuter
      trainXClassNoise <- trainXnoiseOuter
      
      #iteration to (progressively) eliminate selected voxels to produce images
      for (reduction in 1:redSteps){
       
        
        #make boosting model with FIXED mstop 
        model <- glmboost(trainX, trainyOuter,  center = FALSE, control = boost_control(mstop = fixedMstop, trace = FALSE, nu = fixedNu))
        modelNoisy <- glmboost(trainXnoise, trainyOuter, center = FALSE, control = boost_control(mstop = fixedMstop, trace = FALSE, nu = fixedNu))
        modelClass <- glmboost(trainXClass, trainyOuterClass, center = FALSE, control = boost_control(mstop = fixedMstop, trace = FALSE, nu = fixedNu))
        modelClassNoisy <- glmboost(trainXClassNoise, trainyOuterClass, center = FALSE, control = boost_control(mstop = fixedMstop, trace = FALSE, nu = fixedNu))
        
        #CI <- confint(model)
        #extract coefficients and offset of models
        cfs <- coef(model)
        cfsNoise <- coef(modelNoisy)
        #offset <- offset + attr(coef(model), "offset")/nOuterFolds
        offsetFinal[simulation, kkk, reduction] <- attr(coef(model), "offset")
        offsetFinalNoise[simulation, kkk, reduction] <- attr(coef(modelNoisy), "offset")
        modelList[kkk,as.numeric(substr(names(cfs), 2, 100))] <- cfs #* curWeight ############# I DON'T THINK KKK IS NEEDED HERE ####################
        modelListNoise[kkk,as.numeric(substr(names(cfsNoise), 2, 100))] <- cfsNoise
        predModelList[simulation, kkk, reduction, ] <- modelList[kkk,] ############# I DON'T THINK KKK IS NEEDED HERE ####################
        predModelListNoise[simulation, kkk, reduction, ] <- modelListNoise[kkk,]
        
        cfsClass <- coef(modelClass)
        cfsClassNoise <- coef(modelClassNoisy)
        #offsetClass <- offsetClass + attr(coef(modelClass), "offset")/nOuterFolds
        offsetFinalClass[simulation, kkk, reduction] <- attr(coef(modelClass), "offset")
        offsetFinalClassNoise[simulation, kkk, reduction] <- attr(coef(modelClassNoisy), "offset")
        modelListClass[kkk,as.numeric(substr(names(cfsClass), 2, 100))] <- cfsClass #* curWeight #history of coefficient values as selected at each reduction
        modelListClassNoise[kkk,as.numeric(substr(names(cfsClassNoise), 2, 100))] <- cfsClassNoise #* curWeight #history of coefficient values as selected at each reduction
        predModelListClass[simulation, kkk, reduction, ] <- modelListClass[kkk,] ############# I DON'T THINK KKK IS NEEDED HERE ##################
        predModelListClassNoise[simulation, kkk, reduction, ] <- modelListClassNoise[kkk,] ############# I DON'T THINK KKK IS NEEDED HERE ##################
        
        #build the final linear models for the current reduction
        lTempModel <- rep(0, nVariables)
        lTempOffset <- 0
        lTempModelNoise <- rep(0, nVariables)
        lTempOffsetNoise <- 0
        lTempModelClass <- rep(0, nVariables)
        lTempOffsetClass <- 0
        lTempModelClassNoise <- rep(0, nVariables)
        lTempOffsetClassNoise <- 0
        for (tempReduction in 1:reduction) {
          #basic model
          tempSingleModel <- predModelList[simulation, kkk, tempReduction, ]
          lTempModel <- lTempModel + tempSingleModel/reduction
          lTempOffset <- lTempOffset + offsetFinal[simulation, kkk, tempReduction]/reduction
          #noisy model
          tempSingleModelNoise <- predModelListNoise[simulation, kkk, tempReduction, ]
          lTempModelNoise <- lTempModelNoise + tempSingleModelNoise/reduction
          lTempOffsetNoise <- lTempOffsetNoise + offsetFinalNoise[simulation, kkk, tempReduction]/reduction
          #classification model
          tempSingleModelClass <- predModelListClass[simulation, kkk, tempReduction, ]
          lTempModelClass <- lTempModelClass + tempSingleModelClass/reduction
          lTempOffsetClass <- lTempOffsetClass + offsetFinalClass[simulation, kkk, tempReduction]/reduction
          #classification noisy model
          tempSingleModelClassNoise <- predModelListClassNoise[simulation, kkk, tempReduction, ]
          lTempModelClassNoise <- lTempModelClassNoise + tempSingleModelClassNoise/reduction
          lTempOffsetClassNoise <- lTempOffsetClassNoise + offsetFinalClassNoise[simulation, kkk, tempReduction]/reduction
        }
        
        # make the temporary predictions from the linear model based on coefficients, to later evaluate how final performance changes at each reduction step
        #lTempModel <- colSums(imgModelList)
        testXtemp <- matrix(originalX[simulation, indvecOuter, ], nrow = length(indvecOuter))  #use the original, unreduced design matrix!
        testXtempNoise <- matrix(originalXnoise[simulation, indvecOuter, ], nrow = length(indvecOuter))  
        predictionVector[simulation, reduction, indvecOuter] <- testXtemp %*% lTempModel + lTempOffset
        predictionVectorNoise[simulation, reduction, indvecOuter] <- testXtempNoise %*% lTempModelNoise + lTempOffsetNoise
        predictionVectorClass[simulation, reduction, indvecOuter] <- testXtemp %*% lTempModelClass + lTempOffsetClass
        predictionVectorClassNoise[simulation, reduction, indvecOuter] <- testXtempNoise %*% lTempModelClassNoise + lTempOffsetClassNoise
        
        #       #store the number of non-zero coefficients so far (for monitoring)
        #       coefNumbers[reduction] <- length(which(lTempModel!=0))
        
        #using some random coef which are not 0 in the current model, find predictors to be temporarily excluded
        voxelNum = round(length(which(colSums(modelList)!=0)) * sampleRatio)  #calculate number of predictors to remove
        voxelNumNoise = round(length(which(colSums(modelListNoise)!=0)) * sampleRatio)
        mostSelInd <- sample(which(colSums(modelList)!=0), voxelNum) #calculate random indices
        mostSelIndNoise <- sample(which(colSums(modelListNoise)!=0), voxelNumNoise) #calculate random indices
        voxelNumClass = round(length(which(colSums(modelListClass)!=0)) * sampleRatio)  #calculate number of predictors to remove
        mostSelIndClass <- sample(which(colSums(modelListClass)!=0), voxelNumClass) #calculate random indices
        voxelNumClassNoise = round(length(which(colSums(modelListClassNoise)!=0)) * sampleRatio)  #calculate number of predictors to remove
        mostSelIndClassNoise <- sample(which(colSums(modelListClassNoise)!=0), voxelNumClassNoise) #calculate random indices
        
        #print(paste("MostSel",mostSelInd))            ####### DEBUG ###########
        #print(paste("MostSelClass",mostSelIndClass))  ####### DEBUG ###########
        
        #start with all predictors
        trainX <- trainXOuter
        trainXnoise <- trainXnoiseOuter
        trainXClass <- trainXOuter
        trainXClassNoise <- trainXnoiseOuter
        #remove the selected ones
        for (indRow in 1:nrow(trainXOuter)) {
          #set the selected predictors to zero for the next iteration
          trainX[indRow,mostSelInd] <- 0
          trainXnoise[indRow,mostSelIndNoise] <- 0
          trainXClass[indRow,mostSelIndClass] <- 0
          trainXClassNoise[indRow,mostSelIndClassNoise] <- 0
        }
        
      } #end of reduction block
      
    } #end of fold block
    
  } #end of simulation block
  
  
  #make a name for the output file, using input file name and parameters passed
  #outLabel <- paste(dynamic,"_OUT_", nOuterFolds, "_", redSteps, "_", sampleRatio, "_", fixedMstop, "_", fixedNu, ".rda", sep = "")
  #save all variables from input file, parameters and output
  list(originalY = dynamic$Y, y = y, Ynoise = dynamic$Ynoise, yBin = dynamic$yBin,yBinNoise = dynamic$yBinNoise, 
       originalX = originalX, originalXnoise = originalXnoise, coeffs = dynamic$coeffs, 
       predictors = dynamic$predictors, kappa = dynamic$kappa, samples = dynamic$samples, 
       simulations = simulations, nOuterFolds = nOuterFolds, redSteps = redSteps, 
       sampleRatio = sampleRatio, fixedMstop = fixedMstop, fixedNu = fixedNu,
       offsetFinal = offsetFinal, predModelList = predModelList, offsetFinalClass = offsetFinalClass,
       predModelListClass = predModelListClass, predictionVector = predictionVector, 
       predictionVectorClass = predictionVectorClass,offsetFinalNoise = offsetFinalNoise, 
       predModelListNoise = predModelListNoise, predictionVectorNoise = predictionVectorNoise, 
       offsetFinalClassNoise = offsetFinalClassNoise, predModelListClassNoise = predModelListClassNoise,
       predictionVectorClassNoise = predictionVectorClassNoise)
  
  
} #end function

addAlgorithm(reg, id = "sampledboosting", fun = sampledboosting.wrapper)




################################################
################################################
################################################
################################################










# Define problem parameters:
pars = list(simulations = 200, predictors = 8000)
mytest3D.design = makeDesign("mytest3D", exhaustive = pars)

# Define sampledboosting parameters:
pars = list(sampleRatio = c(0.1, 0.5, 0.9))
sampledboosting.design = makeDesign("sampledboosting", exhaustive = pars)






# Add experiments to the registry:
# Use  previously defined experimental designs.
addExperiments(reg, prob.designs = mytest3D.design,
               algo.designs = sampledboosting.design,
               repls = 1) # usually you would set repls to 100 or more.




# Optional: Short summary over problems and algorithms.
summarizeExperiments(reg)

# Submit the jobs to the batch system
submitJobs(reg, resources = list(walltime = 60L*60L*168L, memory = 250000L), max.retries = 10L)
