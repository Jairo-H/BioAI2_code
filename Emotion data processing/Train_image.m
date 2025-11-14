
ifclear = 1;

if ifclear
    clear
    parentDir = '.\All_data\datasetImage\ImageNorm_with_force\Train'; % Header needs to be changed
    allImages = imageDatastore(parentDir, ...
        "IncludeSubfolders",true, ...
        "LabelSource","foldernames");
    % Get the list of file names
    fileNames = {allImages.Files};
    numImages = numel(fileNames);
    allImages = shuffle(allImages);
    imgsTrain = allImages;
    disp("Number of training images: "+num2str(numel(imgsTrain.Files)))

    parentDir = '.\All_data\datasetImage\ImageNorm_with_force\Test';
    imgsValidation = imageDatastore(parentDir, ...
        "IncludeSubfolders",true, ...
        "LabelSource","foldernames");
    % valabel = zeros(length(fileNames{1}), 1, 3);
    % for i = 1:length(allImages.Labels)
    %     tmp = char(string(allImages.Labels(i)));
    %     v=double(tmp(1))-48;
    %     a=double(tmp(2))-48;
    %     g=double(tmp(3))-48;
    %     va = [v a g];
    %     valabel(i,1,:) = (va-1)./8;
    % end
    % labelstore = arrayDatastore(valabel);
    
    % 预处理数据    
    % height = 224;
    % width = 224;
    % Dataset_Data_img = zeros([height,width,3,length(fileNames{1})]);
    % for i = 1:numel(fileNames)
    %     % 加载原始图像和马赛克图像
    %     for j = 1:length(fileNames{1})
    %         imgOriginal = imread(fileNames{1}{i});
    %         Dataset_Data_img(:,:,:,j) = imgOriginal(:,:,:);
    %     end
    % end
    % randIndex = randperm(size(Dataset_Data_img,4));
    % l_Train = round(length(fileNames{1})*0.8);
    % TotalData = Dataset_Data_img(:,:,:,randIndex);
    % TotalLabel = valabel(randIndex,:);
    % TrainData = (TotalData(:,:,:,1:l_Train));
    % TrainLabel = (valabel(1:l_Train,:));
    % VerifyData = (TotalData(:,:,:,l_Train+1:end));
    % VerifyLabel = (valabel(l_Train+1:end,:));
    
    % save("0411_Data\ImageVA\TrainData3","TrainData");
    % save("0411_Data\ImageVA\TrainLabel","TrainLabel");
    % save("0411_Data\ImageVA\VerifyData","VerifyData");
    % save("0411_Data\ImageVA\VerifyLabel","VerifyLabel");
else
    % TrainData = importdata("0411_Data\ImageVA\TrainData3.mat");
    % TrainLabel = importdata("0411_Data\ImageVA\TrainLabel.mat");
    % VerifyData = importdata("0411_Data\ImageVA\VerifyData.mat");
    % VerifyLabel = importdata("0411_Data\ImageVA\VerifyLabel.mat");
    % load("0711_Data\imgsData\imgsTrain.mat")
    % load("0711_Data\imgsData\imgsValidation.mat")
    load(".\All_Data\networkImageNorm89.65divided\imgsTrain.mat")
    load("All_Data\networkImageNorm89.65divided\imgsValidation.mat")
end
ifTrain = 1;

if ifTrain
    
    % disp("Number of validation images: "+num2str(numel(imgsValidation)))
    
    lgraph = importdata("./All_data/network/layergraph_img_0716_ResNet18_change.mat");
    % % lgraph = importdata("./network/layergraph_img_0821_ResNet18_multihead.mat");
    % lgraph = layerGraph(net);
    numberOfLayers = numel(lgraph.Layers);
    
    % newDropoutLayer = dropoutLayer(0.6,"Name","new_Dropout");
    % lgraph = replaceLayer(lgraph,"pool5-drop_7x7_s1",newDropoutLayer);
    
    % numClasses = numel(categories(imgsTrain.Labels));
    % newConnectedLayer = fullyConnectedLayer(numClasses,"Name","new_fc", ...
    %     "WeightLearnRateFactor",5,"BiasLearnRateFactor",5);
    % lgraph = replaceLayer(lgraph,"fc1000",newConnectedLayer);
    % 
    % newClassLayer = classificationLayer("Name","new_classoutput");
    % lgraph = replaceLayer(lgraph,"ClassificationLayer_predictions",newClassLayer);
    % lgraph = removeLayers(lgraph,"dropout");
    % lgraph = connectLayers(lgraph,"pool5","new_fc");
    % newConnectedLayer = fullyConnectedLayer(3,"Name","new_fc");
    % lgraph = replaceLayer(lgraph,"VA_fc_layer",newConnectedLayer);
    lgraph.Layers(end-4:end)
    
    % options = trainingOptions("adam", ...
    %     MiniBatchSize=15, ...
    %     MaxEpochs=50, ...
    %     InitialLearnRate=1e-3, ...
    %     ValidationData={VerifyData,VerifyLabel}, ...
    %     ValidationFrequency=20, ...
    %     LearnRateSchedule='piecewise', ...
    %     LearnRateDropFactor=0.3, ...
    %     LearnRateDropPeriod=20, ...
    %     Verbose=1, ...
    %     Plots="training-progress");
    options = trainingOptions("sgdm", ...
    MiniBatchSize=15, ...
    MaxEpochs=10, ...
    InitialLearnRate=1e-3, ...
    ValidationData=imgsValidation, ...
    ValidationFrequency=20, ...
    LearnRateSchedule='piecewise', ...
    LearnRateDropFactor=0.3, ...
    LearnRateDropPeriod=20, ...
    Verbose=1, ...
    Plots="training-progress");
    
    % trainedGN = trainNetwork(TrainData, TrainLabel,lgraph,options);
    trainedGN = trainNetwork(imgsTrain,lgraph,options);
else
    [YPred,~] = classify(trainedGN,imgsValidation);
    accuracy = mean(YPred==imgsValidation.Labels);
    disp("ResNet 18 Accuracy: "+num2str(100*accuracy)+"%")
    confusionchart(imgsValidation.Labels, YPred)
end