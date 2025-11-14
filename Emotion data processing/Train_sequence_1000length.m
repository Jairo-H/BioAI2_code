clear
close all
clc

seqOut = 1;

addpath 'Major revision'\'Neural networks'\
run TCNs.m



waveform = 1;
sptrg = 0;
dataFile = './All_Data/Dataset_gesture';
SignalRoot = '.\All_Data\datasetSignal\SignalNorm';
if waveform
    % lgraph = importdata("./network/layergraph_gesture_0923_waveform_24.mat");
    data = load(fullfile(SignalRoot,'/waveform_Train'));
    Dataset_Data_Waveform_Train = data.Dataset_Data_Voltage_Total;
    data = load(fullfile(SignalRoot,'/charge_Train'));
    Dataset_Data_Charge_Train = data.Dataset_Data_Charge_Total;
    data = load(fullfile(SignalRoot,'/label_Train'));
    Dataset_Label_Train = data.Dataset_Label_Total;
    data = load(fullfile(SignalRoot,'/waveform_Verify'));
    Dataset_Data_Waveform_Verify = data.Dataset_Data_Voltage_Total;
    data = load(fullfile(SignalRoot,'/charge_Verify'));
    Dataset_Data_Charge_Verify = data.Dataset_Data_Charge_Total;
    data = load(fullfile(SignalRoot,'/label_Verify'));
    Dataset_Label_Verify = data.Dataset_Label_Total;
end
% l_data = size(Dataset_Data_Sptrg,1);
% l_Train = round(l_data/6*5);
% l_Verify = round(l_Train+l_data/12);
randIndex = randperm(size(Dataset_Label_Train,1));
% height = size(Dataset_Data_Sptrg,2);
% width = size(Dataset_Data_Sptrg,3);
% Dataset_Data_Sptrg_new = reshape(Dataset_Data_Sptrg,[height,width,1,l_data]);
% for h=1:height
%     for w=1:width
%         for n=1:l_data
%             Dataset_Data_Sptrg_new(h,w,1,n) = Dataset_Data_Sptrg(n,h,w);
%         end
%     end
% end
%% 打乱顺序
% Dataset_Data_Sptrg = Dataset_Data_Sptrg_new(:,:,:,randIndex); 
% Dataset_Data_Sptrm = Dataset_Data_Sptrm(randIndex,:);
Dataset_Data_Waveform_Train = Dataset_Data_Waveform_Train(randIndex,:);
Dataset_Data_Charge_Train = Dataset_Data_Charge_Train(randIndex,:);
Dataset_Label_Train = Dataset_Label_Train(randIndex,:);
% Dataset_Name = Dataset_Name(randIndex,:,:);

%% from double to arrayDataStore
TrainData1 = Dataset_Data_Waveform_Train;
TrainData2 = Dataset_Data_Charge_Train;
TrainLabel = categorical(Dataset_Label_Train);
TrainCell = cell(length(TrainData1),1);
TrainLabelCell = cell(length(TrainData1),1);
for i = 1:length(Dataset_Data_Waveform_Train)
    dataWave = Dataset_Data_Waveform_Train(i,:);
    dataWave = dataWave/max(dataWave); %Normalization
    dataCharge = Dataset_Data_Charge_Train(i,:);
    dataCharge = interp1(1:length(dataCharge), dataCharge, 0:length(dataCharge)/999:length(dataCharge),'spline');
    dataCharge = dataCharge/max(dataCharge); %Normalization
    dataMat = [dataWave', dataCharge']';
    dataCell = mat2cell(dataMat,2,1000);
    TrainCell{i,1} = dataCell{1};
    noAct = dataCharge < 0.1; % Choose label without action
    dataCell = mat2cell(repelem(TrainLabel(i),1000),1,1000);
    dataCell{1}(noAct) = "None";% Choose label without action
    TrainLabelCell{i,1} = dataCell{1};
end

% TrainName = Dataset_Name(1:l_Train,:);
% XTrain1 = arrayDatastore(TrainData1);
% XTrain2 = arrayDatastore(TrainData2);
% YTrain = arrayDatastore(TrainLabel);
% dsTrain = combine(XTrain2,XTrain1,YTrain);

VerifyData1 = Dataset_Data_Waveform_Verify;
VerifyData2 = Dataset_Data_Charge_Verify;
VerifyLabel = categorical(Dataset_Label_Verify);

VerifyCell = cell(length(VerifyData1),1);
VerifyLabelCell = cell(length(VerifyData1),1);

for i = 1:length(Dataset_Data_Waveform_Verify)
    dataWave = Dataset_Data_Waveform_Verify(i,:);
    dataWave = dataWave/max(dataWave); %Normalization
    dataCharge = Dataset_Data_Charge_Verify(i,:);
    dataCharge = interp1(1:length(dataCharge), dataCharge, 0:length(dataCharge)/999:length(dataCharge),'spline');
    dataCharge = dataCharge/max(dataCharge); %Normalization
    dataMat = [dataWave', dataCharge']';
    dataCell = mat2cell(dataMat,2,1000);
    VerifyCell{i,1} = dataCell{1};
    noAct = dataCharge < 0.1; % Choose label without action
    dataCell = mat2cell(repelem(VerifyLabel(i),1000),1,1000);
    dataCell{1}(noAct) = "None";% Choose label without action
    VerifyLabelCell{i,1} = dataCell{1};
end

filterSize = 5;
numFilters = 32;
XTrain = TrainCell;
XValidation = VerifyCell;

if seqOut
    TTrain = TrainLabelCell;
    TValidation = VerifyLabelCell;
else
    TTrain = TrainLabel;
    TValidation = VerifyLabel;
end

numChannels = size(XTrain{1},2);
classNames = categories(TrainLabel);
numClasses = numel(classNames);

% layers = [ ...
%     sequenceInputLayer(numChannels)
%     convolution1dLayer(filterSize,numFilters,Padding="causal")
%     reluLayer
%     layerNormalizationLayer
%     convolution1dLayer(filterSize,2*numFilters,Padding="causal")
%     reluLayer
%     layerNormalizationLayer
%     globalAveragePooling1dLayer
%     fullyConnectedLayer(numClasses)
%     softmaxLayer];

options = trainingOptions("adam", ...
     'ExecutionEnvironment','gpu', ...
    MaxEpochs=60, ...
    InitialLearnRate=0.01, ...
    SequencePaddingDirection="left", ...
    ValidationData={XValidation,TValidation}, ...
    Plots="training-progress", ...
    Verbose=false);

net = trainNetwork(XTrain,TTrain,lgraph,options);


% XVerify1 = arrayDatastore(VerifyData1);
% XVerify2 = arrayDatastore(VerifyData2);
% YVerify = arrayDatastore(VerifyLabel);
% dsVerify = combine(XVerify2,XVerify1,YVerify);


%% Training options
% miniBatchSize = 10;
% 
% options = trainingOptions('sgdm', ...
%     'ExecutionEnvironment','gpu', ...
%     'MaxEpochs',100, ...
%     'MiniBatchSize',miniBatchSize, ...
%     'ValidationData',{VerifyCell,VerifyLabel}, ...
%     'Shuffle','every-epoch', ...
%     'Verbose',false, ...
%     'Plots','training-progress');
% 
% %% Start Train
% [net,info] = trainNetwork(TrainCell,TrainLabel,lgraph,options);

% figure
% plot(info.ValidationAccuracy(50:100:end))
% plot(info.ValidationLoss(50:100:end))
% VA = info.ValidationAccuracy(50:100:end);
% VL = info.ValidationLoss(50:100:end);