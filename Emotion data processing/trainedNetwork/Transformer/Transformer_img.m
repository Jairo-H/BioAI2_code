ImageSize = 224;
numClasses = 3*8;

net = dlnetwork;

tempNet = [
    imageInputLayer([ImageSize ImageSize 3],"Name","imageinput","Normalization","zscore")
    patchEmbeddingLayer([16 16],ImageSize*2,"Name","embedding","SpatialFlattenMode","row-major")
    embeddingConcatenationLayer("Name","clsembed_concat")];
net = addLayers(net,tempNet);

tempNet = positionEmbeddingLayer(ImageSize*2,197,"Name","posembed_input");
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","add")
    dropoutLayer(0.1,"Name","dropout")];
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock1_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock1_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock1_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock1_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock1_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock1_conv1d1")
    geluLayer("Name","encoderblock1_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock1_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock1_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock1_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock1_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock2_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock2_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock2_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock2_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock2_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock2_conv1d1")
    geluLayer("Name","encoderblock2_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock2_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock2_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock2_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock2_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock3_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock3_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock3_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock3_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock3_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock3_conv1d1")
    geluLayer("Name","encoderblock3_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock3_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock3_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock3_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock3_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock4_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock4_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock4_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock4_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock4_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock4_conv1d1")
    geluLayer("Name","encoderblock4_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock4_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock4_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock4_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock4_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock5_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock5_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock5_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock5_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock5_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock5_conv1d1")
    geluLayer("Name","encoderblock5_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock5_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock5_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock5_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock5_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock6_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock6_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock6_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock6_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock6_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock6_conv1d1")
    geluLayer("Name","encoderblock6_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock6_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock6_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock6_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock6_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock7_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock7_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock7_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock7_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock7_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock7_conv1d1")
    geluLayer("Name","encoderblock7_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock7_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock7_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock7_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock7_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock8_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock8_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock8_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock8_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock8_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock8_conv1d1")
    geluLayer("Name","encoderblock8_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock8_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock8_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock8_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock8_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock9_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock9_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock9_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock9_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock9_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock9_conv1d1")
    geluLayer("Name","encoderblock9_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock9_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock9_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock9_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock9_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock10_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock10_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock10_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock10_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock10_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock10_conv1d1")
    geluLayer("Name","encoderblock10_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock10_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock10_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock10_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock10_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock11_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock11_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock11_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock11_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock11_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock11_conv1d1")
    geluLayer("Name","encoderblock11_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock11_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock11_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock11_dropout3")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock11_add2");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock12_layernorm1","Epsilon",1e-06)
    selfAttentionLayer(8,ImageSize*2,"Name","encoderblock12_mha","DropoutProbability",0.1,"NumValueChannels",ImageSize*2,"OutputSize",ImageSize*2)
    dropoutLayer(0.1,"Name","encoderblock12_dropout1")];
net = addLayers(net,tempNet);

tempNet = additionLayer(2,"Name","encoderblock12_add1");
net = addLayers(net,tempNet);

tempNet = [
    layerNormalizationLayer("Name","encoderblock12_layernorm2","Epsilon",1e-06)
    convolution1dLayer(1,ImageSize*8,"Name","encoderblock12_conv1d1")
    geluLayer("Name","encoderblock12_gelu","Approximation","tanh")
    dropoutLayer(0.1,"Name","encoderblock12_dropout2")
    convolution1dLayer(1,ImageSize*2,"Name","encoderblock12_conv1d2")
    dropoutLayer(0.1,"Name","encoderblock12_dropout3")];
net = addLayers(net,tempNet);

tempNet = [
    additionLayer(2,"Name","encoderblock12_add2")
    layerNormalizationLayer("Name","encoder_norm","Epsilon",1e-06)
    indexing1dLayer("first","Name","cls_index")
    fullyConnectedLayer(numClasses,"Name","head")
    softmaxLayer("Name","softmax")];
net = addLayers(net,tempNet);

% 清理辅助变量
clear tempNet;

net = connectLayers(net,"clsembed_concat","posembed_input");
net = connectLayers(net,"clsembed_concat","add/in2");
net = connectLayers(net,"posembed_input","add/in1");
net = connectLayers(net,"dropout","encoderblock1_layernorm1");
net = connectLayers(net,"dropout","encoderblock1_add1/in2");
net = connectLayers(net,"encoderblock1_dropout1","encoderblock1_add1/in1");
net = connectLayers(net,"encoderblock1_add1","encoderblock1_layernorm2");
net = connectLayers(net,"encoderblock1_add1","encoderblock1_add2/in2");
net = connectLayers(net,"encoderblock1_dropout3","encoderblock1_add2/in1");
net = connectLayers(net,"encoderblock1_add2","encoderblock2_layernorm1");
net = connectLayers(net,"encoderblock1_add2","encoderblock2_add1/in2");
net = connectLayers(net,"encoderblock2_dropout1","encoderblock2_add1/in1");
net = connectLayers(net,"encoderblock2_add1","encoderblock2_layernorm2");
net = connectLayers(net,"encoderblock2_add1","encoderblock2_add2/in2");
net = connectLayers(net,"encoderblock2_dropout3","encoderblock2_add2/in1");
net = connectLayers(net,"encoderblock2_add2","encoderblock3_layernorm1");
net = connectLayers(net,"encoderblock2_add2","encoderblock3_add1/in2");
net = connectLayers(net,"encoderblock3_dropout1","encoderblock3_add1/in1");
net = connectLayers(net,"encoderblock3_add1","encoderblock3_layernorm2");
net = connectLayers(net,"encoderblock3_add1","encoderblock3_add2/in2");
net = connectLayers(net,"encoderblock3_dropout3","encoderblock3_add2/in1");
net = connectLayers(net,"encoderblock3_add2","encoderblock4_layernorm1");
net = connectLayers(net,"encoderblock3_add2","encoderblock4_add1/in2");
net = connectLayers(net,"encoderblock4_dropout1","encoderblock4_add1/in1");
net = connectLayers(net,"encoderblock4_add1","encoderblock4_layernorm2");
net = connectLayers(net,"encoderblock4_add1","encoderblock4_add2/in2");
net = connectLayers(net,"encoderblock4_dropout3","encoderblock4_add2/in1");
net = connectLayers(net,"encoderblock4_add2","encoderblock5_layernorm1");
net = connectLayers(net,"encoderblock4_add2","encoderblock5_add1/in2");
net = connectLayers(net,"encoderblock5_dropout1","encoderblock5_add1/in1");
net = connectLayers(net,"encoderblock5_add1","encoderblock5_layernorm2");
net = connectLayers(net,"encoderblock5_add1","encoderblock5_add2/in2");
net = connectLayers(net,"encoderblock5_dropout3","encoderblock5_add2/in1");
net = connectLayers(net,"encoderblock5_add2","encoderblock6_layernorm1");
net = connectLayers(net,"encoderblock5_add2","encoderblock6_add1/in2");
net = connectLayers(net,"encoderblock6_dropout1","encoderblock6_add1/in1");
net = connectLayers(net,"encoderblock6_add1","encoderblock6_layernorm2");
net = connectLayers(net,"encoderblock6_add1","encoderblock6_add2/in2");
net = connectLayers(net,"encoderblock6_dropout3","encoderblock6_add2/in1");
net = connectLayers(net,"encoderblock6_add2","encoderblock7_layernorm1");
net = connectLayers(net,"encoderblock6_add2","encoderblock7_add1/in2");
net = connectLayers(net,"encoderblock7_dropout1","encoderblock7_add1/in1");
net = connectLayers(net,"encoderblock7_add1","encoderblock7_layernorm2");
net = connectLayers(net,"encoderblock7_add1","encoderblock7_add2/in2");
net = connectLayers(net,"encoderblock7_dropout3","encoderblock7_add2/in1");
net = connectLayers(net,"encoderblock7_add2","encoderblock8_layernorm1");
net = connectLayers(net,"encoderblock7_add2","encoderblock8_add1/in2");
net = connectLayers(net,"encoderblock8_dropout1","encoderblock8_add1/in1");
net = connectLayers(net,"encoderblock8_add1","encoderblock8_layernorm2");
net = connectLayers(net,"encoderblock8_add1","encoderblock8_add2/in2");
net = connectLayers(net,"encoderblock8_dropout3","encoderblock8_add2/in1");
net = connectLayers(net,"encoderblock8_add2","encoderblock9_layernorm1");
net = connectLayers(net,"encoderblock8_add2","encoderblock9_add1/in2");
net = connectLayers(net,"encoderblock9_dropout1","encoderblock9_add1/in1");
net = connectLayers(net,"encoderblock9_add1","encoderblock9_layernorm2");
net = connectLayers(net,"encoderblock9_add1","encoderblock9_add2/in2");
net = connectLayers(net,"encoderblock9_dropout3","encoderblock9_add2/in1");
net = connectLayers(net,"encoderblock9_add2","encoderblock10_layernorm1");
net = connectLayers(net,"encoderblock9_add2","encoderblock10_add1/in2");
net = connectLayers(net,"encoderblock10_dropout1","encoderblock10_add1/in1");
net = connectLayers(net,"encoderblock10_add1","encoderblock10_layernorm2");
net = connectLayers(net,"encoderblock10_add1","encoderblock10_add2/in2");
net = connectLayers(net,"encoderblock10_dropout3","encoderblock10_add2/in1");
net = connectLayers(net,"encoderblock10_add2","encoderblock11_layernorm1");
net = connectLayers(net,"encoderblock10_add2","encoderblock11_add1/in2");
net = connectLayers(net,"encoderblock11_dropout1","encoderblock11_add1/in1");
net = connectLayers(net,"encoderblock11_add1","encoderblock11_layernorm2");
net = connectLayers(net,"encoderblock11_add1","encoderblock11_add2/in2");
net = connectLayers(net,"encoderblock11_dropout3","encoderblock11_add2/in1");
net = connectLayers(net,"encoderblock11_add2","encoderblock12_layernorm1");
net = connectLayers(net,"encoderblock11_add2","encoderblock12_add1/in2");
net = connectLayers(net,"encoderblock12_dropout1","encoderblock12_add1/in1");
net = connectLayers(net,"encoderblock12_add1","encoderblock12_layernorm2");
net = connectLayers(net,"encoderblock12_add1","encoderblock12_add2/in2");
net = connectLayers(net,"encoderblock12_dropout3","encoderblock12_add2/in1");
net = initialize(net);