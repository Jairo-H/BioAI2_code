confu_test = 1;

if confu_test
    showData = VerifyCell;
    showLabel = VerifyLabel;
else
    showData = TrainCell;
    showLabel = TrainLabel;
end

OutputLabel = categorical(strings(length(showData),1));
for i = 1:length(showData)
    YoutTest = classify(net,showData(i));
    [counts, values] = histcounts(YoutTest{1});
    [v,ind] = max(counts(1:24));
    OutputLabel(i) = values(ind);
    i
end

TestLabel_action = [];
YouLabel_action = [];
TestLabel_affection = [];
YouLabel_affection = [];
for i = 1:length(showLabel)
    [out1, out2] = split(string(showLabel(i)),'_');
    TestLabel_action = [TestLabel_action;out1(1)];
    TestLabel_affection = [TestLabel_affection;out1(2)];
    [out1, out2] = split(string(OutputLabel(i)),'_');
    YouLabel_action = [YouLabel_action;out1(1)];
    YouLabel_affection = [YouLabel_affection;out1(2)];
end

accuracy = mean(showLabel == OutputLabel);
accuracy_action = mean(TestLabel_action == YouLabel_action);
accuracy_affection = mean(TestLabel_affection == YouLabel_affection);