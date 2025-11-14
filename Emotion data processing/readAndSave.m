clc;
close all;
clear;

sampleRate = 5000;
ifSave = 1;
NewDir = '/saveData/';

experimentorSet = ["user1","user2","user3","user4","user5","user6","user7","user8","user9","user10","user11","user12","user13"];
% here the stroke was written as petting.
% For the action with "ing", it means that the action was recording during
% watching the videos. Thus, we would like to use the latter part of data.
Gesture = ["peting","press","push","pating","slap","slaping","touch","tickle","tickling","knock"];
% Gesture = ["push"];
for experimentorName = experimentorSet
for gestVar = Gesture
for video_num = 1:9
    Header1 = ['./',char(experimentorName)]; % Header is required to changed before running.

    %% Read files
    if ~exist([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_V1.mat'],'file')
      continue;
    end
    
    V1 = importdata([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_V1.mat']);
    V2 = importdata([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_V2.mat']);
    V3 = importdata([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_V3.mat']);
    V4 = importdata([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_V4.mat']);
    %x  = importdata([Header1,'/ori_data/',char(gestVar),'_',num2str(video_num),'_x.mat']);
    
    V_all = [V1; V2; V3; V4];
    saveIndex = 1;

    for ii = 1:4
    %% Calculate the integration of V
    V_read = V_all(ii,:) - mean(V_all(ii,:));
    Q_read = zeros(length(V_read),1);
    for i = 2:length(V_read)
        Q_read(i) = Q_read(i-1)+V_read(i);
    end
    
    %% Segment the data，Seg_read is the index
    % Find negative peak
    mpd = 500;

    if gestVar == "knock"
        mpd = 500;
    end

    if gestVar == "tickle"
        [~,Seg_read,w,p] = findpeaks(-smooth(Q_read,50), 'MinPeakDistance', mpd);
    else
        [~,Seg_read,w,p] = findpeaks(-smooth(Q_read,50), 'MinPeakDistance', mpd, 'MinPeakProminence', 10);
    end
    
    wpi = max(p)/3;
    if gestVar == "peting"
        wpi = 2;
    elseif gestVar == "touch"
        wpi = 0.5;
    elseif strcmp(gestVar,"tickling") | strcmp(gestVar,"tickle") 
        wpi = 1;
    elseif gestVar == "knock"
        wpi = 0.5;
    elseif gestVar == "slaping"
        wpi = max(p)/5;
    elseif gestVar == "pating"
        wpi = max(p)/30;
    end

    figure(3)
    scatter(Seg_read,0.1)
    hold on
    plot(Q_read)
    plot(Seg_read,p)
    x=[0,length(Q_read)];
    y=[wpi,wpi];
    line(x,y)
    hold off

    wrong_peaks_index = find(p<wpi);
    % if gestVar == "knock" && experimentorName ~= "xyk"
    %     for pk_index = 1:length(Seg_read)-1
    %         if ~sum(ismember(wrong_peaks_index,pk_index))
    %             if ~sum(ismember(wrong_peaks_index,pk_index+1))
    %                 if p((pk_index)) > p((pk_index+1))
    %                     wrong_peaks_index = [wrong_peaks_index;pk_index+1];
    %                 else
    %                     wrong_peaks_index = [wrong_peaks_index;pk_index];
    %                 end
    %             end
    %         end
    %     end
    % end
    Seg_read(wrong_peaks_index) = [];

    if gestVar == "pating"
        evenNum = 2:2:length(Seg_read);
        Seg_read(evenNum) = [];
    % elseif gestVar == "knock" && length(wrong_peaks_index) > 20
    %     evenNum = 2:2:length(Seg_read);
    %     Seg_read(evenNum) = [];
    end
        
    %% 绘制分段情况图
    figure(1)
    scatter(Seg_read,0.1)
    hold on
    plot(V_read)
    plot(Q_read/100)
    plot(smooth(Q_read,500)/100)
    hold off
        
    indexNum = length(Seg_read)-1;
    indexStart = 0;
    
    % For the action with "ing", it means that the action was recording during
    % watching the videos. Thus, we would like to use the latter part of data.
    if extract(gestVar,"ing") == "ing"
        indexStart = 0;%round(indexNum/2);
        saveDataDir_before = [Header1,NewDir,char(gestVar),'Before_',num2str(video_num),'_'];
        saveDataDir_before_wrong = [Header1,NewDir,char(gestVar),'Before_',num2str(video_num),'_wrong_'];
        saveDataDir = [Header1,NewDir,char(gestVar),'_',num2str(video_num),'_'];
        saveDataDir_wrong = [Header1,NewDir,char(gestVar),'_',num2str(video_num),'_wrong_'];
    else
        saveDataDir = [Header1,NewDir,char(gestVar),'_',num2str(video_num),'_'];
        saveDataDir_wrong = [Header1,NewDir,char(gestVar),'_',num2str(video_num),'_wrong_'];
    end
    if ifSave && ii == 1
        if exist([saveDataDir,'spectrogram'],'dir') == 0
            status = mkdir([saveDataDir,'spectrogram']);
            status = mkdir([saveDataDir,'spectrum']);
            status = mkdir([saveDataDir,'charge']);
        else
            status = rmdir([saveDataDir,'spectrogram'],'s');
            status = rmdir([saveDataDir,'spectrum'],'s');
            status = rmdir([saveDataDir,'charge'],'s');
            status = mkdir([saveDataDir,'spectrogram']);
            status = mkdir([saveDataDir,'spectrum']);
            status = mkdir([saveDataDir,'charge']);
        end
        if exist([saveDataDir_wrong,'spectrogram'],'dir') == 0
            status = mkdir([saveDataDir_wrong,'spectrogram']);
            status = mkdir([saveDataDir_wrong,'spectrum']);
            status = mkdir([saveDataDir_wrong,'charge']);
        else
            status = rmdir([saveDataDir_wrong,'spectrogram'],'s');
            status = rmdir([saveDataDir_wrong,'spectrum'],'s');
            status = rmdir([saveDataDir_wrong,'charge'],'s');
            status = mkdir([saveDataDir_wrong,'spectrogram']);
            status = mkdir([saveDataDir_wrong,'spectrum']);
            status = mkdir([saveDataDir_wrong,'charge']);
        end
        if indexStart > 0
            if exist([saveDataDir_before,'spectrogram'],'dir') == 0
                status = mkdir([saveDataDir_before,'spectrogram']);
                status = mkdir([saveDataDir_before,'spectrum']);
                status = mkdir([saveDataDir_before,'charge']);
            else
                status = rmdir([saveDataDir_before,'spectrogram'],'s');
                status = rmdir([saveDataDir_before,'spectrum'],'s');
                status = rmdir([saveDataDir_before,'charge'],'s');
                status = mkdir([saveDataDir_before,'spectrogram']);
                status = mkdir([saveDataDir_before,'spectrum']);
                status = mkdir([saveDataDir_before,'charge']);
            end
            if exist([saveDataDir_before_wrong,'spectrogram'],'dir') == 0
                status = mkdir([saveDataDir_before_wrong,'spectrogram']);
                status = mkdir([saveDataDir_before_wrong,'spectrum']);
                status = mkdir([saveDataDir_before_wrong,'charge']);
            else
                status = rmdir([saveDataDir_before_wrong,'spectrogram'],'s');
                status = rmdir([saveDataDir_before_wrong,'spectrum'],'s');
                status = rmdir([saveDataDir_before_wrong,'charge'],'s');
                status = mkdir([saveDataDir_before_wrong,'spectrogram']);
                status = mkdir([saveDataDir_before_wrong,'spectrum']);
                status = mkdir([saveDataDir_before_wrong,'charge']);
            end
        end
    end

    char_press='e';
    saveWrongIndex = 1;
    for index = 1:indexNum
        figure(2)
        clf
        if indexStart > 0 % For the data acquiring during watching video
            if index <= indexStart % Front part
                [T,F,Pout,FV1_ROI,Pout2,Xout,Qout,V_Seg] = saveFileFunc(index, Seg_read, V_read, Q_read, sampleRate);
                saveDDir = saveDataDir_before;
                saveDDir_wrong = saveDataDir_before_wrong;
                subplot(4,1,1)
                title([char(experimentorName),'-',char(gestVar),'-Video',num2str(video_num),'-',num2str(index)],'FontSize',16)
            else % Latter part
                [T,F,Pout,FV1_ROI,Pout2,Xout,Qout,V_Seg] = saveFileFunc(index, Seg_read, V_read, Q_read, sampleRate);
                saveDDir = saveDataDir;
                saveDDir_wrong = saveDataDir_wrong;
                subplot(4,1,1)
                title([char(experimentorName),'-',char(gestVar),'-Video',num2str(video_num),'-',num2str(index)],'FontSize',16)
            end
            if index == indexStart
                saveIndex = 1;
                saveWrongIndex = 1;
                saveDDir = saveDataDir;
                saveDDir_wrong = saveDataDir_wrong;
            end
        else
            [T,F,Pout,FV1_ROI,Pout2,Xout,Qout,V_Seg] = saveFileFunc(index, Seg_read, V_read, Q_read, sampleRate);
            saveDDir = saveDataDir;
            saveDDir_wrong = saveDataDir_wrong;
            subplot(4,1,1)
            title([char(experimentorName),'-',char(gestVar),'-Video',num2str(video_num),'-',num2str(index)],'FontSize',16)
        end
        if ifSave
            % waitforbuttonpress
            % key = get(gcf,'CurrentKey');
            % if strcmp(key, 's') || strcmp(key, "downarrow")
            %     disp(['You pressed: ', key, ':save']);
                filename = num2str(saveIndex);
                % save([saveDDir,'spectrogram/',filename],"T","F","Pout");
                % save([saveDDir,'spectrum/',filename],"FV1_ROI","Pout2");
                save([saveDDir,'charge/',filename],"Xout","Qout","V_Seg");
                saveIndex = saveIndex+1;
            % else
            %     disp(['You pressed: ', key, ':next']);
            %     filename = num2str(saveWrongIndex);
            %     save([saveDDir_wrong,'spectrogram/',filename],"T","F","Pout");
            %     save([saveDDir_wrong,'spectrum/',filename],"FV1_ROI","Pout2");
            %     save([saveDDir_wrong,'charge/',filename],"Xout","Qout","V_Seg");
            %     saveWrongIndex = saveWrongIndex+1;
            % end
        else
            pause(0.3)
        end
    end
    end
end
end
end

function [T,F,Pout,FV1_ROI,Pout2,Xout,Qout,V_Seg] = saveFileFunc(index, Seg_read, V_read, Q_read, sampleRate)
    % Set the parameters of the spectrum graph
    frequencyLimits = [0 1000]; % Hz
    timeResolution = 0.1; % 秒
    overlapPercent = 50;
    % Obtain the range of the segmented intervals
    Index_Seg_read = Seg_read(index):Seg_read(index+1);

    % Obtain the segmented voltage signal
    V_Seg_read = V_read(Index_Seg_read);

    timeValues_read = (0:length(V_Seg_read)-1).'/sampleRate;
    
    % Draw time-frequency diagram
    [P1,F,T] = pspectrum(V_Seg_read,timeValues_read, ...
        'spectrogram', ...
        'FrequencyLimits',frequencyLimits, ...
        'TimeResolution',timeResolution, ...
        'OverlapPercent',overlapPercent);
    
    % Draw FFT spectrum
    [PV1_ROI, FV1_ROI] = pspectrum(V_Seg_read,sampleRate, ...
        'FrequencyLimits',frequencyLimits);
    % figure(2)
    % subplot(4,1,1)
    % surface(T,F,10*log10(P1))
    % shading interp
    % colormap(jet)
    % colorbar
    % clim([-100 -40])

    % Save time-frequency diagram
    Pout = 10*log10(P1);
    % subplot(4,1,2)
    % plot(FV1_ROI, 10*log10(PV1_ROI))
    ylim([-100 -20])

    % Save FFT spectrum
    Pout2 = 10*log10(PV1_ROI);
    
    % Draw charge
    % subplot(4,1,3)
    % plot(Q_read(Index_Seg_read)-min(Q_read(Index_Seg_read)))
    
    % Draw voltage
    Xout = Index_Seg_read ./ 5000;
    % subplot(4,1,4)
    % plot(Xout,V_Seg_read)
    % Save charge
    Qout = Q_read(Index_Seg_read)-min(Q_read(Index_Seg_read));
    V_Seg = V_Seg_read;
end