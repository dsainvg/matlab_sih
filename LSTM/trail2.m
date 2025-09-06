
% %% 7-Day Plant Health & Multivariate Forecast LSTM
% 
% clc; clear; close all;
% %% Load Dataset
% data = readtable('plant_health_data.csv');
% 
% %% Create Plant_Health_Status if missing
% if ~ismember('Plant_Health_Status', data.Properties.VariableNames)
%     warning('Plant_Health_Status column missing. Creating synthetic health labels.');
%     data.Plant_Health_Status = categorical(repmat("Healthy",height(data),1));
% end
% data.Plant_Health_Status = categorical(data.Plant_Health_Status);
% 
% %% Feature Engineering: Synthetic Stress Score
% if ~ismember('Stress_Score', data.Properties.VariableNames)
%     fprintf('Creating synthetic Stress_Score based on soil moisture, temperature, and humidity.\n');
%     % Example: Higher temp + low moisture → higher stress
%     normalizedMoisture = (data.Soil_Moisture - min(data.Soil_Moisture)) / (max(data.Soil_Moisture)-min(data.Soil_Moisture)+eps);
%     normalizedTemp     = (data.Ambient_Temperature - min(data.Ambient_Temperature)) / (max(data.Ambient_Temperature)-min(data.Ambient_Temperature)+eps);
%     normalizedHumidity = (data.Humidity - min(data.Humidity)) / (max(data.Humidity)-min(data.Humidity)+eps);
%     % Stress score = weighted combination
%     data.Stress_Score = 0.5*(1-normalizedMoisture) + 0.3*normalizedTemp + 0.2*(1-normalizedHumidity);
% end
% 
% %% Define Input Features and Targets
% inputFeatures = {'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', ...
%                  'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', ...
%                  'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', ...
%                  'Electrochemical_Signal'};
% 
% targetVars = {'Soil_Moisture','Ambient_Temperature','Humidity','Soil_Temperature','Chlorophyll_Content','Stress_Score'};
% 
% %% Normalize Inputs & Targets
% 
% X_all = data{:,inputFeatures};
% minX = min(X_all); maxX = max(X_all);
% X_norm_all = (X_all - minX) ./ (maxX - minX + eps);
% 
% Y_all = data{:,targetVars};
% minY = min(Y_all); maxY = max(Y_all);
% Y_norm_all = (Y_all - minY) ./ (maxY - minY + eps);
% 
% %% Sequence Preparation
% seqLength = 30; % past days
% predDays  = 7;  % future days
% numRows   = height(data);
% numSamples = numRows - seqLength - predDays + 1;
% 
% X_seq = cell(1,numSamples);
% Y_seq = cell(1,numSamples);
% Y_health_seq = cell(1,numSamples);
% 
% for i = 1:numSamples
%     X_seq{i} = X_norm_all(i:i+seqLength-1,:)';
%     futureNorm = Y_norm_all(i+seqLength:i+seqLength+predDays-1,:);
%     Y_seq{i} = futureNorm(:)'; % flatten
%     futureHealth = data.Plant_Health_Status(i+seqLength:i+seqLength+predDays-1);
%     Y_health_seq{i} = mode(futureHealth); % mode over 7 days
% end
% 
% %% Remove sequences with NaNs
% 
% valid = true(1,numSamples);
% for i=1:numSamples
%     if any(isnan(X_seq{i}(:))) || any(isnan(Y_seq{i}(:)))
%         valid(i) = false;
%     end
% end
% X_seq = X_seq(valid);
% Y_seq = Y_seq(valid);
% Y_health_seq = Y_health_seq(valid);
% numSamplesClean = numel(X_seq);
% fprintf('Usable sequences after NaN-cleaning: %d\n', numSamplesClean);
% 
% %% Train-Test Split
% idxTrain = floor(0.8*numSamplesClean);
% XTrain = X_seq(1:idxTrain);
% XTest  = X_seq(idxTrain+1:end);
% 
% YTrain_reg_cells = Y_seq(1:idxTrain);
% YTest_reg_cells  = Y_seq(idxTrain+1:end);
% 
% YTrain_class = categorical([Y_health_seq{1:idxTrain}]);
% YTest_class  = categorical([Y_health_seq{idxTrain+1:end}]);
% 
% % Convert regression cells to matrix
% YTrain_reg_mat = vertcat(YTrain_reg_cells{:});
% YTest_reg_mat  = vertcat(YTest_reg_cells{:});
% fprintf('Train samples: %d | Test samples: %d | Regression target dim: %d\n', numel(XTrain), numel(XTest), size(YTrain_reg_mat,2));
% 
% %% LSTM Regression Model
% numFeatures  = size(XTrain{1},1);
% numResponses = size(YTrain_reg_mat,2);
% 
% layers_reg = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(128,'OutputMode','last')
%     fullyConnectedLayer(numResponses)
%     regressionLayer];
% 
% options_reg = trainingOptions('adam', ...
%     'MaxEpochs',50, ...
%     'MiniBatchSize',32, ...
%     'Shuffle','every-epoch', ...
%     'Plots','none', ... % no recursive plotting
%     'Verbose',0, ...
%     'ValidationData',{XTest,YTest_reg_mat});
% 
% regressionNet = trainNetwork(XTrain,YTrain_reg_mat,layers_reg,options_reg);
% 
% %% LSTM Classification Model
% numClasses = numel(categories(YTrain_class));
% layers_class = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(128,'OutputMode','last')
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer];
% 
% options_class = trainingOptions('adam', ...
%     'MaxEpochs',50, ...
%     'MiniBatchSize',32, ...
%     'Shuffle','every-epoch', ...
%     'Plots','none', ...
%     'Verbose',0, ...
%     'ValidationData',{XTest,YTest_class});
% 
% classificationNet = trainNetwork(XTrain,YTrain_class,layers_class,options_class);
% 
% %% Evaluate Test Predictions
% YPred_reg_norm = predict(regressionNet,XTest);
% YPred_class = classify(classificationNet,XTest);
% 
% % Compute accuracies
% trainAcc = mean(classify(classificationNet,XTrain) == YTrain_class);
% valAcc   = mean(YPred_class == YTest_class);
% fprintf('Plant Health Accuracy - Train: %.2f%% | Validation: %.2f%%\n', trainAcc*100, valAcc*100);
% 
% %% Build Full Test Forecast Table
% numTest = numel(XTest);
% dayCols = strcat("Day", string(1:predDays));
% 
% %% Column names
% colNames = strings(1, numel(targetVars)*predDays);
% for t=1:numel(targetVars)
%     for d=1:predDays
%         colNames((t-1)*predDays+d) = targetVars{t} + "_Day" + string(d);
%     end
% end
% colNames = [colNames, "Predicted_Plant_Health"];
% 
% 
% % Example plot: True vs Predicted soil moisture (first feature)
% figure;
% plot(YTest_reg_mat(:,1), 'b', 'LineWidth', 1.5);
% hold on;
% plot(YPred_reg(:,1), 'r--', 'LineWidth', 1.5);
% xlabel('Test Sample');
% ylabel('Soil Moisture (normalized)');
% legend('True','Predicted');
% title('Soil Moisture 7-day Prediction');
% 
% %% Table data
% fullTableData = zeros(numTest, numel(targetVars)*predDays);
% for i=1:numTest
%     yPred = YPred_reg_norm(i,:);
%     yPredMat = reshape(yPred,[predDays,numel(targetVars)]);
%     yPredOrig = yPredMat .* (maxY - minY) + minY;
%     fullTableData(i,:) = yPredOrig(:)';
% end
% fullTable = array2table([fullTableData, zeros(numTest,0)], 'VariableNames', colNames(1:end-1));
% fullTable.Predicted_Plant_Health = string(YPred_class);
% 
% % Display first 5 test samples
% disp('Full 7-Day Forecast for Test Samples (first 5 rows):');
% disp(fullTable(1:min(5,numTest),:));
% 
% % Save CSV
% writetable(fullTable,'7Day_Forecast_TestSamples.csv');
% fprintf('Test forecast table saved as 7Day_Forecast_TestSamples.csv\n');
% 
% %% ---------------------------
% % Predict Real Upcoming 7 Days
% %% ---------------------------
% latestX = data{end-seqLength+1:end, inputFeatures};
% latestX_norm = (latestX - minX) ./ (maxX - minX + eps);
% latestX_seq = latestX_norm';
% latestX_seqCell = {latestX_seq};
% 
% % Regression
% yPred_norm = predict(regressionNet, latestX_seqCell);
% yPredMat = reshape(yPred_norm,[predDays,numel(targetVars)]);
% yPredOrig = yPredMat .* (maxY - minY) + minY;
% 
% % Classification
% yHealthPred = classify(classificationNet, latestX_seqCell);
% 
% % Build table
% forecastTable = array2table(yPredOrig','VariableNames',dayCols,'RowNames',targetVars);



%% trail2_reported.m  — 7-Day Plant Health & Multivariate Forecast LSTM (fixed + reporting)
clc; clear; close all;

%% Load Dataset
data = readtable('plant_health_data.csv');

%% Create Plant_Health_Status if missing
if ~ismember('Plant_Health_Status', data.Properties.VariableNames)
    warning('Plant_Health_Status column missing. Creating synthetic health labels.');
    data.Plant_Health_Status = categorical(repmat("Healthy",height(data),1));
end
data.Plant_Health_Status = categorical(data.Plant_Health_Status);

%% Feature Engineering: Synthetic Stress Score
if ~ismember('Stress_Score', data.Properties.VariableNames)
    fprintf('Creating synthetic Stress_Score based on soil moisture, temperature, and humidity.\n');
    normalizedMoisture = (data.Soil_Moisture - min(data.Soil_Moisture)) ./ (max(data.Soil_Moisture)-min(data.Soil_Moisture)+eps);
    normalizedTemp     = (data.Ambient_Temperature - min(data.Ambient_Temperature)) ./ (max(data.Ambient_Temperature)-min(data.Ambient_Temperature)+eps);
    normalizedHumidity = (data.Humidity - min(data.Humidity)) ./ (max(data.Humidity)-min(data.Humidity)+eps);
    data.Stress_Score = 0.5*(1-normalizedMoisture) + 0.3*normalizedTemp + 0.2*(1-normalizedHumidity);
end

%% Define Inputs/Targets
inputFeatures = {'Soil_Moisture', 'Ambient_Temperature', 'Soil_Temperature', ...
                 'Humidity', 'Light_Intensity', 'Soil_pH', 'Nitrogen_Level', ...
                 'Phosphorus_Level', 'Potassium_Level', 'Chlorophyll_Content', ...
                 'Electrochemical_Signal'};

targetVars = {'Soil_Moisture','Ambient_Temperature','Humidity','Soil_Temperature','Chlorophyll_Content','Stress_Score'};

%% Normalize Inputs & Targets
X_all = data{:,inputFeatures};
minX = min(X_all); maxX = max(X_all);
X_norm_all = (X_all - minX) ./ (maxX - minX + eps);

Y_all = data{:,targetVars};
minY = min(Y_all); maxY = max(Y_all);
Y_norm_all = (Y_all - minY) ./ (maxY - minY + eps);

%% Sequence Prep
seqLength = 30; predDays = 7;
numRows = height(data);
numSamples = numRows - seqLength - predDays + 1;

X_seq = cell(1,numSamples);
Y_seq = cell(1,numSamples);
Y_health_seq = cell(1,numSamples);

for i = 1:numSamples
    X_seq{i} = X_norm_all(i:i+seqLength-1,:)';
    futureNorm = Y_norm_all(i+seqLength:i+seqLength+predDays-1,:);
    Y_seq{i} = futureNorm(:)'; % flattened
    futureHealth = data.Plant_Health_Status(i+seqLength:i+seqLength+predDays-1);
    Y_health_seq{i} = mode(futureHealth);
end

%% Remove sequences with NaNs
valid = true(1,numSamples);
for i=1:numSamples
    if any(isnan(X_seq{i}(:))) || any(isnan(Y_seq{i}(:)))
        valid(i) = false;
    end
end
X_seq = X_seq(valid);
Y_seq = Y_seq(valid);
Y_health_seq = Y_health_seq(valid);
numSamplesClean = numel(X_seq);
fprintf('Usable sequences after NaN-cleaning: %d\n', numSamplesClean);

%% Train-Test Split
idxTrain = floor(0.8*numSamplesClean);
XTrain = X_seq(1:idxTrain); XTest  = X_seq(idxTrain+1:end);

YTrain_reg_cells = Y_seq(1:idxTrain);
YTest_reg_cells  = Y_seq(idxTrain+1:end);

YTrain_class = categorical([Y_health_seq{1:idxTrain}]);
YTest_class  = categorical([Y_health_seq{idxTrain+1:end}]);

YTrain_reg_mat = vertcat(YTrain_reg_cells{:});
YTest_reg_mat  = vertcat(YTest_reg_cells{:});
fprintf('Train samples: %d | Test samples: %d | Regression target dim: %d\n', numel(XTrain), numel(XTest), size(YTrain_reg_mat,2));

%% LSTM Regression Model
numFeatures  = size(XTrain{1},1);
numResponses = size(YTrain_reg_mat,2);

layers_reg = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(128,'OutputMode','last')
    fullyConnectedLayer(numResponses)
    regressionLayer];

options_reg = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',0, ...
    'ValidationData',{XTest,YTest_reg_mat});

regressionNet = trainNetwork(XTrain,YTrain_reg_mat,layers_reg,options_reg);

%% LSTM Classification Model
numClasses = numel(categories(YTrain_class));
layers_class = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(128,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options_class = trainingOptions('adam', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'Plots','none', ...
    'Verbose',0, ...
    'ValidationData',{XTest,YTest_class});

classificationNet = trainNetwork(XTrain,YTrain_class,layers_class,options_class);

%% Evaluate Test Predictions (regression & classification)
YPred_reg_norm = predict(regressionNet,XTest);    % normalized flattened predictions
YPred_class = classify(classificationNet,XTest);

% single-line summary metrics
trainAcc = mean(classify(classificationNet,XTrain) == YTrain_class);
valAcc   = mean(YPred_class == YTest_class);
fprintf('Plant Health Classification Accuracy — Train: %.2f%% | Validation: %.2f%%\n', trainAcc*100, valAcc*100);

%% Denormalize predictions & ground truth for regression
numTest = numel(XTest);
numTargets = numel(targetVars);

% Denorm matrices
YTest_reg_mat_denorm = zeros(size(YTest_reg_mat));
YPred_reg_denorm = zeros(size(YPred_reg_norm));

% reshape/denorm each flattened sample
for i=1:size(YTest_reg_mat,1)
    yTrue_norm = YTest_reg_mat(i,:);
    yPred_norm = YPred_reg_norm(i,:);
    % convert back sample-wise
    yTrueMat = reshape(yTrue_norm, [predDays, numTargets]);
    yPredMat = reshape(yPred_norm, [predDays, numTargets]);
    yTrueOrig = yTrueMat .* (maxY - minY) + minY;
    yPredOrig = yPredMat .* (maxY - minY) + minY;
    YTest_reg_mat_denorm(i,:) = yTrueOrig(:)';
    YPred_reg_denorm(i,:) = yPredOrig(:)';
end

%% Compute per-target RMSE (aggregated across all days & samples)
rmsePerTarget = zeros(1,numTargets);
for t=1:numTargets
    idx = (t-1)*predDays + (1:predDays);
    errors = YTest_reg_mat_denorm(:,idx) - YPred_reg_denorm(:,idx);
    rmsePerTarget(t) = sqrt(mean(errors(:).^2));
end

% Print RMSEs
fprintf('Regression RMSE per target:\n');
for t=1:numTargets
    fprintf('  %s: %.4f\n', targetVars{t}, rmsePerTarget(t));
end

%% Visuals: Per-target true vs predicted for the first N test samples (stacked)
Nplot = min(8, numTest); % how many test-samples to visualize
figure('Position',[100 100 1100 700]);
for t = 1:numTargets
    subplot(ceil(numTargets/2),2,t);
    hold on; box on;
    for i=1:Nplot
        idx = (t-1)*predDays + (1:predDays);
        plot(1:predDays, YTest_reg_mat_denorm(i,idx), 'LineWidth', 1); % true
        plot(1:predDays, YPred_reg_denorm(i,idx), '--', 'LineWidth', 1); % pred
    end
    xlabel('Day'); ylabel(targetVars{t});
    title(sprintf('%s — True (solid) vs Pred (dashed) — first %d test samples', targetVars{t}, Nplot));
    legend({'True','Pred'}, 'Location','best');
    hold off;
end
sgtitle('Per-target 7-day True vs Predicted (first test samples)');
exportgraphics(gcf,'regression_true_vs_pred_samples.pdf','ContentType','vector');

%% Plot aggregated error box per target (for each day)
figure('Position',[100 100 1000 400]);
errMatrix = zeros(numTest, numTargets);
for t=1:numTargets
    idx = (t-1)*predDays + (1:predDays);
    errMatrix(:,t) = mean(abs(YTest_reg_mat_denorm(:,idx) - YPred_reg_denorm(:,idx)), 2); % per-sample avg abs error for target
end
boxplot(errMatrix, 'Labels', targetVars);
ylabel('Mean Absolute Error (per sample across 7 days)');
title('Error distribution per target across test samples');
exportgraphics(gcf,'regression_error_boxplot.pdf','ContentType','vector');

%% Classification: Confusion Matrix & metrics
figure('Position',[100 100 700 600]);
confusionchart(YTest_class, YPred_class);
title('Confusion Matrix — Plant Health (Validation)'); 
exportgraphics(gcf,'confusion_matrix.pdf','ContentType','vector');

%% Build full forecast table for test set (denormalized)
dayCols = strcat("Day", string(1:predDays));
colNames = strings(1, numTargets*predDays);
for t=1:numTargets
    for d=1:predDays
        colNames((t-1)*predDays+d) = targetVars{t} + "_Day" + string(d);
    end
end
colNames = [colNames, "Predicted_Plant_Health"];

fullTableData = zeros(numTest, numTargets*predDays);
for i=1:numTest
    fullTableData(i,:) = YPred_reg_denorm(i,:);
end
fullTable = array2table(fullTableData, 'VariableNames', colNames(1:end-1));
fullTable.Predicted_Plant_Health = string(YPred_class);

% Save first 10 rows to CSV for quick view
writetable(fullTable(1:min(10,numTest),:),'7Day_Forecast_TestSamples_first10.csv');
writetable(fullTable,'7Day_Forecast_TestSamples_full.csv');
fprintf('Test forecast tables saved as CSV (first10 and full).\n');

%% Display first 5 rows in command window
disp('Full 7-Day Forecast for Test Samples (first 5 rows):');
disp(fullTable(1:min(5,numTest),:));

%% ---------------------------
% Predict Real Upcoming 7 Days (latest)
%% ---------------------------
latestX = data{end-seqLength+1:end, inputFeatures};
latestX_norm = (latestX - minX) ./ (maxX - minX + eps);
latestX_seq = latestX_norm';
latestX_seqCell = {latestX_seq};

yPred_norm = predict(regressionNet, latestX_seqCell);
yPredMat = reshape(yPred_norm,[predDays,numTargets]);
yPredOrig = yPredMat .* (maxY - minY) + minY;

yHealthPred = classify(classificationNet, latestX_seqCell);

% Build table for forecast
forecastTable = array2table(yPredOrig','VariableNames',dayCols,'RowNames',targetVars);
% forecastTable.Predicted_Plant_Health = string(yHealthPred);

disp('Upcoming 7-Day Forecast (latest window):');
disp(forecastTable);
tempTbl = array2table(yPredOrig','VariableNames',dayCols);
tempTbl = addvars(tempTbl, targetVars', 'Before', 1, 'NewVariableNames', "Target");
writetable(tempTbl, 'Upcoming_7Day_Forecast.csv');

% plot the upcoming forecast in separate figure
figure('Position',[100 100 900 600]);
for t=1:numTargets
    subplot(ceil(numTargets/2),2,t);
    plot(1:predDays, yPredOrig(:,t), '-o', 'LineWidth',1.5);
    xlabel('Day'); ylabel(targetVars{t});
    title(sprintf('Upcoming forecast - %s', targetVars{t}));
end
sgtitle('Predicted Upcoming 7-Day Forecast (latest window)');
exportgraphics(gcf,'upcoming_7day_forecast.pdf','ContentType','vector');


