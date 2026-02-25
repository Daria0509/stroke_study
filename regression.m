%{
Global network efficiency vs lesion impact score (group-level analysis)

This script tests whether a lesion-derived impact score is associated with
whole-brain network efficiency across a cohort.

Overview
--------
For each patient:
1) Load a weighted structural connectivity matrix (FA-based) from *_FA_END.mat.
2) Clean the matrix (symmetrize, zero diagonal, remove NaNs).
3) Apply proportional thresholding to a fixed network density (e.g., top 15% edges),
   preserving original weights on retained edges.
4) Compute graph metrics using the Brain Connectivity Toolbox (BCT):
     - Global efficiency (Eglob) on the thresholded weighted matrix
     - Mean betweenness centrality (BC_mean) as an optional secondary metric

Then:
5) Load lesion impact scores (Excel) and inner-join by patient_id.
6) Compute correlations between Eglob and lesion_score (raw and log10-transformed).
7) Fit linear models:
     - Univariate: Eglob ~ lesion_score
     - Univariate: Eglob ~ log_lesion
     - Standardized: z_Eglob ~ z_log_lesion
     - With covariates (if present): Eglob ~ log_lesion + infarct_volume_ml + brain_volume_ml + Age + Sex
8) (Optional) Generate scatter / marginal-prediction / added-variable plots.

Inputs (edit in “Paths”)
------------------------
- data_dir:
    Folder containing one connectivity file per subject:
      <PATIENT_ID>_FA_END.mat
    Each .mat file must contain the variable:
      CM   (AAL90 × AAL90 weighted connectivity matrix)
- lesionXLSX:
    Excel file containing at minimum:
      patient_id, lesion_score
    Optional covariates (used if present):
      infarct_volume_ml, brain_volume_ml, Age, Sex
- Brain Connectivity Toolbox (BCT) must be on the MATLAB path:
    betweenness_wei, efficiency_wei, etc.

Key parameters
--------------
- density (e.g., 0.15):
    Proportion of undirected edges kept by weight (proportional thresholding).
- aal_n (e.g., 90):
    Number of AAL regions (matrix size expected).

Outputs
-------
- Console output:
    Per-subject Eglob and BC_mean during processing, plus regression summaries.
- Figures (optional; see filenames defined at the top):
    Saved to outDir (data_dir/stats) when the corresponding plotting blocks are enabled.

Notes 
-------------------
- The proportional thresholding is implemented in `fix_density()`:
    it keeps the strongest edges in the upper triangle (excluding diagonal) and
    mirrors them to enforce an undirected graph.
- For weighted shortest paths, the script converts weights to “lengths” via:
      L = 1 ./ W   (for W > 0)
  so stronger connections correspond to shorter path lengths.
- Log-transform uses:
      log_lesion = log10(lesion_score + eps)
  Set eps > 0 if lesion_score can be zero and you want to avoid -Inf.
%}

close all; clc; clear

% Paths 

addpath(genpath('.../Documents/MATLAB/BCT')); % Brain Connectivity Toolbox
savepath

data_dir        = '.../network_analysis/L6_step2/fa_end';
pattern         = '*_FA_END.mat'; % e.g. 02_CG_s1_FA_END.mat
lesionXLSX      = '.../network_analysis/lesion_impact_scores.xlsx'; % file with demographics included

outDir          = fullfile(data_dir, 'stats');

% Remember to change the figures' titles according to the CSD-mode (e.g. CSD, L_{max}=6)

figScatterRaw     = fullfile(outDir, 'uni_rawlesion.png');
figScatterLog     = fullfile(outDir, 'uni_loglesion.png');
figCovMarginalRaw = fullfile(outDir, 'cov_marginal_rawlesion.png');
figCovMarginalLog = fullfile(outDir, 'cov_marginal_loglesion.png');
figCovAddedLog    = fullfile(outDir, 'cov_added_variable_loglesion.png');

density = 0.15; % fixed density (15%)
aal_n = 90;     % AAL 90 regions


% --- FA matrices (filter AppleDouble "._" files)
files = dir(fullfile(data_dir, pattern));
if isempty(files)
    error('No files found matching "%s" in %s', pattern, data_dir);
end

% files = files(~startsWith({files.name}, '._'));  % drop AppleDouble files
% fprintf('Found %d candidate files.\n', numel(files));

% --- Compute metrics for each patient file 
rows = {};
for patient = 1:numel(files)
    file = fullfile(files(patient).folder, files(patient).name);
    [~, base] = fileparts(file); % e.g. 02_CG_s1_FA_END
    id  = string(erase(base, '_FA_END')); 
    S = load(file);      
    W = S.CM; % FA matrix

    W = (W + W.')./2;     % symmetrize
    W(1:aal_n+1:end) = 0; % zero diagonal
    W(isnan(W)) = 0;      % drop NaNs

    % % normalization to [0,1]: makes the corr/regr positive
    % mx = max(W(:));
    % if mx > 0
    %     W = W / mx;
    % end

    % --- 15% threshold
    W15 = fix_density(W, density);
    
    % --- metrics ---
    L = W15; 
    pos = L > 0;
    L(pos) = 1 ./ L(pos); 

    BC = betweenness_wei(L); % optional: node betweenness centrality
    BC_mean = mean(BC);

    Eglob = efficiency_wei(W15);

    rows(end+1,:) = {id, files(patient).name, Eglob, BC_mean}; %#ok<SAGROW>
    fprintf('%s\tEglob=%.5f\tBC_mean=%.5f\n', files(patient).name, Eglob, BC_mean);
end

table_network = cell2table(rows, 'VariableNames', {'patient_id','filename','Eglob','BC_mean'});
table_network.patient_id = string(table_network.patient_id);

% --- Read lesion impact scores & join 

table_lesion = readtable(lesionXLSX);
% required + optional covariates
req = {'patient_id','lesion_score'};
opt = {'infarct_volume_ml','brain_volume_ml', 'Age', 'Sex'};

if ~all(ismember(req, lower(string(table_lesion.Properties.VariableNames))))
     error('lesion_scores.xlsx must contain at least: patient_id, Lesion_Score');
end
table_lesion.patient_id = string(table_lesion.patient_id);

table_all = innerjoin(table_network, table_lesion, 'Keys','patient_id');
if isempty(table_all), error('No patient_id matches between connectivity and lesion sheet.'); end

% log-transform lesion
%eps = 0.000000001;
eps=0;
table_all.log_lesion = log10(table_all.lesion_score + eps);

% --- Correlations (raw + log)
[xr, xp]     = corr(table_all.Eglob, table_all.lesion_score, 'Type','Pearson',  'Rows','complete');
[xrho, xpp]  = corr(table_all.Eglob, table_all.lesion_score, 'Type','Spearman', 'Rows','complete');
[lr, lp]     = corr(table_all.Eglob, table_all.log_lesion,   'Type','Pearson',  'Rows','complete');
[lrho, lpp]  = corr(table_all.Eglob, table_all.log_lesion,   'Type','Spearman', 'Rows','complete');

% Univariate regression: Eglob ~ lesion
modelspec_raw = 'Eglob ~ lesion_score';
mdl_raw = fitlm(table_all, modelspec_raw);
disp(mdl_raw)

% Univariate regression: Eglob ~ log(lesion)
modelspec_uni = 'Eglob ~ log_lesion';
mdl_uni = fitlm(table_all, modelspec_uni);
disp(mdl_uni)

% Standardized (β): z-score both variables
modelspec_std = 'z_Eglob ~ z_log_lesion';
table_all.z_Eglob = zscore(table_all.Eglob);
table_all.z_log_lesion = zscore(table_all.log_lesion);
mdl_std = fitlm(table_all, modelspec_std);
disp(mdl_std)

% --- Covariate model: Eglob ~ lesion + covs
have = ismember(lower(opt), lower(string(table_all.Properties.VariableNames)));
covNames = opt(have);
mdl_cov_raw = [];
if ~isempty(covNames)
    rhs_raw = strjoin(['lesion_score', covNames], ' + ');
    formula_raw = sprintf('Eglob ~ %s', rhs_raw);
    mdl_cov_raw = fitlm(table_all, formula_raw);
    disp(mdl_cov_raw)
end

% --- Covariate model: Eglob ~ log_lesion + covs
have = ismember(lower(opt), lower(string(table_all.Properties.VariableNames)));
covNames = opt(have);
mdl_cov = [];
if ~isempty(covNames)
    rhs = strjoin(['log_lesion', covNames], ' + ');
    formula = sprintf('Eglob ~ %s', rhs);
    mdl_cov = fitlm(table_all, formula);
    disp(mdl_cov)
end


%% ========= PLOTS =========
bandColor = [0.85 0.90 1.00];
lineWidth = 2;
markerSize = 60;

% % ===== 1) UNIVARIATE: raw lesion vs Eglob =====
% xfit = linspace(min(table_all.lesion_score), max(table_all.lesion_score), 200)';
% Tpred = table(xfit, 'VariableNames', {'lesion_score'});
% [yfit, yCI] = predict(mdl_raw, Tpred, 'Alpha', 0.05);
% coeffs = mdl_raw.Coefficients.Estimate;
% p_slope = mdl_raw.Coefficients.pValue(2);
% R2 = mdl_raw.Rsquared.Ordinary;
% eqn_raw = sprintf('y = %.3f %+.8f·x', coeffs(1), coeffs(2));
% 
% figure('Color','w'); hold on
% hData = scatter(table_all.lesion_score, table_all.Eglob, markerSize, 'filled');
% hPatch = fill([xfit; flipud(xfit)], [yCI(:,1); flipud(yCI(:,2))], ...
%               bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
% hLine = plot(xfit, yfit, 'LineWidth', lineWidth);
% 
% ax = gca;
% ax.FontSize = 14;        
% ax.LineWidth = 1.25;            
% grid on; box on
% xlabel('Lesion impact score', 'FontSize', 18, 'FontWeight', 'bold'); 
% ylabel('Global network efficiency', 'FontSize', 18, 'FontWeight', 'bold');
% title('Univariate: Eglob vs Lesion (CSD, L_{max}=6)', 'FontSize', 20, 'FontWeight', 'bold');
% leg = legend([hData hLine hPatch], {'Data', eqn_raw, '95% CI'}, ...
%              'Location','southeast', ...       
%              'Interpreter','none', ...
%              'FontSize',16);                    
% 
% saveas(gcf, figScatterRaw);
% 
% % ===== 2) UNIVARIATE: log(lesion) vs Eglob =====
% xfit_uni   = linspace(min(table_all.log_lesion), max(table_all.log_lesion), 200)';
% Tpred_uni  = table(xfit_uni, 'VariableNames', {'log_lesion'});
% [yfit_uni, yCI_uni] = predict(mdl_uni, Tpred_uni, 'Alpha', 0.05);
% coeffs_uni = mdl_uni.Coefficients.Estimate;
% p_slope_uni = mdl_uni.Coefficients.pValue(2);
% R2_uni = mdl_uni.Rsquared.Ordinary;
% eqn_uni   = sprintf('y = %.3f %+.8f·x', coeffs_uni(1), coeffs_uni(2));
% 
% figure('Color','w'); hold on
% hData2  = scatter(table_all.log_lesion, table_all.Eglob, markerSize, 'filled');
% hPatch2 = fill([xfit_uni; flipud(xfit_uni)], [yCI_uni(:,1); flipud(yCI_uni(:,2))], ...
%                bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
% hLine2  = plot(xfit_uni, yfit_uni, 'LineWidth', lineWidth);
% 
% ax = gca;
% ax.FontSize = 14;
% ax.LineWidth = 1.25;
% grid on; box on
% xlabel('log(Lesion impact score)', 'FontSize', 18, 'FontWeight', 'bold');
% ylabel('Global network efficiency', 'FontSize', 18, 'FontWeight', 'bold');
% title('Univariate: Eglob vs log(Lesion)', 'FontSize', 20, 'FontWeight', 'bold');
% leg2 = legend([hData2 hLine2 hPatch2], {'Data', eqn_uni, '95% CI'}, ...
%              'Location','southeast', ...       
%              'Interpreter','none', ...
%              'FontSize',16);                   
% 
% saveas(gcf, figScatterLog);
% 
% 

% % ===== 3) COVARIATES: Marginal prediction vs raw lesion impact score =====
% if ~isempty(covNames)
%     % Prediction grid for lesion_score while holding covariates constant
%     xgrid_raw = linspace(min(table_all.lesion_score), max(table_all.lesion_score), 200)';
%     Tmarg_raw = table(xgrid_raw, 'VariableNames', {'lesion_score'});
%     for c = covNames
%         cn = string(c);
%         v  = table_all.(cn);
%         if isnumeric(v)
%             Tmarg_raw.(cn) = repmat(mean(v, 'omitnan'), size(xgrid_raw));
%         else
%             % for categorical/char/string covariates: use mode
%             if iscell(v) || isstring(v)
%                 u = unique(string(v));
%                 [~,idx] = max(arrayfun(@(s) sum(string(v)==s), u));
%                 Tmarg_raw.(cn) = repmat(u(idx), size(xgrid_raw));
%             else
%                 try
%                     Tmarg_raw.(cn) = repmat(mode(v), size(xgrid_raw));
%                 catch
%                     Tmarg_raw.(cn) = v(1); % fallback
%                 end
%             end
%         end
%     end
% 
%     % Predictions and confidence intervals
%     [yhat_raw, yCI_raw] = predict(mdl_cov_raw, Tmarg_raw, 'Alpha', 0.05);
% 
%     % Extract slope, intercept, and stats for the raw lesion term
%     coeffsC_raw = mdl_cov_raw.Coefficients.Estimate;
%     rowLesion = strcmp(mdl_cov_raw.Coefficients.Row, 'lesion_score');
%     beta0_raw  = coeffsC_raw(1);
%     beta1_raw  = mdl_cov_raw.Coefficients.Estimate(rowLesion);
%     p_slope_raw = mdl_cov_raw.Coefficients.pValue(rowLesion);
%     R2_raw      = mdl_cov_raw.Rsquared.Ordinary;
%     adjR2_raw   = mdl_cov_raw.Rsquared.Adjusted;
%     eqn_rawcov  = sprintf('y = %.3f %+.4f·x', beta0_raw, beta1_raw);
%     statStr_raw = sprintf('%s\nR² = %.3f (adj. %.3f),  p = %.3g', eqn_rawcov, R2_raw, adjR2_raw, p_slope_raw);
% 
%     % === Plot ===
%     figure('Color','w'); hold on
%     hData_raw  = scatter(table_all.lesion_score, table_all.Eglob, markerSize, 'filled');
%     hPatch_raw = fill([xgrid_raw; flipud(xgrid_raw)], [yCI_raw(:,1); flipud(yCI_raw(:,2))], ...
%                       bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
%     hLine_raw  = plot(xgrid_raw, yhat_raw, 'LineWidth', lineWidth);
% 
%     % Axis and layout styling 
%     ax = gca; 
%     ax.FontSize = 14; 
%     ax.LineWidth = 1.25;
%     grid on; box on
% 
%     xlabel('Lesion impact score', 'FontSize', 18, 'FontWeight', 'bold');
%     ylabel('Global network efficiency', 'FontSize', 18, 'FontWeight', 'bold');
%     title('Global efficiency vs lesion impact score (CSD, L_{max}=6)', ...
%           'FontSize', 20, 'FontWeight', 'bold');
%     leg3 = legend([hData_raw hLine_raw hPatch_raw], {'Data', eqn_rawcov, '95% CI'}, ...
%                  'Location','southeast', ...
%                  'Interpreter','none', ...
%                  'FontSize',16);
%     saveas(gcf, figCovMarginalRaw);
% end


% % ===== 4) COVARIATES: Added-variable (partial) plot for log_lesion =====
% if ~isempty(mdl_cov)
%     % Residualize Y and X against other covariates (excl. log_lesion)
%     covOnly = setdiff(covNames, 'log_lesion');  % just in case
%     if ~isempty(covOnly)
%         mdl_y_res = fitlm(table_all, sprintf('Eglob ~ %s', strjoin(covOnly,' + ')));
%         mdl_x_res = fitlm(table_all, sprintf('log_lesion ~ %s', strjoin(covOnly,' + ')));
%         y_res = mdl_y_res.Residuals.Raw;
%         x_res = mdl_x_res.Residuals.Raw;
%     else
%         y_res = table_all.Eglob;
%         x_res = table_all.log_lesion;
%     end
% 
%     % Partial regression
%     Tpartial   = table(x_res, y_res, 'VariableNames', {'x','y'});
%     mdl_partial = fitlm(Tpartial, 'y ~ x');
% 
%     xg = linspace(min(x_res), max(x_res), 200)';
%     [yhat_p, yCI_p] = predict(mdl_partial, table(xg,'VariableNames',{'x'}));
% 
%     coeffsP        = mdl_partial.Coefficients.Estimate;
%     p_slope_partial = mdl_partial.Coefficients.pValue(2);
%     R2_partial      = mdl_partial.Rsquared.Ordinary;
%     eqn_partial     = sprintf('y = %.3f %+.8f·x', coeffsP(1), coeffsP(2));
% 
%     % Plot
%     figure('Color','w'); hold on
%     hData3  = scatter(x_res, y_res, markerSize, 'filled');
%     hPatch3 = fill([xg; flipud(xg)], [yCI_p(:,1); flipud(yCI_p(:,2))], ...
%                    bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
%     hLine3  = plot(xg, yhat_p, 'LineWidth', lineWidth);
% 
%     ax = gca; ax.FontSize = 14; ax.LineWidth = 1.25;
% 
%     grid on; box on
%     xlabel('Partial log(lesion) (residualized)', 'FontSize', 18, 'FontWeight', 'bold');
%     ylabel('Partial Eglob (residualized)',      'FontSize', 18, 'FontWeight', 'bold');
%     title('Covariates model: added-variable plot for log(lesion)', 'FontSize', 20, 'FontWeight', 'bold');
%     leg4 = legend([hData3 hLine3 hPatch3], {'Data', eqn_partial, '95% CI'}, ...
%              'Location','southeast', ...       
%              'Interpreter','none', ...
%              'FontSize',16);     
%     saveas(gcf, figCovAdded);

%     % ===== 5) COVARIATES: Marginal prediction vs log_lesion =====
%     xgrid = linspace(min(table_all.log_lesion), max(table_all.log_lesion), 200)';
%     Tmarg = table(xgrid, 'VariableNames', {'log_lesion'});
%     for c = covNames
%         cn = string(c);
%         v  = table_all.(cn);
%         if isnumeric(v)
%             Tmarg.(cn) = repmat(mean(v,'omitnan'), size(xgrid));
%         else
%             if iscell(v) || isstring(v)
%                 u = unique(string(v));
%                 [~,idx] = max(arrayfun(@(s) sum(string(v)==s), u));
%                 Tmarg.(cn) = repmat(u(idx), size(xgrid));
%             else
%                 try
%                     Tmarg.(cn) = repmat(mode(v), size(xgrid));
%                 catch
%                     Tmarg.(cn) = v(1);
%                 end
%             end
%         end
%     end
% 
%     [yhat_m, yCI_m] = predict(mdl_cov, Tmarg);
% 
%     coeffsC = mdl_cov.Coefficients.Estimate;
%     rowLog  = strcmp(mdl_cov.Coefficients.Row, 'log_lesion');
%     beta0   = coeffsC(1);
%     beta1   = mdl_cov.Coefficients.Estimate(rowLog);
%     p_slope_cov = mdl_cov.Coefficients.pValue(rowLog);
%     R2_cov      = mdl_cov.Rsquared.Ordinary;
%     eqn_cov   = sprintf('y = %.3f %+.4f·x', beta0, beta1);
% 
%     % Plot
%     figure('Color','w'); hold on
%     hData4  = scatter(table_all.log_lesion, table_all.Eglob, markerSize, 'filled');
%     hPatch4 = fill([xgrid; flipud(xgrid)], [yCI_m(:,1); flipud(yCI_m(:,2))], ...
%                    bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
%     hLine4  = plot(xgrid, yhat_m, 'LineWidth', lineWidth);
% 
%     ax = gca; ax.FontSize = 14; ax.LineWidth = 1.25;
% 
%     grid on; box on
%     xlabel('log(Lesion impact score)', 'FontSize', 18, 'FontWeight', 'bold');
%     ylabel('Global network efficiency', 'FontSize', 18, 'FontWeight', 'bold');
%     % title('Covariates model: marginal prediction vs log(lesion)', 'FontSize', 20, 'FontWeight', 'bold');
%     title('Global efficiency vs lesion impact score (CSD, L_{max}=6)', 'FontSize', 20, 'FontWeight', 'bold');
%     leg5 = legend([hData4 hLine4 hPatch4], {'Data', eqn_cov, '95% CI'}, ...
%              'Location','southeast', ...       
%              'Interpreter','none', ...
%              'FontSize',16);     
%     saveas(gcf, figCovMarginal);
% end


%% ===== Betweenness centrality (BC) vs lesion impact score =====
% % Univariate: BC_mean ~ log_lesion
% mdl_bc_uni = fitlm(table_all, 'BC_mean ~ log_lesion');
% 
% xfit_bc  = linspace(min(table_all.log_lesion), max(table_all.log_lesion), 200)';
% Tpred_bc = table(xfit_bc, 'VariableNames', {'log_lesion'});
% [yfit_bc, yCI_bc] = predict(mdl_bc_uni, Tpred_bc, 'Alpha', 0.05);
% 
% coeffs_bc   = mdl_bc_uni.Coefficients.Estimate;
% p_slope_bc  = mdl_bc_uni.Coefficients.pValue(2);
% R2_bc       = mdl_bc_uni.Rsquared.Ordinary;
% eqn_bc      = sprintf('y = %.3f %+.3f·x', coeffs_bc(1), coeffs_bc(2));
% statStr_bc  = sprintf('%s\nR^2 = %.3f,  p = %.3g', eqn_bc, R2_bc, p_slope_bc);
% 
% figure('Color','w'); hold on
% hData_bc  = scatter(table_all.log_lesion, table_all.BC_mean, markerSize, 'filled');
% hPatch_bc = fill([xfit_bc; flipud(xfit_bc)], [yCI_bc(:,1); flipud(yCI_bc(:,2))], ...
%                  bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
% hLine_bc  = plot(xfit_bc, yfit_bc, 'LineWidth', lineWidth);
% 
% ax = gca; ax.FontSize = 14; ax.LineWidth = 1.25;
% grid on; box on
% xlabel('log(Lesion impact score)', 'FontSize', 18, 'FontWeight', 'bold');
% ylabel('Mean betweenness centrality', 'FontSize', 18, 'FontWeight', 'bold');
% title('Univariate: BC vs log(lesion)', 'FontSize', 20, 'FontWeight', 'bold');
% legend([hData_bc hLine_bc hPatch_bc], {'Data', eqn_bc, '95% CI'}, ...
%        'Location','southeast', 'Interpreter','none', 'FontSize',16);
% saveas(gcf, fullfile(outDir, 'uni_log_lesion_vs_BC.png'));

% Covariates model: BC_mean ~ log_lesion + covariates (marginal prediction)
% if ~isempty(covNames)
%     rhs_bc   = strjoin(['log_lesion', covNames], ' + ');
%     formula_bc = sprintf('BC_mean ~ %s', rhs_bc);
%     mdl_bc_cov = fitlm(table_all, formula_bc);
% 
%     % Prediction grid for log_lesion with covariates held constant
%     xgrid_bc = linspace(min(table_all.log_lesion), max(table_all.log_lesion), 200)';
%     Tmarg_bc = table(xgrid_bc, 'VariableNames', {'log_lesion'});
%     for c = covNames
%         cn = string(c); v = table_all.(cn);
%         if isnumeric(v)
%             Tmarg_bc.(cn) = repmat(mean(v,'omitnan'), size(xgrid_bc));
%         else
%             if iscell(v) || isstring(v)
%                 u = unique(string(v));
%                 [~,idx] = max(arrayfun(@(s) sum(string(v)==s), u));
%                 Tmarg_bc.(cn) = repmat(u(idx), size(xgrid_bc));
%             else
%                 try Tmarg_bc.(cn) = repmat(mode(v), size(xgrid_bc));
%                 catch, Tmarg_bc.(cn) = v(1);
%                 end
%             end
%         end
%     end
% 
%     [yhat_bc, yCI_bc_cov] = predict(mdl_bc_cov, Tmarg_bc, 'Alpha', 0.05);
% 
%     % Pull intercept and slope for log_lesion
%     rowLog_bc  = strcmp(mdl_bc_cov.Coefficients.Row, 'log_lesion');
%     beta0_bc   = mdl_bc_cov.Coefficients.Estimate(1);
%     beta1_bc   = mdl_bc_cov.Coefficients.Estimate(rowLog_bc);
%     p_slope_bc_cov = mdl_bc_cov.Coefficients.pValue(rowLog_bc);
%     R2_bc_cov      = mdl_bc_cov.Rsquared.Ordinary;
%     eqn_bc_cov     = sprintf('y = %.2f %+.2f·x', beta0_bc, beta1_bc);
%     statStr_bc_cov = sprintf('%s\nR^2 = %.3f,  p = %.3g', eqn_bc_cov, R2_bc_cov, p_slope_bc_cov);
% 
%     figure('Color','w'); hold on
%     hData_bcc  = scatter(table_all.log_lesion, table_all.BC_mean, markerSize, 'filled');
%     hPatch_bcc = fill([xgrid_bc; flipud(xgrid_bc)], [yCI_bc_cov(:,1); flipud(yCI_bc_cov(:,2))], ...
%                       bandColor, 'EdgeColor','none', 'FaceAlpha',0.5);
%     hLine_bcc  = plot(xgrid_bc, yhat_bc, 'LineWidth', lineWidth);
% 
%     ax = gca; ax.FontSize = 14; ax.LineWidth = 1.25;
%     grid on; box on
%     xlabel('log(Lesion impact score)', 'FontSize', 18, 'FontWeight', 'bold');
%     ylabel('Mean betweenness centrality', 'FontSize', 18, 'FontWeight', 'bold');
%     title('Betweenness centrality vs lesion impact score (CSD, L_{max}=6)', ...
%           'FontSize', 20, 'FontWeight', 'bold');
%     legend([hData_bcc hLine_bcc hPatch_bcc], {'Data', eqn_bc_cov, '95% CI'}, ...
%            'Location','southeast', 'Interpreter','none', 'FontSize',16);
%     saveas(gcf, fullfile(outDir, 'cov_marginal_loglesion_BC.png'));
%     disp(mdl_bc_cov);
% end

%% === Helper functions ===

% --- proportional thresholding, keep weights
function Wd = fix_density(W, d)
    % Keeps the top p proportion of undirected edges by weight (excluding diagonal).
    % Preserves original weights on those edges; others set to 0.
    if d <= 0
        Wd = zeros(size(W));
        return;
    end
    n = size(W, 1);
    [ii, jj, v] = find(triu(W, 1)); % upper triangle
    if isempty(v)
        Wd = zeros(n);
        return;
    end
    m = numel(v);
    k = max(1, round(d * n * (n - 1) / 2)); % how many edges to keep
    k = min(k, m);
    [~, ord] = sort(v, 'descend'); 
    keep = ord(1:k);
    r = ii(keep);
    c = jj(keep);
    wkeep = v(keep);
    Wd = zeros(n);
    idx1 = sub2ind([n, n], r, c);
    idx2 = sub2ind([n, n], c, r);
    Wd(idx1) = wkeep;
    Wd(idx2) = wkeep;
end


%   efficiency_wei: 
%   The global efficiency is the average of inverse shortest path length,
%   and is inversely related to the characteristic path length.
%
%   Inputs:     W, weighted undirected or directed connection matrix
%               local, optional argument, local=0: computes the global efficiency (default).
%   Output:     Eglob, global efficiency (scalar)



