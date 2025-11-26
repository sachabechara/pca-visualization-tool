function pca_builder()
% main entry point - sets up everything
close all; clc;

% colors for teh whole app
colors = struct();
colors.var1 = [0.95, 0.7, 0.75];
colors.var2 = [0.7, 0.85, 0.95];
colors.var3 = [0.85, 0.95, 0.75];
colors.mean = [1, 0.6, 0.4];
colors.pc1 = [1, 0.5, 0.5];
colors.pc2 = [0.6, 0.7, 1];
colors.pc3 = [0.5, 0.9, 0.7];
colors.ellipse = [1, 0.85, 0.4];

% main figure
fig = figure('Position', [50, 50, 1400, 850], 'Color', [0.08, 0.08, 0.08], 'NumberTitle', 'off', 'Name', 'PCA Builder');

% state holds everyting
state = struct();
state.n = 50;
state.stage = 1;
state.pca_step = 0;
state.mean1 = 0; state.sd1 = 1;
state.mean2 = 0; state.sd2 = 1;
state.mean3 = 0; state.sd3 = 1;
state.corr12 = 0.7;
state.corr13 = 0.6;
state.corr23 = 0.5;
state.show_mean = true;
state.show_sd = true;
state.show_deviations = false;
state.show_component_deviations = false;
state.show_ellipsoid = true;
state.show_reference_cube = true;
state.show_grid = true;
state.show_pc_projections = false;
state.color_by_pc1 = false;
state.dot_size = 80;
state.dot_color = [0.95, 0.7, 0.75];
state.deviation_color = [0.6, 0.6, 0.6];
state.color_axes_by_var = false;
state.selected_point_idx = -1;
state.var_pair_2d = [1, 2];
state.interaction_mode = 'rotate';
state.animating = false;
state.colors = colors;
state = generate_data(state);

% 3x1D axes stacked vertically
state.ax1 = axes('Position', [0.05, 0.72, 0.50, 0.18], 'Color', [0.05, 0.05, 0.05]);
state.ax2 = axes('Position', [0.05, 0.42, 0.50, 0.18], 'Color', [0.05, 0.05, 0.05]);
state.ax3 = axes('Position', [0.05, 0.12, 0.50, 0.18], 'Color', [0.05, 0.05, 0.05]);
state.ax_main = axes('Position', [0.05, 0.12, 0.50, 0.78], 'Color', [0.05, 0.05, 0.05]);

% hide main axis - we start in 3x1D mode
set(state.ax_main, 'Visible', 'off');

% build UI and start
create_ui(fig, state);
setappdata(fig, 'state', state);
update_controls_visibility(fig);
update_display(fig);

fprintf('PCA Builder Ready!\n');
end

%% data generation
function state = generate_data(state)
n = state.n;

% correlation values
r12 = state.corr12;
r13 = state.corr13;
r23 = state.corr23;

if state.stage == 1
    % stage 1 = independent vars, no correlation
    state.I1 = state.mean1 + state.sd1 * randn(n, 1);
    state.I2 = state.mean2 + state.sd2 * randn(n, 1);
    state.I3 = state.mean3 + state.sd3 * randn(n, 1);

elseif state.stage == 2
    % stage 2 - check if corr matrix is valid first
    R = [1, r12, r13; r12, 1, r23; r13, r23, 1];
    [~, p] = chol(R);

    if p > 0
        % not valid so we need to fix r23
        min_r23 = r12*r13 - sqrt(max(0, (1-r12^2)*(1-r13^2)));
        max_r23 = r12*r13 + sqrt(max(0, (1-r12^2)*(1-r13^2)));
        r23 = max(min_r23, min(max_r23, r23));
        R = [1, r12, r13; r12, 1, r23; r13, r23, 1];
        [~, p] = chol(R);
    end

    % which var pair
    pair = state.var_pair_2d;
    
    if isequal(pair, [1,2]) && abs(r12) >= 0.999 && state.sd1 > 0 && state.sd2 > 0
        % pair 1-2 with perfect corr
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
        state.I2 = state.mean2 + sign(r12) * state.sd2 * (state.I1 - state.mean1) / state.sd1;
        state.I3 = state.mean3 + state.sd3 * randn(n, 1);
    elseif isequal(pair, [1,3]) && abs(r13) >= 0.999 && state.sd1 > 0 && state.sd3 > 0
        % pair 1-3 with perfect corr
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
        state.I3 = state.mean3 + sign(r13) * state.sd3 * (state.I1 - state.mean1) / state.sd1;
        state.I2 = state.mean2 + state.sd2 * randn(n, 1);
    elseif isequal(pair, [2,3]) && abs(r23) >= 0.999 && state.sd2 > 0 && state.sd3 > 0
        % pair 2-3 with perfect corr
        state.I2 = state.mean2 + state.sd2 * randn(n, 1);
        state.I3 = state.mean3 + sign(r23) * state.sd3 * (state.I2 - state.mean2) / state.sd2;
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
    elseif p == 0 && state.sd1 > 0 && state.sd2 > 0 && state.sd3 > 0
        % matrix valid, use mvnrnd
        Sigma = diag([state.sd1, state.sd2, state.sd3]) * R * diag([state.sd1, state.sd2, state.sd3]);
        mu = [state.mean1, state.mean2, state.mean3];
        raw = mvnrnd(mu, Sigma, n);
        state.I1 = raw(:, 1);
        state.I2 = raw(:, 2);
        state.I3 = raw(:, 3);
    else
        % fallback
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
        z = randn(n, 1);
        state.I2 = state.mean2 + state.sd2 * (r12 * (state.I1 - state.mean1)/state.sd1 + sqrt(max(0, 1-r12^2)) * z);
        state.I3 = state.mean3 + state.sd3 * randn(n, 1);
    end

else
    % stage 3+ full 3D
    R = [1, r12, r13; r12, 1, r23; r13, r23, 1];
    [~, p] = chol(R);

    % check perfect correlations first
    if abs(r12) >= 0.999 && abs(r13) >= 0.999 && abs(r23) >= 0.999
        % all on a line
        t = randn(n, 1);
        state.I1 = state.mean1 + state.sd1 * t;
        state.I2 = state.mean2 + sign(r12) * state.sd2 * t;
        state.I3 = state.mean3 + sign(r13) * state.sd3 * t;
    elseif abs(r12) >= 0.999 && state.sd1 > 0 && state.sd2 > 0
        % perfect r12
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
        state.I2 = state.mean2 + sign(r12) * state.sd2 * (state.I1 - state.mean1) / state.sd1;
        state.I3 = state.mean3 + state.sd3 * randn(n, 1);
    elseif abs(r13) >= 0.999 && state.sd1 > 0 && state.sd3 > 0
        % perfect r13
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
        state.I3 = state.mean3 + sign(r13) * state.sd3 * (state.I1 - state.mean1) / state.sd1;
        state.I2 = state.mean2 + state.sd2 * randn(n, 1);
    elseif abs(r23) >= 0.999 && state.sd2 > 0 && state.sd3 > 0
        % perfect r23
        state.I2 = state.mean2 + state.sd2 * randn(n, 1);
        state.I3 = state.mean3 + sign(r23) * state.sd3 * (state.I2 - state.mean2) / state.sd2;
        state.I1 = state.mean1 + state.sd1 * randn(n, 1);
    elseif p > 0
        % matrix invalid - adjust r23 to fix it
        min_r23 = r12*r13 - sqrt(max(0, (1-r12^2)*(1-r13^2)));
        max_r23 = r12*r13 + sqrt(max(0, (1-r12^2)*(1-r13^2)));
        r23_adjusted = max(min_r23 + 0.001, min(max_r23 - 0.001, r23));
        R = [1, r12, r13; r12, 1, r23_adjusted; r13, r23, 1];
        [~, p2] = chol(R);
        if p2 == 0 && state.sd1 > 0 && state.sd2 > 0 && state.sd3 > 0
            Sigma = diag([state.sd1, state.sd2, state.sd3]) * R * diag([state.sd1, state.sd2, state.sd3]);
            mu = [state.mean1, state.mean2, state.mean3];
            raw = mvnrnd(mu, Sigma, n);
            state.I1 = raw(:, 1);
            state.I2 = raw(:, 2);
            state.I3 = raw(:, 3);
        else
            % last resort
            state.I1 = state.mean1 + state.sd1 * randn(n, 1);
            state.I2 = state.mean2 + state.sd2 * randn(n, 1);
            state.I3 = state.mean3 + state.sd3 * randn(n, 1);
        end
    else
        % matrix valid
        if state.sd1 > 0 && state.sd2 > 0 && state.sd3 > 0
            Sigma = diag([state.sd1, state.sd2, state.sd3]) * R * diag([state.sd1, state.sd2, state.sd3]);
            mu = [state.mean1, state.mean2, state.mean3];
            raw = mvnrnd(mu, Sigma, n);
            state.I1 = raw(:, 1);
            state.I2 = raw(:, 2);
            state.I3 = raw(:, 3);
        else
            state.I1 = state.mean1 * ones(n, 1);
            state.I2 = state.mean2 * ones(n, 1);
            state.I3 = state.mean3 * ones(n, 1);
        end
    end

    % pca stats
    state.mean_vec = [mean(state.I1), mean(state.I2), mean(state.I3)];
    if std(state.I1) > 0.001 && std(state.I2) > 0.001 && std(state.I3) > 0.001
        state.cov = cov([state.I1, state.I2, state.I3]);
        [V, D] = eig(state.cov);
        [eigenvalues, idx] = sort(diag(D), 'descend');
        state.eigenvalues = eigenvalues;
        state.eigenvectors = V(:, idx);
        state.var_exp = eigenvalues / max(sum(eigenvalues), 0.001) * 100;
    else
        state.cov = eye(3) * 0.001;
        state.eigenvalues = [0.001; 0.001; 0.001];
        state.eigenvectors = eye(3);
        state.var_exp = [33.3; 33.3; 33.3];
    end
end

% IMPORTANT: handle SD=0 cases or things break
if state.sd1 == 0, state.I1 = state.mean1 * ones(n, 1); end
if state.sd2 == 0, state.I2 = state.mean2 * ones(n, 1); end
if state.sd3 == 0, state.I3 = state.mean3 * ones(n, 1); end
end

%% UI
function create_ui(fig, state)
x = 0.60; w = 0.38; y = 0.96;
bg = [0.08, 0.08, 0.08];
sep_color = [0.3, 0.3, 0.3];

% stage buttons
uicontrol('Style', 'text', 'String', 'STAGE:', 'Units', 'normalized', ...
    'Position', [x, y, 0.08, 0.025], 'BackgroundColor', bg, ...
    'ForegroundColor', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

stages = {'3x1D', '2D', '3D', 'PCA'};
for i = 1:4
    uicontrol('Style', 'pushbutton', 'String', stages{i}, ...
        'Units', 'normalized', 'Position', [x + 0.08 + (i-1)*0.07, y-0.005, 0.065, 0.035], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'ForegroundColor', 'w', 'FontSize', 10, ...
        'Tag', sprintf('stage_%d', i), ...
        'Callback', @(~,~) change_stage(fig, i));
end

% 2D pair selection buttons
y = y - 0.04;
uicontrol('Style', 'text', 'String', '2D Pair:', 'Units', 'normalized', ...
    'Position', [x, y, 0.08, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.7, 0.7, 0.7], 'FontSize', 9, 'HorizontalAlignment', 'left', ...
    'Tag', 'pair_label', 'Visible', 'off');

pairs = {'1-2', '1-3', '2-3'};
pair_vals = {[1,2], [1,3], [2,3]};
for i = 1:3
    uicontrol('Style', 'pushbutton', 'String', pairs{i}, ...
        'Units', 'normalized', 'Position', [x + 0.08 + (i-1)*0.055, y, 0.05, 0.025], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'ForegroundColor', 'w', 'FontSize', 10, ...
        'Tag', sprintf('pair_%d', i), 'Visible', 'off', ...
        'Callback', @(~,~) change_pair(fig, pair_vals{i}));
end

% ---
y = y - 0.012;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% variables section
y = y - 0.025;
uicontrol('Style', 'text', 'String', '📊 VARIABLES', 'Units', 'normalized', ...
    'Position', [x, y, w, 0.022], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.6, 0.85, 1], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

y = y - 0.045;
y = create_var_controls(fig, x, y, w, 1, state);
y = y - 0.008;
y = create_var_controls(fig, x, y, w, 2, state);
y = y - 0.008;
y = create_var_controls(fig, x, y, w, 3, state);

% ---
y = y - 0.027;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% correlations section
y = y - 0.025;
uicontrol('Style', 'text', 'String', '🔗 CORRELATIONS', 'Units', 'normalized', ...
    'Position', [x, y, w, 0.022], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.6, 0.85, 1], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

y = y - 0.025;
y = create_corr_control(fig, x, y, w, '1-2', state.corr12);
y = create_corr_control(fig, x, y, w, '1-3', state.corr13);
y = create_corr_control(fig, x, y, w, '2-3', state.corr23);

% n slider
y = y - 0.005;
uicontrol('Style', 'text', 'String', 'N:', 'Units', 'normalized', ...
    'Position', [x, y, 0.04, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.8, 0.8, 0.8], 'HorizontalAlignment', 'left');
uicontrol('Style', 'slider', 'Min', 3, 'Max', 200, 'Value', state.n, ...
    'Units', 'normalized', 'Position', [x+0.04, y, 0.15, 0.02], ...
    'BackgroundColor', [0.15, 0.15, 0.15], 'Tag', 'n_slider', ...
    'Callback', @(src,~) update_n(fig, round(src.Value)));
uicontrol('Style', 'text', 'String', sprintf('%d', state.n), ...
    'Units', 'normalized', 'Position', [x+0.20, y, 0.04, 0.02], ...
    'BackgroundColor', bg, 'ForegroundColor', [1, 1, 1], ...
    'FontWeight', 'bold', 'Tag', 'n_val');

% ---
y = y - 0.015;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% action buttons row
y = y - 0.06;
uicontrol('Style', 'pushbutton', 'String', 'New Sample', ...
    'Units', 'normalized', 'Position', [x, y, 0.12, 0.035], ...
    'BackgroundColor', [0.3, 0.15, 0.15], 'ForegroundColor', 'w', 'FontSize', 10, ...
    'Callback', @(~,~) new_sample(fig));

uicontrol('Style', 'pushbutton', 'String', 'CLEAR SELECTION', ...
    'Units', 'normalized', 'Position', [x+0.13, y, 0.14, 0.035], ...
    'BackgroundColor', [0.15, 0.15, 0.4], 'ForegroundColor', 'y', 'FontSize', 9, 'FontWeight', 'bold', ...
    'Callback', @(~,~) clear_selection(fig));

uicontrol('Style', 'pushbutton', 'String', 'MODE: ROTATE', ...
    'Units', 'normalized', 'Position', [x+0.28, y, 0.10, 0.035], ...
    'BackgroundColor', [0.15, 0.3, 0.15], 'ForegroundColor', 'w', 'FontSize', 9, 'FontWeight', 'bold', ...
    'Tag', 'mode_toggle', ...
    'Callback', @(~,~) toggle_mode(fig));

% ---
y = y - 0.015;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% appearance section
y = y - 0.045;
uicontrol('Style', 'text', 'String', '🎨 APPEARANCE', 'Units', 'normalized', ...
    'Position', [x, y, w, 0.022], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.6, 0.85, 1], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

% dot size slider
y = y - 0.028;
uicontrol('Style', 'text', 'String', 'Size:', 'Units', 'normalized', ...
    'Position', [x, y, 0.05, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.8, 0.8, 0.8], 'HorizontalAlignment', 'left');
uicontrol('Style', 'slider', 'Min', 20, 'Max', 200, 'Value', state.dot_size, ...
    'Units', 'normalized', 'Position', [x+0.05, y, 0.12, 0.02], ...
    'BackgroundColor', [0.15, 0.15, 0.15], ...
    'Callback', @(src,~) update_dot_size(fig, round(src.Value)));

% dot color presets
y = y - 0.028;
uicontrol('Style', 'text', 'String', 'Dots:', 'Units', 'normalized', ...
    'Position', [x, y, 0.05, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.8, 0.8, 0.8], 'HorizontalAlignment', 'left');
color_presets = {[0.95, 0.7, 0.75]; [0.7, 0.85, 0.95]; [0.85, 0.95, 0.75]; [1, 0.8, 0.4]; [0.9, 0.5, 0.5]; [0.7, 0.5, 0.9]; [1, 1, 1]; [0.3, 0.8, 0.8]};
for i = 1:8
    col = color_presets{i};
    uicontrol('Style', 'pushbutton', 'String', '', ...
        'Units', 'normalized', 'Position', [x+0.05+(i-1)*0.032, y, 0.028, 0.022], ...
        'BackgroundColor', col, 'Callback', @(~,~) update_dot_color(fig, col));
end

% deviation line colors
y = y - 0.028;
uicontrol('Style', 'text', 'String', 'Devs:', 'Units', 'normalized', ...
    'Position', [x, y, 0.05, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.8, 0.8, 0.8], 'HorizontalAlignment', 'left');
for i = 1:8
    col = color_presets{i};
    uicontrol('Style', 'pushbutton', 'String', '', ...
        'Units', 'normalized', 'Position', [x+0.05+(i-1)*0.032, y, 0.028, 0.022], ...
        'BackgroundColor', col, 'Callback', @(~,~) update_deviation_color(fig, col));
end

% color axes checkbox
y = y - 0.028;
uicontrol('Style', 'checkbox', 'String', 'Color Axes by Variable', 'Value', state.color_axes_by_var, ...
    'Units', 'normalized', 'Position', [x, y, 0.20, 0.022], ...
    'BackgroundColor', bg, 'ForegroundColor', 'w', 'FontSize', 9, ...
    'Callback', @(src,~) toggle_option(fig, 'color_axes_by_var', src.Value));

% ---
y = y - 0.012;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% display toggles section
y = y - 0.055;
uicontrol('Style', 'text', 'String', '👁️ DISPLAY', 'Units', 'normalized', ...
    'Position', [x, y, w, 0.022], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.6, 0.85, 1], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

% checkboxes in 3 columns
y = y - 0.02;
toggles = {'show_mean', 'Mean'; 'show_sd', 'SD'; 'show_deviations', 'Devs'; ...
           'show_component_deviations', 'Comps'; 'show_ellipsoid', 'Ellipse'; 'show_reference_cube', 'Cube'; ...
           'show_grid', 'Grid'; 'show_pc_projections', 'Proj'; 'color_by_pc1', 'PC1 Col'};
col_w = w / 3;
for i = 1:size(toggles, 1)
    row = floor((i-1) / 3);
    col = mod(i-1, 3);
    uicontrol('Style', 'checkbox', 'String', toggles{i,2}, 'Value', state.(toggles{i,1}), ...
        'Units', 'normalized', 'Position', [x + col*col_w, y - row*0.026, col_w-0.01, 0.024], ...
        'BackgroundColor', bg, 'ForegroundColor', 'w', 'FontSize', 9, ...
        'Tag', toggles{i,1}, 'Callback', @(src,~) toggle_option(fig, toggles{i,1}, src.Value));
end

% ---
y = y - 0.07;
uipanel('Parent', fig, 'Units', 'normalized', 'Position', [x, y, w, 0.002], ...
    'BackgroundColor', sep_color, 'BorderType', 'none');

% pca steps - only visible in stage 4
y = y - 0.012;
h1 = uicontrol('Style', 'text', 'String', '🎯 PCA STEPS', 'Units', 'normalized', ...
    'Position', [x, y, w, 0.022], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.6, 0.85, 1], 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
    'Tag', 'pca_label', 'Visible', 'off');

y = y - 0.032;
steps = {'None', 'Ellip', '+PC1', '+PC2', '+PC3', 'All'};
pca_handles = h1;
for i = 1:6
    h = uicontrol('Style', 'pushbutton', 'String', steps{i}, 'Units', 'normalized', ...
        'Position', [x + (i-1)*0.062, y, 0.058, 0.028], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'ForegroundColor', 'w', 'FontSize', 10, ...
        'Tag', sprintf('pca_%d', i-1), 'Visible', 'off', ...
        'Callback', @(~,~) set_pca(fig, i-1));
    pca_handles = [pca_handles, h];
end
setappdata(fig, 'pca_handles', pca_handles);

% help button
y = y - 0.038;
uicontrol('Style', 'pushbutton', 'String', '📚 HELP', ...
    'Units', 'normalized', 'Position', [x, y, 0.18, 0.032], ...
    'BackgroundColor', [0.2, 0.35, 0.5], 'ForegroundColor', 'w', ...
    'FontSize', 11, 'FontWeight', 'bold', ...
    'Callback', @(~,~) open_help_window(fig));
end

function y = create_var_controls(fig, x, y, w, var_num, state)
colors = {[0.95, 0.7, 0.75], [0.7, 0.85, 0.95], [0.85, 0.95, 0.75]};
col = colors{var_num};
bg = [0.08, 0.08, 0.08];
mean_val = state.(sprintf('mean%d', var_num));
sd_val = state.(sprintf('sd%d', var_num));

uicontrol('Style', 'text', 'String', sprintf('Var %d:', var_num), 'Units', 'normalized', ...
    'Position', [x, y, 0.06, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', col, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

uicontrol('Style', 'text', 'String', 'μ', 'Units', 'normalized', ...
    'Position', [x+0.06, y, 0.02, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.7, 0.7, 0.7], 'HorizontalAlignment', 'center');
uicontrol('Style', 'slider', 'Min', -10, 'Max', 10, 'Value', mean_val, ...
    'Units', 'normalized', 'Position', [x+0.08, y, 0.10, 0.02], ...
    'BackgroundColor', [0.15, 0.15, 0.15], 'Tag', sprintf('mean%d_slider', var_num), ...
    'Callback', @(src,~) update_param(fig, sprintf('mean%d', var_num), src.Value));
uicontrol('Style', 'text', 'String', sprintf('%.1f', mean_val), ...
    'Units', 'normalized', 'Position', [x+0.19, y, 0.04, 0.02], ...
    'BackgroundColor', bg, 'ForegroundColor', col, 'FontWeight', 'bold', ...
    'Tag', sprintf('mean%d_val', var_num));

uicontrol('Style', 'text', 'String', 'σ', 'Units', 'normalized', ...
    'Position', [x+0.24, y, 0.02, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.7, 0.7, 0.7], 'HorizontalAlignment', 'center');
uicontrol('Style', 'slider', 'Min', 0, 'Max', 5, 'Value', sd_val, ...
    'Units', 'normalized', 'Position', [x+0.26, y, 0.08, 0.02], ...
    'BackgroundColor', [0.15, 0.15, 0.15], 'Tag', sprintf('sd%d_slider', var_num), ...
    'Callback', @(src,~) update_param(fig, sprintf('sd%d', var_num), src.Value));
uicontrol('Style', 'text', 'String', sprintf('%.1f', sd_val), ...
    'Units', 'normalized', 'Position', [x+0.35, y, 0.03, 0.02], ...
    'BackgroundColor', bg, 'ForegroundColor', col, 'FontWeight', 'bold', ...
    'Tag', sprintf('sd%d_val', var_num));
y = y - 0.035;
end

function y = create_corr_control(fig, x, y, w, label, val)
tag = ['corr', strrep(label, '-', '')];
bg = [0.08, 0.08, 0.08];
uicontrol('Style', 'text', 'String', ['r' label ':'], 'Units', 'normalized', ...
    'Position', [x, y, 0.06, 0.02], 'BackgroundColor', bg, ...
    'ForegroundColor', [0.8, 0.8, 0.8], 'FontSize', 10, 'HorizontalAlignment', 'left', ...
    'Tag', [tag '_label'], 'Visible', 'off');
uicontrol('Style', 'slider', 'Min', -1, 'Max', 1, 'Value', val, ...
    'Units', 'normalized', 'Position', [x+0.06, y, 0.15, 0.02], ...
    'BackgroundColor', [0.15, 0.15, 0.15], 'Tag', [tag '_slider'], 'Visible', 'off', ...
    'Callback', @(src,~) update_corr(fig, tag, src.Value));
uicontrol('Style', 'text', 'String', sprintf('%.2f', val), ...
    'Units', 'normalized', 'Position', [x+0.22, y, 0.05, 0.02], ...
    'BackgroundColor', bg, 'ForegroundColor', [1, 1, 1], 'FontWeight', 'bold', ...
    'Tag', [tag '_val'], 'Visible', 'off');
y = y - 0.032;
end

%% callbacks
function change_stage(fig, new_stage)
state = getappdata(fig, 'state');
state.stage = new_stage;
state.selected_point_idx = -1;
state.animating = false;
if new_stage == 4, state.pca_step = 0; end
state = generate_data(state);
setappdata(fig, 'state', state);
update_controls_visibility(fig);
update_display(fig);
end

function change_pair(fig, pair)
state = getappdata(fig, 'state');
state.var_pair_2d = pair;
state.selected_point_idx = -1;
state.animating = false;
state = generate_data(state);
setappdata(fig, 'state', state);
update_pair_buttons(fig, pair);
update_display(fig);
end

function update_pair_buttons(fig, pair)
pairs = {[1,2], [1,3], [2,3]};
for i = 1:3
    btn = findobj(fig, 'Tag', sprintf('pair_%d', i));
    if ~isempty(btn)
        if isequal(pairs{i}, pair)
            set(btn, 'BackgroundColor', [0.4, 0.25, 0.25]);
        else
            set(btn, 'BackgroundColor', [0.15, 0.15, 0.15]);
        end
    end
end
end

function update_param(fig, name, val)
state = getappdata(fig, 'state');
state.(name) = val;
state.animating = false;
label = findobj(fig, 'Tag', [name '_val']);
if ~isempty(label), set(label, 'String', sprintf('%.1f', val)); end
state = generate_data(state);
setappdata(fig, 'state', state);
update_display(fig);
end

function update_corr(fig, tag, val)
state = getappdata(fig, 'state');
state.(tag) = val;
state.animating = false;
label = findobj(fig, 'Tag', [tag '_val']);
if ~isempty(label), set(label, 'String', sprintf('%.2f', val)); end
state = generate_data(state);
setappdata(fig, 'state', state);
update_display(fig);
end

function update_n(fig, val)
state = getappdata(fig, 'state');
state.n = val;
state.animating = false;
label = findobj(fig, 'Tag', 'n_val');
if ~isempty(label), set(label, 'String', sprintf('%d', val)); end
state = generate_data(state);
setappdata(fig, 'state', state);
update_display(fig);
end

function update_dot_size(fig, val)
state = getappdata(fig, 'state');
state.dot_size = val;
setappdata(fig, 'state', state);
update_display(fig);
end

function update_dot_color(fig, col)
state = getappdata(fig, 'state');
state.dot_color = col;
setappdata(fig, 'state', state);
update_display(fig);
end

function update_deviation_color(fig, col)
state = getappdata(fig, 'state');
state.deviation_color = col;
setappdata(fig, 'state', state);
update_display(fig);
end

function toggle_option(fig, name, val)
state = getappdata(fig, 'state');
state.(name) = logical(val);
setappdata(fig, 'state', state);
update_display(fig);
end

function new_sample(fig)
state = getappdata(fig, 'state');
state.selected_point_idx = -1;
state.animating = false;
state = generate_data(state);
setappdata(fig, 'state', state);
update_display(fig);
end

function clear_selection(fig)
state = getappdata(fig, 'state');
state.selected_point_idx = -1;
state.animating = false;
setappdata(fig, 'state', state);
update_display(fig);
fprintf('Selection cleared!\n');
end

function toggle_mode(fig)
state = getappdata(fig, 'state');
if strcmp(state.interaction_mode, 'rotate')
    state.interaction_mode = 'select';
else
    state.interaction_mode = 'rotate';
end
setappdata(fig, 'state', state);
update_mode_button(fig);
update_display(fig);
end

function update_mode_button(fig)
state = getappdata(fig, 'state');
btn = findobj(fig, 'Tag', 'mode_toggle');
if ~isempty(btn)
    if strcmp(state.interaction_mode, 'select')
        set(btn, 'String', 'MODE: SELECT', 'BackgroundColor', [0.3, 0.15, 0.3]);
    else
        set(btn, 'String', 'MODE: ROTATE', 'BackgroundColor', [0.15, 0.3, 0.15]);
    end
end
end

function set_pca(fig, step)
state = getappdata(fig, 'state');
old_step = state.pca_step;
state.pca_step = step;
setappdata(fig, 'state', state);
update_pca_buttons(fig, step);

% animate when stepping through pca
if step > old_step && step >= 2
    animate_pca_variance(fig, old_step, step);
else
    update_display(fig);
end
end

function update_pca_buttons(fig, step)
for i = 0:5
    btn = findobj(fig, 'Tag', sprintf('pca_%d', i));
    if ~isempty(btn)
        if i == step
            set(btn, 'BackgroundColor', [0.4, 0.25, 0.25]);
        else
            set(btn, 'BackgroundColor', [0.15, 0.15, 0.15]);
        end
    end
end
end

function update_controls_visibility(fig)
state = getappdata(fig, 'state');

% stage buttons - highlight current one
for i = 1:4
    btn = findobj(fig, 'Tag', sprintf('stage_%d', i));
    if ~isempty(btn)
        if i == state.stage
            set(btn, 'BackgroundColor', [0.5, 0.3, 0.3]);
        else
            set(btn, 'BackgroundColor', [0.15, 0.15, 0.15]);
        end
    end
end

% 2D pair buttons - only show in stage 2
pair_tags = {'pair_label', 'pair_1', 'pair_2', 'pair_3'};
vis = 'off'; if state.stage == 2, vis = 'on'; end
for i = 1:length(pair_tags)
    obj = findobj(fig, 'Tag', pair_tags{i});
    if ~isempty(obj), set(obj, 'Visible', vis); end
end
if state.stage == 2
    update_pair_buttons(fig, state.var_pair_2d);
end

% corr sliders - show for stages 2+
corr_tags = {'corr12_label', 'corr12_slider', 'corr12_val', ...
             'corr13_label', 'corr13_slider', 'corr13_val', ...
             'corr23_label', 'corr23_slider', 'corr23_val'};
vis = 'off'; if state.stage >= 2, vis = 'on'; end
for i = 1:length(corr_tags)
    obj = findobj(fig, 'Tag', corr_tags{i});
    if ~isempty(obj), set(obj, 'Visible', vis); end
end

% pca step buttons - only stage 4
handles = getappdata(fig, 'pca_handles');
vis = 'off'; if state.stage == 4, vis = 'on'; end
if ~isempty(handles)
    set(handles, 'Visible', vis);
end
end

%% display
function update_display(fig)
state = getappdata(fig, 'state');

if state.stage == 1
    % stage 1: three 1D plots stacked, hide main axis
    cla(state.ax_main, 'reset');
    set(state.ax_main, 'Visible', 'off');
    
    % show the three 1D axes
    set(state.ax1, 'Visible', 'on');
    set(state.ax2, 'Visible', 'on');
    set(state.ax3, 'Visible', 'on');

    render_1d(state.ax1, state.I1, 'Variable 1', state.colors.var1, state, state.mean1, state.sd1);
    render_1d(state.ax2, state.I2, 'Variable 2', state.colors.var2, state, state.mean2, state.sd2);
    render_1d(state.ax3, state.I3, 'Variable 3', state.colors.var3, state, state.mean3, state.sd3);
else
    % stages 2-4: show main axis, hide teh 1D ones
    cla(state.ax1, 'reset'); set(state.ax1, 'Visible', 'off');
    cla(state.ax2, 'reset'); set(state.ax2, 'Visible', 'off');
    cla(state.ax3, 'reset'); set(state.ax3, 'Visible', 'off');

    % reset main axis for clean render
    cla(state.ax_main, 'reset');
    set(state.ax_main, 'Visible', 'on');
    set(state.ax_main, 'Color', [0.05, 0.05, 0.05]);
    
    switch state.stage
        case 2, render_2d(state.ax_main, state);
        case 3, render_3d(state.ax_main, state);
        case 4, render_pca(state.ax_main, state);
    end
end
drawnow;
end

%% render 1D
function render_1d(ax, data, label, color, state, param_mean, param_sd)
cla(ax);
hold(ax, 'on');

% axis line
plot(ax, [-10, 10], [0, 0], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 2);

% tick marks
for tick = -10:2:10
    plot(ax, [tick, tick], [-0.1, 0.1], 'Color', [0.5, 0.5, 0.5], 'LineWidth', 1);
    if tick == 0
        text(ax, tick, -0.28, '0', 'Color', [1, 1, 1], 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    else
        text(ax, tick, -0.28, sprintf('%d', tick), 'Color', [0.6, 0.6, 0.6], 'FontSize', 8, 'HorizontalAlignment', 'center');
    end
end

% deviation lines - draw before points so points are on top
if state.show_deviations && param_sd > 0
    for i = 1:length(data)
        plot(ax, [data(i), param_mean], [0, 0], 'Color', [state.deviation_color, 0.6], 'LineWidth', 2);
    end
end

% points on top
scatter(ax, data, zeros(size(data)), state.dot_size, state.dot_color, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);

% mean line
if state.show_mean
    plot(ax, [param_mean, param_mean], [-0.4, 0.4], 'Color', state.colors.mean, 'LineWidth', 4);
    text(ax, param_mean, 0.55, sprintf('%.1f', param_mean), 'Color', state.colors.mean, 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

% sd lines
if state.show_sd && param_sd > 0
    plot(ax, [param_mean-param_sd, param_mean-param_sd], [-0.3, 0.3], 'Color', state.colors.mean, 'LineWidth', 2, 'LineStyle', '--');
    plot(ax, [param_mean+param_sd, param_mean+param_sd], [-0.3, 0.3], 'Color', state.colors.mean, 'LineWidth', 2, 'LineStyle', '--');
end

title(ax, sprintf('%s (μ=%.1f, σ=%.1f)', label, param_mean, param_sd), 'Color', color, 'FontSize', 11, 'FontWeight', 'bold');
xlim(ax, [-10, 10]); ylim(ax, [-0.7, 0.7]);
set(ax, 'YTick', [], 'XTick', [], 'Color', [0.05, 0.05, 0.05], 'XColor', [0.5, 0.5, 0.5], 'YColor', [0.5, 0.5, 0.5]);
hold(ax, 'off');
end

%% render 2D
function render_2d(ax, state)
hold(ax, 'on');
view(ax, 2);

% figure out which var pair we're showing
pair = state.var_pair_2d;
if isequal(pair, [1,2])
    dx = state.I1; dy = state.I2; cx = state.colors.var1; cy = state.colors.var2; lx = 'Var 1'; ly = 'Var 2';
elseif isequal(pair, [1,3])
    dx = state.I1; dy = state.I3; cx = state.colors.var1; cy = state.colors.var3; lx = 'Var 1'; ly = 'Var 3';
else
    dx = state.I2; dy = state.I3; cx = state.colors.var2; cy = state.colors.var3; lx = 'Var 2'; ly = 'Var 3';
end

% ellipse at bottom layer
if state.show_ellipsoid
    plot_ellipse(ax, [mean(dx), mean(dy)], cov([dx, dy]), state.colors.ellipse);
end

% deviation lines - clickable, drawn before points
if state.show_deviations
    mx = mean(dx); my = mean(dy);
    for i = 1:length(dx)
        h_dev = plot(ax, [dx(i), mx], [dy(i), my], 'Color', [state.deviation_color, 0.6], 'LineWidth', 1.5);
        % make clickable for animation
        set(h_dev, 'ButtonDownFcn', @(~,~) click_deviation_2d(ancestor(ax,'figure'), i, dx, dy, cx, cy));
    end
end

% component deviations - x and y parts separately
if state.show_component_deviations
    mx = mean(dx); my = mean(dy);
    for i = 1:length(dx)
        plot(ax, [mx, dx(i)], [my, my], 'Color', [cx, 0.5], 'LineWidth', 1.5);
        plot(ax, [dx(i), dx(i)], [my, dy(i)], 'Color', [cy, 0.5], 'LineWidth', 1.5);
    end
end

% points on top
h = scatter(ax, dx, dy, state.dot_size, state.dot_color, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);
set(h, 'ButtonDownFcn', @(src,~) click_2d(ax, dx, dy));

% mean point
if state.show_mean
    scatter(ax, mean(dx), mean(dy), 200, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
end

% selected point highlight on top of everything
if state.selected_point_idx > 0 && state.selected_point_idx <= length(dx)
    draw_2d_highlight(ax, dx, dy, state.selected_point_idx);
end

title(ax, sprintf('%s vs %s', lx, ly), 'Color', 'w', 'FontSize', 13);
xlabel(ax, lx, 'Color', cx); ylabel(ax, ly, 'Color', cy);
set(ax, 'XLim', [-12, 12], 'YLim', [-12, 12]);
set(ax, 'Color', [0.05, 0.05, 0.05], 'XColor', [0.6, 0.6, 0.6], 'YColor', [0.6, 0.6, 0.6]);
grid(ax, 'on'); set(ax, 'GridColor', [0.3, 0.3, 0.3]);
axis(ax, 'equal');
hold(ax, 'off');
end

function click_2d(ax, dx, dy)
fig = ancestor(ax, 'figure');
state = getappdata(fig, 'state');

% dont allow clicks during animation
if state.animating
    return;
end

pt = get(ax, 'CurrentPoint');
x_click = pt(1,1); y_click = pt(1,2);

distances = sqrt((dx - x_click).^2 + (dy - y_click).^2);
[~, idx] = min(distances);

current_state = getappdata(fig, 'state');

% toggle - click same point again to clear
if idx == current_state.selected_point_idx
    current_state.selected_point_idx = -1;
else
    current_state.selected_point_idx = idx;
end
setappdata(fig, 'state', current_state);
update_display(fig);
end

function click_deviation_2d(fig, idx, dx, dy, cx, cy)
% clicked a deviation line - do the decomposition animation
state = getappdata(fig, 'state');

if state.animating
    return;
end

state.animating = true;
setappdata(fig, 'state', state);

animate_deviation_decomposition_2d(state.ax_main, dx, dy, idx, cx, cy, state);

% done animating
state = getappdata(fig, 'state');
state.animating = false;
setappdata(fig, 'state', state);
end

function animate_deviation_decomposition_2d(ax, dx, dy, idx, cx, cy, state)
% animate breaking down 2D deviation into x and y components
mx = mean(dx);
my = mean(dy);
x = dx(idx);
y = dy(idx);

cla(ax, 'reset');
hold(ax, 'on');
view(ax, 2);

% animation params
n_frames = 50;
pause_time = 0.03;

% phase 1: show original 2D deviation
for frame = 1:15
    cla(ax);
    hold(ax, 'on');

    % faded background
    scatter(ax, dx, dy, state.dot_size*0.5, [0.3, 0.3, 0.3], 'filled', 'MarkerEdgeAlpha', 0.3);

    % mean
    scatter(ax, mx, my, 150, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % selected point
    scatter(ax, x, y, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % 2D deviation line
    plot(ax, [x, mx], [y, my], 'Color', [1, 1, 0], 'LineWidth', 3);

    text(ax, mx + (x-mx)*0.5, my + (y-my)*0.5 + 1, 'Euclidean Deviation', ...
        'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    set(ax, 'XLim', [-12, 12], 'YLim', [-12, 12]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'equal');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

% phase 2: transform mean point into crosshairs
for frame = 1:15
    cla(ax);
    hold(ax, 'on');
    alpha = frame / 15;

    % faded background
    scatter(ax, dx, dy, state.dot_size*0.3, [0.2, 0.2, 0.2], 'filled', 'MarkerEdgeAlpha', 0.2);

    % mean transforms into cross
    point_size = 150 * (1 - alpha);
    if point_size > 5
        scatter(ax, mx, my, point_size, state.colors.mean, 'filled', 'MarkerEdgeAlpha', 1-alpha);
    end

    % vertical line at mean X
    line_extent = 12 * alpha;
    plot(ax, [mx, mx], [my - line_extent, my + line_extent], 'Color', [cx, alpha], 'LineWidth', 3);

    % horizontal line at mean Y
    plot(ax, [mx - line_extent, mx + line_extent], [my, my], 'Color', [cy, alpha], 'LineWidth', 3);

    scatter(ax, x, y, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % fading 2D deviation
    plot(ax, [x, mx], [y, my], 'Color', [1, 1, 0, 1-alpha], 'LineWidth', 3);

    set(ax, 'XLim', [-12, 12], 'YLim', [-12, 12]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'equal');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

% phase 3: show x and y component deviations
for frame = 1:20
    cla(ax);
    hold(ax, 'on');
    alpha = min(1, frame / 10);

    % mean lines
    plot(ax, [mx, mx], [-12, 12], 'Color', cx, 'LineWidth', 3);
    plot(ax, [-12, 12], [my, my], 'Color', cy, 'LineWidth', 3);

    % mean labels
    text(ax, mx, -11, 'μₓ', 'Color', cx, 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    text(ax, -11, my, 'μᵧ', 'Color', cy, 'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    scatter(ax, x, y, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % x component deviation (horizontal)
    plot(ax, [x, mx], [y, y], 'Color', [cx, alpha], 'LineWidth', 4, 'LineStyle', '-');
    if alpha > 0.5
        text(ax, mx + (x-mx)*0.5, y - 0.8, sprintf('Δx = %.1f', x-mx), ...
            'Color', cx, 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
            'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
    end

    % y component deviation (vertical)
    plot(ax, [x, x], [y, my], 'Color', [cy, alpha], 'LineWidth', 4, 'LineStyle', '-');
    if alpha > 0.5
        text(ax, x + 1.2, my + (y-my)*0.5, sprintf('Δy = %.1f', y-my), ...
            'Color', cy, 'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', ...
            'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
    end

    % show reconstructed 2D deviation faintly
    if alpha > 0.7
        plot(ax, [mx, x], [my, y], 'Color', [1, 1, 0, 0.4], 'LineWidth', 2, 'LineStyle', '--');
        euc_dist = sqrt((x-mx)^2 + (y-my)^2);
        text(ax, mx + (x-mx)*0.5, my + (y-my)*0.5 + 1, sprintf('||d|| = %.1f', euc_dist), ...
            'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
            'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
    end

    set(ax, 'XLim', [-12, 12], 'YLim', [-12, 12]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'equal');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

% hold final frame then go back to normal
pause(1.5);

fig = ancestor(ax, 'figure');
update_display(fig);
end

function draw_2d_highlight(ax, dx, dy, idx)
x = dx(idx); y = dy(idx);

% lines from point to axes
plot(ax, [x, x], [y, -12], 'Color', [1, 1, 0], 'LineWidth', 2.5, 'LineStyle', '--');
plot(ax, [x, -12], [y, y], 'Color', [1, 1, 0], 'LineWidth', 2.5, 'LineStyle', '--');

% axis markers
scatter(ax, x, -11, 100, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
scatter(ax, -11, y, 100, 'y', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);

% coordinate labels
text(ax, x, -11.5, sprintf('%.1f', x), 'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
text(ax, -11.5, y, sprintf('%.1f', y), 'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');

% point label
text(ax, x+0.5, y+0.7, sprintf('(%.1f, %.1f)', x, y), 'Color', 'y', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);

% highlight circle
scatter(ax, x, y, 350, 'y', 'LineWidth', 3);
end

%% render 3D
function render_3d(ax, state)
hold(ax, 'on');
view(ax, 3);

% cube frame
if state.show_reference_cube
    draw_cube_frame(ax, state);
end

% grid inside cube
if state.show_grid && state.show_reference_cube
    draw_grid(ax);
end

% axis lines - always visible even without cube
draw_axis_lines(ax, state);

% ellipsoid drawn before deviations so its underneath
if state.show_ellipsoid
    draw_ellipsoid(ax, state.mean_vec, state.cov, state.colors.ellipse, 0.3);
end

% deviations - clickable for animation
if state.show_deviations
    for i = 1:length(state.I1)
        h_dev = plot3(ax, [state.I1(i), state.mean_vec(1)], [state.I2(i), state.mean_vec(2)], [state.I3(i), state.mean_vec(3)], 'Color', [state.deviation_color, 0.6], 'LineWidth', 1.5);
        set(h_dev, 'ButtonDownFcn', @(~,~) click_deviation_3d(ancestor(ax,'figure'), i));
    end
end

% component deviations - x y z separately
if state.show_component_deviations
    mx = state.mean_vec(1); my = state.mean_vec(2); mz = state.mean_vec(3);
    for i = 1:length(state.I1)
        x = state.I1(i); y = state.I2(i); z = state.I3(i);
        plot3(ax, [mx, x], [my, my], [mz, mz], 'Color', [state.colors.var1, 0.6], 'LineWidth', 2);
        plot3(ax, [x, x], [my, y], [mz, mz], 'Color', [state.colors.var2, 0.6], 'LineWidth', 2);
        plot3(ax, [x, x], [y, y], [mz, z], 'Color', [state.colors.var3, 0.6], 'LineWidth', 2);
    end
end

% points on top
h = scatter3(ax, state.I1, state.I2, state.I3, state.dot_size, state.dot_color, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 1);
set(h, 'ButtonDownFcn', @(src,~) click_3d(ax));

% mean point
if state.show_mean
    scatter3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), 200, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
end

% selected point highlight on top
if state.selected_point_idx > 0 && state.selected_point_idx <= length(state.I1)
    draw_3d_highlight(ax, state, state.selected_point_idx);
end

% labels
title(ax, '3D View', 'Color', 'w', 'FontSize', 13);
xlabel(ax, 'Var 1', 'Color', state.colors.var1);
ylabel(ax, 'Var 2', 'Color', state.colors.var2);
zlabel(ax, 'Var 3', 'Color', state.colors.var3);
set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
set(ax, 'Color', [0.05, 0.05, 0.05], 'XColor', [0.6, 0.6, 0.6], 'YColor', [0.6, 0.6, 0.6], 'ZColor', [0.6, 0.6, 0.6]);
axis(ax, 'vis3d');

% rotation mode
if strcmp(state.interaction_mode, 'rotate')
    rotate3d(ax, 'on');
else
    rotate3d(ax, 'off');
end

hold(ax, 'off');
end

function click_3d(ax)
fig = ancestor(ax, 'figure');
state = getappdata(fig, 'state');

% store current view so it doesnt reset
[az, el] = view(ax);

pt = get(ax, 'CurrentPoint');
ray_origin = pt(1,:);
ray_dir = pt(2,:) - pt(1,:);
if norm(ray_dir) > 0
    ray_dir = ray_dir / norm(ray_dir);
end

% find closest point to click ray
min_dist = inf; idx = 1;
for i = 1:length(state.I1)
    p = [state.I1(i), state.I2(i), state.I3(i)];
    v = p - ray_origin;
    t = dot(v, ray_dir);
    closest = ray_origin + t * ray_dir;
    d = norm(p - closest);
    if d < min_dist
        min_dist = d;
        idx = i;
    end
end

% toggle - click same point to clear
if idx == state.selected_point_idx
    state.selected_point_idx = -1;
else
    state.selected_point_idx = idx;
end
setappdata(fig, 'state', state);

update_display(fig);

% restore view after update
state = getappdata(fig, 'state');
view(state.ax_main, az, el);
drawnow;
end

function click_deviation_3d(fig, idx)
% clicked a 3D deviation - animate decomposition
state = getappdata(fig, 'state');

if state.animating
    return;
end

% store view
[az, el] = view(state.ax_main);

state.animating = true;
setappdata(fig, 'state', state);

animate_deviation_decomposition_3d(state.ax_main, idx, state, az, el);

% done
state = getappdata(fig, 'state');
state.animating = false;
setappdata(fig, 'state', state);
end

function animate_deviation_decomposition_3d(ax, idx, state, az, el)
% animate 3D deviation breaking into x y z components
mx = state.mean_vec(1); my = state.mean_vec(2); mz = state.mean_vec(3);
x = state.I1(idx); y = state.I2(idx); z = state.I3(idx);

cla(ax, 'reset');
hold(ax, 'on');
view(ax, az, el);

pause_time = 0.03;

% phase 1: show original 3D euclidean deviation
for frame = 1:15
    cla(ax);
    hold(ax, 'on');
    view(ax, az, el);

    % faded points
    scatter3(ax, state.I1, state.I2, state.I3, state.dot_size*0.4, [0.3, 0.3, 0.3], 'filled', 'MarkerEdgeAlpha', 0.3);

    % mean
    scatter3(ax, mx, my, mz, 150, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % selected point
    scatter3(ax, x, y, z, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % 3D deviation
    plot3(ax, [x, mx], [y, my], [z, mz], 'Color', [1, 1, 0], 'LineWidth', 3);

    euc_dist = sqrt((x-mx)^2 + (y-my)^2 + (z-mz)^2);
    text(ax, mx + (x-mx)*0.5, my + (y-my)*0.5, mz + (z-mz)*0.5 + 1.5, ...
        sprintf('3D Euclidean = %.1f', euc_dist), ...
        'Color', 'y', 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

    set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'vis3d');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

% phase 2: transform mean into 3 planes
for frame = 1:15
    cla(ax);
    hold(ax, 'on');
    view(ax, az, el);
    alpha = frame / 15;

    % faded points
    scatter3(ax, state.I1, state.I2, state.I3, state.dot_size*0.3, [0.2, 0.2, 0.2], 'filled', 'MarkerEdgeAlpha', 0.2);

    % mean shrinking
    point_size = 150 * (1 - alpha);
    if point_size > 5
        scatter3(ax, mx, my, mz, point_size, state.colors.mean, 'filled', 'MarkerEdgeAlpha', 1-alpha);
    end

    % three perpendicular planes growing
    extent = 10 * alpha;

    % YZ plane at mean X
    [YY, ZZ] = meshgrid(my + [-extent, extent], mz + [-extent, extent]);
    XX = mx * ones(size(YY));
    surf(ax, XX, YY, ZZ, 'FaceColor', state.colors.var1, 'FaceAlpha', 0.3*alpha, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    % XZ plane at mean Y
    [XX2, ZZ2] = meshgrid(mx + [-extent, extent], mz + [-extent, extent]);
    YY2 = my * ones(size(XX2));
    surf(ax, XX2, YY2, ZZ2, 'FaceColor', state.colors.var2, 'FaceAlpha', 0.3*alpha, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    % XY plane at mean Z
    [XX3, YY3] = meshgrid(mx + [-extent, extent], my + [-extent, extent]);
    ZZ3 = mz * ones(size(XX3));
    surf(ax, XX3, YY3, ZZ3, 'FaceColor', state.colors.var3, 'FaceAlpha', 0.3*alpha, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    scatter3(ax, x, y, z, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % fading 3D line
    plot3(ax, [x, mx], [y, my], [z, mz], 'Color', [1, 1, 0, 1-alpha], 'LineWidth', 3);

    set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'vis3d');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

% phase 3: show x y z component deviations
for frame = 1:20
    cla(ax);
    hold(ax, 'on');
    view(ax, az, el);
    alpha = min(1, frame / 10);

    % mean planes at full size
    extent = 10;
    [YY, ZZ] = meshgrid(my + [-extent, extent], mz + [-extent, extent]);
    XX = mx * ones(size(YY));
    surf(ax, XX, YY, ZZ, 'FaceColor', state.colors.var1, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    [XX2, ZZ2] = meshgrid(mx + [-extent, extent], mz + [-extent, extent]);
    YY2 = my * ones(size(XX2));
    surf(ax, XX2, YY2, ZZ2, 'FaceColor', state.colors.var2, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    [XX3, YY3] = meshgrid(mx + [-extent, extent], my + [-extent, extent]);
    ZZ3 = mz * ones(size(XX3));
    surf(ax, XX3, YY3, ZZ3, 'FaceColor', state.colors.var3, 'FaceAlpha', 0.25, 'EdgeColor', 'none', 'PickableParts', 'none', 'HitTest', 'off');

    % plane labels
    text(ax, mx - 9, my, mz, 'μₓ', 'Color', state.colors.var1, 'FontSize', 14, 'FontWeight', 'bold');
    text(ax, mx, my - 9, mz, 'μᵧ', 'Color', state.colors.var2, 'FontSize', 14, 'FontWeight', 'bold');
    text(ax, mx, my, mz - 9, 'μz', 'Color', state.colors.var3, 'FontSize', 14, 'FontWeight', 'bold');

    scatter3(ax, x, y, z, state.dot_size*1.5, 'y', 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

    % component deviations
    plot3(ax, [x, mx], [y, y], [z, z], 'Color', [state.colors.var1, alpha], 'LineWidth', 4);
    plot3(ax, [mx, mx], [y, my], [z, z], 'Color', [state.colors.var2, alpha], 'LineWidth', 4);
    plot3(ax, [mx, mx], [my, my], [z, mz], 'Color', [state.colors.var3, alpha], 'LineWidth', 4);

    % component labels
    if alpha > 0.5
        text(ax, mx + (x-mx)*0.5, y, z + 0.8, sprintf('Δx=%.1f', x-mx), ...
            'Color', state.colors.var1, 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
        text(ax, mx, my + (y-my)*0.5, z + 0.8, sprintf('Δy=%.1f', y-my), ...
            'Color', state.colors.var2, 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
        text(ax, mx - 1, my, mz + (z-mz)*0.5, sprintf('Δz=%.1f', z-mz), ...
            'Color', state.colors.var3, 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
    end

    % show reconstructed 3D line faintly
    if alpha > 0.7
        plot3(ax, [mx, x], [my, y], [mz, z], 'Color', [1, 1, 0, 0.4], 'LineWidth', 2, 'LineStyle', '--');
        euc_dist = sqrt((x-mx)^2 + (y-my)^2 + (z-mz)^2);
        text(ax, mx + (x-mx)*0.5, my + (y-my)*0.5, mz + (z-mz)*0.5 + 1.5, ...
            sprintf('||d|| = %.1f', euc_dist), ...
            'Color', 'y', 'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
    end

    set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
    set(ax, 'Color', [0.05, 0.05, 0.05]);
    axis(ax, 'vis3d');
    grid(ax, 'on');
    drawnow;
    pause(pause_time);
end

pause(1.5);

% go back to normal view, keep angle
fig = ancestor(ax, 'figure');
state = getappdata(fig, 'state');
update_display(fig);
view(state.ax_main, az, el);
drawnow;
end

function draw_3d_highlight(ax, state, idx)
x = state.I1(idx); y = state.I2(idx); z = state.I3(idx);

% colored lines to axis planes
plot3(ax, [x, x], [y, y], [-10, z], 'Color', state.colors.var3, 'LineWidth', 3, 'LineStyle', '-');
plot3(ax, [x, x], [-10, y], [z, z], 'Color', state.colors.var2, 'LineWidth', 3, 'LineStyle', '-');
plot3(ax, [-10, x], [y, y], [z, z], 'Color', state.colors.var1, 'LineWidth', 3, 'LineStyle', '-');

% markers on axis planes
scatter3(ax, x, y, -10, 150, state.colors.var3, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
scatter3(ax, x, -10, z, 150, state.colors.var2, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);
scatter3(ax, -10, y, z, 150, state.colors.var1, 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 2);

% axis projection labels
text(ax, x, y, -11, sprintf('%.1f', z), 'Color', state.colors.var3, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x, -11, z, sprintf('%.1f', y), 'Color', state.colors.var2, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, -11, y, z, sprintf('%.1f', x), 'Color', state.colors.var1, 'FontSize', 10, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);

% point coord label with colored values - built up piece by piece
label_str = '(';
text(ax, x+0.5, y+0.5, z+1.8, label_str, 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9], 'Interpreter', 'none');

text(ax, x+0.8, y+0.5, z+1.8, sprintf('%.1f', x), 'Color', state.colors.var1, 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x+1.6, y+0.5, z+1.8, ', ', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x+1.9, y+0.5, z+1.8, sprintf('%.1f', y), 'Color', state.colors.var2, 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x+2.7, y+0.5, z+1.8, ', ', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x+3.0, y+0.5, z+1.8, sprintf('%.1f', z), 'Color', state.colors.var3, 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);
text(ax, x+3.8, y+0.5, z+1.8, ')', 'Color', 'w', 'FontSize', 11, 'FontWeight', 'bold', 'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);

% highlight circle around point
scatter3(ax, x, y, z, 400, 'w', 'LineWidth', 4);
end

%% render PCA
function render_pca(ax, state)
hold(ax, 'on');
view(ax, 3);

% cube frame
if state.show_reference_cube
    draw_cube_frame(ax, state);
end

% grid
if state.show_grid && state.show_reference_cube
    draw_grid(ax);
end

% axis lines always visible
draw_axis_lines(ax, state);

% points - color by PC1 score if enabled
pc1_scores = ([state.I1, state.I2, state.I3] - state.mean_vec) * state.eigenvectors(:,1);
if state.color_by_pc1 && state.pca_step >= 2
    pc1_norm = (pc1_scores - min(pc1_scores)) / (max(pc1_scores) - min(pc1_scores) + 0.001);
    colors_map = [pc1_norm, 0.3*ones(size(pc1_norm)), 1-pc1_norm];
    h = scatter3(ax, state.I1, state.I2, state.I3, state.dot_size, colors_map, 'filled', 'MarkerEdgeColor', 'w');
else
    h = scatter3(ax, state.I1, state.I2, state.I3, state.dot_size, state.dot_color, 'filled', 'MarkerEdgeColor', 'w');
end
set(h, 'ButtonDownFcn', @(src,~) click_3d(ax));

% mean
if state.show_mean
    scatter3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), 200, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);
end

% ellipsoid
if state.pca_step >= 1 && state.show_ellipsoid
    draw_ellipsoid(ax, state.mean_vec, state.cov, state.colors.ellipse, 0.25);
end

% PC axes with spread indicators showing eigenvalue magnitude
scale = 2;
if state.pca_step >= 2
    dir1 = state.eigenvectors(:,1)' * sqrt(state.eigenvalues(1)) * scale;
    % dashed axis line
    plot3(ax, state.mean_vec(1) + [-8,8]*state.eigenvectors(1,1), state.mean_vec(2) + [-8,8]*state.eigenvectors(2,1), state.mean_vec(3) + [-8,8]*state.eigenvectors(3,1), 'Color', [state.colors.pc1, 0.3], 'LineWidth', 1, 'LineStyle', '--');
    % arrow
    quiver3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), dir1(1), dir1(2), dir1(3), 'Color', state.colors.pc1, 'LineWidth', 5, 'MaxHeadSize', 0.8, 'AutoScale', 'off');

    % spread indicator - thick line showing data spread along PC1
    spread = sqrt(state.eigenvalues(1));
    plot3(ax, state.mean_vec(1) + [-spread, spread]*state.eigenvectors(1,1), ...
          state.mean_vec(2) + [-spread, spread]*state.eigenvectors(2,1), ...
          state.mean_vec(3) + [-spread, spread]*state.eigenvectors(3,1), ...
          'Color', state.colors.pc1, 'LineWidth', 8, 'LineStyle', '-');

    ep1 = state.mean_vec + dir1 * 1.1;
    text(ax, ep1(1), ep1(2), ep1(3), sprintf('PC1 %.1f%%', state.var_exp(1)), 'Color', state.colors.pc1, 'FontSize', 9, 'FontWeight', 'bold');
end

if state.pca_step >= 3
    dir2 = state.eigenvectors(:,2)' * sqrt(state.eigenvalues(2)) * scale;
    plot3(ax, state.mean_vec(1) + [-6,6]*state.eigenvectors(1,2), state.mean_vec(2) + [-6,6]*state.eigenvectors(2,2), state.mean_vec(3) + [-6,6]*state.eigenvectors(3,2), 'Color', [state.colors.pc2, 0.3], 'LineWidth', 1, 'LineStyle', '--');
    quiver3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), dir2(1), dir2(2), dir2(3), 'Color', state.colors.pc2, 'LineWidth', 4, 'MaxHeadSize', 0.8, 'AutoScale', 'off');

    % spread indicator for PC2
    spread2 = sqrt(state.eigenvalues(2));
    plot3(ax, state.mean_vec(1) + [-spread2, spread2]*state.eigenvectors(1,2), ...
          state.mean_vec(2) + [-spread2, spread2]*state.eigenvectors(2,2), ...
          state.mean_vec(3) + [-spread2, spread2]*state.eigenvectors(3,2), ...
          'Color', state.colors.pc2, 'LineWidth', 6, 'LineStyle', '-');

    ep2 = state.mean_vec + dir2 * 1.1;
    text(ax, ep2(1), ep2(2), ep2(3), sprintf('PC2 %.1f%%', state.var_exp(2)), 'Color', state.colors.pc2, 'FontSize', 9, 'FontWeight', 'bold');
end

if state.pca_step >= 4
    dir3 = state.eigenvectors(:,3)' * sqrt(state.eigenvalues(3)) * scale;
    plot3(ax, state.mean_vec(1) + [-4,4]*state.eigenvectors(1,3), state.mean_vec(2) + [-4,4]*state.eigenvectors(2,3), state.mean_vec(3) + [-4,4]*state.eigenvectors(3,3), 'Color', [state.colors.pc3, 0.3], 'LineWidth', 1, 'LineStyle', '--');
    quiver3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), dir3(1), dir3(2), dir3(3), 'Color', state.colors.pc3, 'LineWidth', 3, 'MaxHeadSize', 0.8, 'AutoScale', 'off');

    % spread indicator for PC3
    spread3 = sqrt(state.eigenvalues(3));
    plot3(ax, state.mean_vec(1) + [-spread3, spread3]*state.eigenvectors(1,3), ...
          state.mean_vec(2) + [-spread3, spread3]*state.eigenvectors(2,3), ...
          state.mean_vec(3) + [-spread3, spread3]*state.eigenvectors(3,3), ...
          'Color', state.colors.pc3, 'LineWidth', 4, 'LineStyle', '-');

    ep3 = state.mean_vec + dir3 * 1.1;
    text(ax, ep3(1), ep3(2), ep3(3), sprintf('PC3 %.1f%%', state.var_exp(3)), 'Color', state.colors.pc3, 'FontSize', 9, 'FontWeight', 'bold');
end

% projections - perpendicular drop from each point to PC axis
if state.show_pc_projections && state.pca_step >= 2
    n_pcs = min(state.pca_step - 1, 3);

    for i = 1:length(state.I1)
        pt = [state.I1(i), state.I2(i), state.I3(i)];

        for pc = 1:n_pcs
            pc_score = dot(pt - state.mean_vec, state.eigenvectors(:,pc)');
            proj = state.mean_vec + pc_score * state.eigenvectors(:,pc)';

            pc_colors = [state.colors.pc1; state.colors.pc2; state.colors.pc3];
            plot3(ax, [pt(1), proj(1)], [pt(2), proj(2)], [pt(3), proj(3)], ...
                  'Color', [pc_colors(pc,:), 0.6], 'LineWidth', 1.5);
        end
    end
end

% selected point
if state.selected_point_idx > 0 && state.selected_point_idx <= length(state.I1)
    draw_3d_highlight(ax, state, state.selected_point_idx);
end

titles = {'PCA: Start', 'PCA: Ellipsoid', 'PCA: +PC1', 'PCA: +PC2', 'PCA: +PC3', 'PCA: Complete'};
title(ax, titles{state.pca_step + 1}, 'Color', 'w', 'FontSize', 13);
xlabel(ax, 'Var 1', 'Color', state.colors.var1);
ylabel(ax, 'Var 2', 'Color', state.colors.var2);
zlabel(ax, 'Var 3', 'Color', state.colors.var3);
set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
set(ax, 'Color', [0.05, 0.05, 0.05], 'XColor', [0.6, 0.6, 0.6], 'YColor', [0.6, 0.6, 0.6], 'ZColor', [0.6, 0.6, 0.6]);
axis(ax, 'vis3d');

% rotation mode
if strcmp(state.interaction_mode, 'rotate')
    rotate3d(ax, 'on');
else
    rotate3d(ax, 'off');
end

hold(ax, 'off');
end

function animate_pca_variance(fig, old_step, new_step)
% animate when stepping through pca - shows variance bars growing
state = getappdata(fig, 'state');
ax = state.ax_main;

% store view
[az, el] = view(ax);

cla(ax, 'reset');
hold(ax, 'on');
view(ax, az, el);

draw_axis_lines(ax, state);

pause_time = 0.04;
n_frames = 30;

% which PC are we adding
pc_idx = new_step - 1;

if pc_idx >= 1 && pc_idx <= 3
    pc_color = [state.colors.pc1; state.colors.pc2; state.colors.pc3];
    pc_col = pc_color(pc_idx, :);

    % projections onto this PC
    pc_scores = ([state.I1, state.I2, state.I3] - state.mean_vec) * state.eigenvectors(:, pc_idx);

    for frame = 1:n_frames
        cla(ax);
        hold(ax, 'on');
        view(ax, az, el);

        draw_axis_lines(ax, state);

        % all points faded
        scatter3(ax, state.I1, state.I2, state.I3, state.dot_size*0.6, [0.4, 0.4, 0.4], 'filled', 'MarkerEdgeAlpha', 0.5);

        % mean
        scatter3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), 200, state.colors.mean, 'filled', 'MarkerEdgeColor', 'w', 'LineWidth', 2);

        % PC axis
        scale = 2;
        dir = state.eigenvectors(:, pc_idx)' * sqrt(state.eigenvalues(pc_idx)) * scale;
        plot3(ax, state.mean_vec(1) + [-8,8]*state.eigenvectors(1,pc_idx), ...
              state.mean_vec(2) + [-8,8]*state.eigenvectors(2,pc_idx), ...
              state.mean_vec(3) + [-8,8]*state.eigenvectors(3,pc_idx), ...
              'Color', [pc_col, 0.3], 'LineWidth', 2, 'LineStyle', '--');
        quiver3(ax, state.mean_vec(1), state.mean_vec(2), state.mean_vec(3), ...
                dir(1), dir(2), dir(3), 'Color', pc_col, 'LineWidth', 6, 'MaxHeadSize', 0.8, 'AutoScale', 'off');

        % animate spread along PC
        alpha = frame / n_frames;

        % show projections gradually
        n_show = floor(alpha * length(state.I1));
        for i = 1:n_show
            pt_orig = [state.I1(i), state.I2(i), state.I3(i)];
            pt_proj = state.mean_vec + pc_scores(i) * state.eigenvectors(:, pc_idx)';

            % projection line
            plot3(ax, [pt_orig(1), pt_proj(1)], [pt_orig(2), pt_proj(2)], [pt_orig(3), pt_proj(3)], ...
                  'Color', [pc_col, 0.4], 'LineWidth', 1);

            % projected point on axis
            scatter3(ax, pt_proj(1), pt_proj(2), pt_proj(3), 40, pc_col, 'filled', 'MarkerEdgeAlpha', 0.8);
        end

        % spread indicator - thick line showing variance
        if alpha > 0.5
            spread = sqrt(state.eigenvalues(pc_idx));
            plot3(ax, state.mean_vec(1) + [-spread, spread]*state.eigenvectors(1,pc_idx), ...
                  state.mean_vec(2) + [-spread, spread]*state.eigenvectors(2,pc_idx), ...
                  state.mean_vec(3) + [-spread, spread]*state.eigenvectors(3,pc_idx), ...
                  'Color', pc_col, 'LineWidth', 8*(alpha-0.5)*2, 'LineStyle', '-');
        end

        % variance label
        text(ax, 0, 0, 9, sprintf('PC%d: %.1f%% of variance\n(data spread in this direction)', pc_idx, state.var_exp(pc_idx)), ...
             'Color', pc_col, 'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
             'BackgroundColor', [0.1, 0.1, 0.1, 0.9]);

        set(ax, 'XLim', [-10, 10], 'YLim', [-10, 10], 'ZLim', [-10, 10]);
        set(ax, 'Color', [0.05, 0.05, 0.05]);
        axis(ax, 'vis3d');
        grid(ax, 'on');
        xlabel(ax, 'Var 1', 'Color', state.colors.var1);
        ylabel(ax, 'Var 2', 'Color', state.colors.var2);
        zlabel(ax, 'Var 3', 'Color', state.colors.var3);

        drawnow;
        pause(pause_time);
    end

    pause(0.8);
end

% back to normal view, keep angle
update_display(fig);
view(state.ax_main, az, el);
drawnow;
end

%% helpers
function draw_cube_frame(ax, state)
% cube edges
c = [0.5, 0.5, 0.5]; lw = 1.2;

% bottom
plot3(ax, [-10, 10], [-10, -10], [-10, -10], 'Color', c, 'LineWidth', lw);
plot3(ax, [-10, -10], [-10, 10], [-10, -10], 'Color', c, 'LineWidth', lw);
plot3(ax, [10, 10], [-10, 10], [-10, -10], 'Color', c, 'LineWidth', lw);
plot3(ax, [-10, 10], [10, 10], [-10, -10], 'Color', c, 'LineWidth', lw);

% top
plot3(ax, [-10, 10], [-10, -10], [10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [-10, -10], [-10, 10], [10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [10, 10], [-10, 10], [10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [-10, 10], [10, 10], [10, 10], 'Color', c, 'LineWidth', lw);

% vertical edges
plot3(ax, [-10, -10], [-10, -10], [-10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [10, 10], [-10, -10], [-10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [-10, -10], [10, 10], [-10, 10], 'Color', c, 'LineWidth', lw);
plot3(ax, [10, 10], [10, 10], [-10, 10], 'Color', c, 'LineWidth', lw);

% tick labels
for v = [-10, -5, 5, 10]
    text(ax, v, -10.8, -10.8, sprintf('%d', v), 'Color', [0.6, 0.6, 0.6], 'FontSize', 8, 'HorizontalAlignment', 'center');
    text(ax, -10.8, v, -10.8, sprintf('%d', v), 'Color', [0.6, 0.6, 0.6], 'FontSize', 8, 'HorizontalAlignment', 'center');
    text(ax, -10.8, -10.8, v, sprintf('%d', v), 'Color', [0.6, 0.6, 0.6], 'FontSize', 8, 'HorizontalAlignment', 'center');
end
text(ax, 0, -10.8, -10.8, '0', 'Color', [1, 1, 1], 'FontSize', 9, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
end

function draw_axis_lines(ax, state)
% axis lines thru origin - always visible
if state.color_axes_by_var
    plot3(ax, [-10, 10], [0, 0], [0, 0], 'Color', state.colors.var1, 'LineWidth', 2.5);
    plot3(ax, [0, 0], [-10, 10], [0, 0], 'Color', state.colors.var2, 'LineWidth', 2.5);
    plot3(ax, [0, 0], [0, 0], [-10, 10], 'Color', state.colors.var3, 'LineWidth', 2.5);
else
    plot3(ax, [-10, 10], [0, 0], [0, 0], 'Color', [1, 1, 1], 'LineWidth', 2);
    plot3(ax, [0, 0], [-10, 10], [0, 0], 'Color', [1, 1, 1], 'LineWidth', 2);
    plot3(ax, [0, 0], [0, 0], [-10, 10], 'Color', [1, 1, 1], 'LineWidth', 2);
end

% origin dot
scatter3(ax, 0, 0, 0, 80, 'w', 'filled', 'MarkerEdgeColor', 'k');
end

function draw_grid(ax)
c = [0.4, 0.4, 0.4]; lw = 0.5;

% horizontal grids at multiple Z levels
for z = -10:5:10
    for v = -10:5:10
        plot3(ax, [-10,10], [v,v], [z,z], 'Color', c, 'LineWidth', lw);
        plot3(ax, [v,v], [-10,10], [z,z], 'Color', c, 'LineWidth', lw);
    end
end

% vertical grids
for y = -10:5:10
    for v = -10:5:10
        plot3(ax, [v,v], [y,y], [-10,10], 'Color', c, 'LineWidth', lw);
    end
end

for x = -10:5:10
    for v = -10:5:10
        plot3(ax, [x,x], [v,v], [-10,10], 'Color', c, 'LineWidth', lw);
    end
end
end

function plot_ellipse(ax, center, cov_matrix, color)
% 2D ellipse from covariance matrix
if all(diag(cov_matrix) < 0.001), return; end
[V, D] = eig(cov_matrix);
D(D < 0.001) = 0.001;
theta = linspace(0, 2*pi, 100);
ellipse = 2 * V * sqrt(D) * [cos(theta); sin(theta)];
plot(ax, ellipse(1,:) + center(1), ellipse(2,:) + center(2), 'Color', color, 'LineWidth', 2.5);
end

function draw_ellipsoid(ax, center, cov_matrix, color, alpha)
% 3D ellipsoid from covariance - shows shape of data cloud
if all(diag(cov_matrix) < 0.001), return; end

[V, D] = eig(cov_matrix);
D(D < 0.001) = 0.001;

% higher res sphere
[X, Y, Z] = sphere(30);
sphere_points = [X(:)'; Y(:)'; Z(:)'];
ellipsoid_points = 2 * V * sqrt(D) * sphere_points;

X_ell = reshape(ellipsoid_points(1,:) + center(1), size(X));
Y_ell = reshape(ellipsoid_points(2,:) + center(2), size(Y));
Z_ell = reshape(ellipsoid_points(3,:) + center(3), size(Z));

% use mesh to show grid structure - reveals shape better
mesh(ax, X_ell, Y_ell, Z_ell, ...
     'FaceColor', color, ...
     'FaceAlpha', alpha, ...
     'EdgeColor', [color * 0.7], ...
     'EdgeAlpha', 0.6, ...
     'LineWidth', 0.8, ...
     'PickableParts', 'none', ...
     'HitTest', 'off');
end

%% help window
function open_help_window(main_fig)
% help window with dark theme

% check if already exists
existing = findobj('Type', 'figure', 'Tag', 'pca_help_window');
if ~isempty(existing)
    figure(existing);
    return;
end

help_fig = figure('Name', 'PCA Builder Help', 'Tag', 'pca_help_window', ...
    'NumberTitle', 'off', 'MenuBar', 'none', 'ToolBar', 'none', ...
    'Color', [0.1, 0.1, 0.1], 'Position', [200, 100, 600, 500]);

tab_group = uitabgroup(help_fig, 'Units', 'normalized', 'Position', [0, 0, 1, 1]);

tab_bg = [0.12, 0.12, 0.12];
text_color = [0.9, 0.9, 0.9];

%% tab 1: welcome
tab0 = uitab(tab_group, 'Title', 'Welcome', 'BackgroundColor', tab_bg);
welcome_text = sprintf([...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n' ...
    '       Welcome to PCA Builder & Visualization!\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'Hi! I''m Sacha Bechara, a student who learned PCA in my\n' ...
    'second year of university while doing independent research,\n' ...
    'and then relearned it again in my third year.\n\n' ...
    'WHY THIS TOOL EXISTS\n\n' ...
    'I think a lot of the confusion surrounding PCA has to do\n' ...
    'with individuals struggling with - or lacking - a geometric\n' ...
    'visual understanding of variance, standard deviation, mean,\n' ...
    'and vectors.\n\n' ...
    'While I won''t explain linear algebra from scratch in this\n' ...
    'tool, I hope to make it visually intuitive for as many\n' ...
    'people as possible and allow you to manipulate graphs in\n' ...
    'multiple dimensions.\n\n' ...
    'PCA requires thinking in multiple dimensions, which is not\n' ...
    'an easy conceptual task. Hopefully this will make it a\n' ...
    'little bit easier for you!\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'RECOMMENDED RESOURCES\n\n' ...
    'I''d strongly advise watching 3Blue1Brown''s series of videos\n' ...
    'on YouTube about linear algebra for an intuitive\n' ...
    'understanding of:\n\n' ...
    '  * Vectors\n' ...
    '  * Matrices and matrix operations\n' ...
    '  * Determinants\n' ...
    '  * Eigenvectors and eigenvalues\n' ...
    '  * Introduction to computational and numerical aspects\n' ...
    '    of linear algebra\n\n' ...
    'His videos provide both the geometric AND intuitive\n' ...
    'understanding beyond just the computational side.\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'HOW TO USE THIS TOOL\n\n' ...
    'Start simple:\n' ...
    '  1. Begin with Stage "3x1D" to see individual variables\n' ...
    '  2. Move to "2D" to see how correlation affects shape\n' ...
    '  3. Explore "3D" to see the full data cloud\n' ...
    '  4. Try "PCA" to see how principal components work\n\n' ...
    'Use ROTATE mode to spin the view, then switch to SELECT\n' ...
    'mode to click on points and see animations.\n\n' ...
    'Try the "Try This" tab for preset scenarios that\n' ...
    'demonstrate key concepts!\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'I hope this tool will be useful for you!\n\n' ...
    '                                      - Sacha Bechara\n']);
create_scrollable_text(tab0, welcome_text, text_color, tab_bg);

%% tab 2: concepts
tab1 = uitab(tab_group, 'Title', 'Concepts', 'BackgroundColor', tab_bg);
concepts_text = sprintf([...
    'STATISTICAL CONCEPTS\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'MEAN (μ)\n' ...
    'The average value of a variable.\n' ...
    'Formula: μ = (1/n) × Σxᵢ\n\n' ...
    'STANDARD DEVIATION (σ)\n' ...
    'Measures spread of data around the mean.\n' ...
    'Formula: σ = √[(1/n) × Σ(xᵢ - μ)²]\n\n' ...
    'VARIANCE (σ²)\n' ...
    'Square of standard deviation.\n' ...
    'Formula: σ² = (1/n) × Σ(xᵢ - μ)²\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'CORRELATION (r)\n' ...
    'Measures linear relationship between two variables.\n' ...
    'Range: -1 to +1\n' ...
    'Formula: r = Σ[(xᵢ-μₓ)(yᵢ-μᵧ)] / (n×σₓ×σᵧ)\n\n' ...
    '• r = +1: Perfect positive correlation\n' ...
    '• r = 0: No linear correlation\n' ...
    '• r = -1: Perfect negative correlation\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'COVARIANCE MATRIX\n' ...
    'Symmetric matrix showing variances (diagonal)\n' ...
    'and covariances (off-diagonal).\n\n' ...
    'For 3 variables:\n' ...
    '┌                         ┐\n' ...
    '│ Var(X)   Cov(X,Y)  Cov(X,Z) │\n' ...
    '│ Cov(Y,X) Var(Y)   Cov(Y,Z) │\n' ...
    '│ Cov(Z,X) Cov(Z,Y) Var(Z)   │\n' ...
    '└                         ┘\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'PRINCIPAL COMPONENT ANALYSIS (PCA)\n\n' ...
    'PCA finds new axes (principal components) that:\n' ...
    '1. Are orthogonal (perpendicular) to each other\n' ...
    '2. Maximize variance along each axis\n' ...
    '3. PC1 captures most variance, PC2 second most, etc.\n\n' ...
    'EIGENVALUES (λ)\n' ...
    'Variance explained by each principal component.\n' ...
    'Larger eigenvalue = more important PC.\n\n' ...
    'EIGENVECTORS\n' ...
    'Direction of each principal component.\n' ...
    'Shown as colored arrows in 3D view.\n']);
create_scrollable_text(tab1, concepts_text, text_color, tab_bg);

%% tab 3: controls
tab2 = uitab(tab_group, 'Title', 'Controls', 'BackgroundColor', tab_bg);
controls_text = sprintf([...
    'UI CONTROLS GUIDE\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'STAGE BUTTONS\n\n' ...
    '• 3x1D: View each variable independently on\n' ...
    '        separate 1D axes (histograms)\n' ...
    '• 2D: View pairwise scatter plots\n' ...
    '      (choose which pair with 1-2, 1-3, 2-3)\n' ...
    '• 3D: Full 3D visualization with all three\n' ...
    '      variables plotted together\n' ...
    '• PCA: Principal component analysis view\n' ...
    '       with PC axes and variance info\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'VARIABLE SLIDERS (Var 1, Var 2, Var 3)\n\n' ...
    '• Mean (μ): Shifts the center of the distribution\n' ...
    '  Range: -10 to +10\n\n' ...
    '• SD (σ): Controls spread/width of distribution\n' ...
    '  Range: 0 to 5\n' ...
    '  Small SD = tight cluster, Large SD = spread out\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'CORRELATION SLIDERS\n\n' ...
    '• r₁₂: Correlation between Var 1 and Var 2\n' ...
    '• r₁₃: Correlation between Var 1 and Var 3\n' ...
    '• r₂₃: Correlation between Var 2 and Var 3\n\n' ...
    'Range: -1 to +1\n' ...
    '(Some combinations may be invalid)\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'SAMPLE SIZE (N)\n\n' ...
    'Number of random points to generate.\n' ...
    'Range: 3 to 200\n' ...
    'More points = clearer patterns\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'ACTION BUTTONS\n\n' ...
    '• New Sample: Generate new random data with\n' ...
    '              current parameters\n' ...
    '• Clear Selection: Deselect any selected point\n' ...
    '• Mode Toggle: Switch between ROTATE and SELECT\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'INTERACTION MODES\n\n' ...
    '• ROTATE mode: Click and drag to rotate 3D view\n' ...
    '• SELECT mode: Click points to see coordinates,\n' ...
    '               click deviation lines for animations\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'DISPLAY TOGGLES\n\n' ...
    '• Mean: Shows/hides the mean point\n' ...
    '        (orange dot at data center)\n' ...
    '• SD: Shows/hides standard deviation\n' ...
    '      indicators on 1D plots\n' ...
    '• Devs: Shows/hides deviation lines from\n' ...
    '        each point to the mean\n' ...
    '• Comps: Shows/hides component deviations\n' ...
    '         (X, Y, Z separately in 3D)\n' ...
    '• Ellipse: Shows/hides the 2σ confidence\n' ...
    '           ellipsoid (represents data spread)\n' ...
    '• Cube: Shows/hides the reference cube\n' ...
    '        frame and grid\n' ...
    '• Grid: Shows/hides the 3D grid lines\n' ...
    '        (only visible when Cube is on)\n' ...
    '• Proj: Shows/hides projections onto PC axes\n' ...
    '        (only in PCA stage)\n' ...
    '• PC1 Col: Colors points by their PC1 scores\n' ...
    '           (red = high, blue = low)\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'PCA STEPS (in PCA stage only)\n\n' ...
    '• None: Just the 3D data cloud\n' ...
    '• Ellip: Add confidence ellipsoid showing\n' ...
    '         data shape\n' ...
    '• +PC1: Add first principal component\n' ...
    '        (captures most variance)\n' ...
    '• +PC2: Add second principal component\n' ...
    '• +PC3: Add third principal component\n' ...
    '• All: Show all components at once\n']);
create_scrollable_text(tab2, controls_text, text_color, tab_bg);

%% tab 4: navigation
tab3 = uitab(tab_group, 'Title', 'Navigation', 'BackgroundColor', tab_bg);
nav_text = sprintf([...
    '3D NAVIGATION\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'ROTATE VIEW\n\n' ...
    '• Click and drag on 3D plot to rotate\n' ...
    '• Or use: Azimuth/Elevation sliders at bottom\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'ZOOM\n\n' ...
    '• Scroll wheel: Zoom in/out\n' ...
    '• Or use zoom tools in figure toolbar\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'PAN\n\n' ...
    '• Hold Shift + drag to pan\n' ...
    '• Or use pan tool in figure toolbar\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'VIEW ANGLES\n\n' ...
    'Azimuth: Horizontal rotation (0-360°)\n' ...
    'Elevation: Vertical angle (-90 to +90°)\n\n' ...
    'Useful views:\n' ...
    '• Az=45, El=30: Default 3D view\n' ...
    '• Az=0, El=0: Side view (X-Z plane)\n' ...
    '• Az=0, El=90: Top view (X-Y plane)\n' ...
    '• Az=90, El=0: Front view (Y-Z plane)\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    '2D PROJECTIONS\n\n' ...
    'The three small plots show 2D views:\n' ...
    '• Top-left: Var1 vs Var2 (X-Y plane)\n' ...
    '• Top-right: Var1 vs Var3 (X-Z plane)\n' ...
    '• Bottom: Var2 vs Var3 (Y-Z plane)\n\n' ...
    'These help visualize pairwise correlations.\n']);
create_scrollable_text(tab3, nav_text, text_color, tab_bg);

%% tab 5: try this
tab4 = uitab(tab_group, 'Title', 'Try This', 'BackgroundColor', tab_bg);
create_scenarios_panel(tab4, main_fig, text_color, tab_bg);

%% tab 6: citation
tab5 = uitab(tab_group, 'Title', 'Citation', 'BackgroundColor', tab_bg);
citation_text = sprintf([...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n' ...
    '                  HOW TO CITE THIS TOOL\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'If you use this tool in your teaching, research, or\n' ...
    'publications, please cite it as:\n\n\n' ...
    'APA STYLE:\n' ...
    '──────────────────────────────────────────────────────────\n' ...
    'Bechara, S. (2024). PCA Builder & Visualization:\n' ...
    '  An Interactive Tool for Understanding Principal\n' ...
    '  Component Analysis [Computer software].\n' ...
    '  https://github.com/sachabechara/pca-visualization-tool\n\n\n' ...
    'BibTeX:\n' ...
    '──────────────────────────────────────────────────────────\n' ...
    '@software{bechara2024pca,\n' ...
    '  author = {Bechara, Sacha},\n' ...
    '  title = {PCA Builder \\& Visualization: An Interactive\n' ...
    '           Tool for Understanding Principal Component\n' ...
    '           Analysis},\n' ...
    '  year = {2024},\n' ...
    '  url = {https://github.com/sachabechara/\n' ...
    '         pca-visualization-tool}\n' ...
    '}\n\n\n' ...
    'SIMPLE TEXT:\n' ...
    '──────────────────────────────────────────────────────────\n' ...
    'Sacha Bechara (2024). PCA Builder & Visualization.\n' ...
    'GitHub: github.com/sachabechara/pca-visualization-tool\n\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'LICENSE\n\n' ...
    'This tool is open source and free to use for educational\n' ...
    'purposes.\n\n' ...
    'If you make modifications or improvements, please consider\n' ...
    'contributing back to the project or sharing your\n' ...
    'enhancements!\n\n\n' ...
    'CONTACT & CONTRIBUTIONS\n\n' ...
    'GitHub: github.com/sachabechara/pca-visualization-tool\n' ...
    'Issues & suggestions welcome!\n\n\n' ...
    '━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n' ...
    'Thank you for using PCA Builder!\n\n' ...
    'If this tool helped you understand PCA better, consider:\n' ...
    '  * Sharing it with others who might benefit\n' ...
    '  * Starring the repository on GitHub\n' ...
    '  * Providing feedback for improvements\n']);
create_scrollable_text(tab5, citation_text, text_color, tab_bg);

end

function create_scrollable_text(parent, content, text_color, bg_color)
% scrollable text area

% try java scroll first, fallback to simple text
try
    % listbox styled as text - allows scrolling
    lines = strsplit(content, '\n');
    uicontrol('Parent', parent, 'Style', 'listbox', ...
        'String', lines, ...
        'Units', 'normalized', 'Position', [0.02, 0.02, 0.96, 0.96], ...
        'BackgroundColor', bg_color, 'ForegroundColor', text_color, ...
        'FontName', 'Consolas', 'FontSize', 10, ...
        'Enable', 'inactive', ...  % prevents selection highlight
        'Max', 2, 'Min', 0);  % enables scrolling
catch
    % fallback to simple text
    uicontrol('Parent', parent, 'Style', 'text', ...
        'String', content, ...
        'Units', 'normalized', 'Position', [0.02, 0.02, 0.96, 0.96], ...
        'BackgroundColor', bg_color, 'ForegroundColor', text_color, ...
        'FontName', 'Consolas', 'FontSize', 10, ...
        'HorizontalAlignment', 'left');
end
end

function create_scenarios_panel(parent, main_fig, text_color, bg_color)
% preset scenarios panel

scenarios = {
    'Uncorrelated Equal', 'All variables independent with equal spread', ...
        struct('mean1',0,'mean2',0,'mean3',0,'sd1',2,'sd2',2,'sd3',2,'r12',0,'r13',0,'r23',0);

    'Strong Positive All', 'All variables positively correlated', ...
        struct('mean1',0,'mean2',0,'mean3',0,'sd1',2,'sd2',2,'sd3',2,'r12',0.8,'r13',0.8,'r23',0.8);

    'One Dominant Axis', 'High variance in one direction only', ...
        struct('mean1',0,'mean2',0,'mean3',0,'sd1',4,'sd2',1,'sd3',1,'r12',0,'r13',0,'r23',0);

    'Planar Data', 'Data lies mostly in a 2D plane', ...
        struct('mean1',0,'mean2',0,'mean3',0,'sd1',3,'sd2',3,'sd3',0.3,'r12',0.5,'r13',0,'r23',0);

    'Mixed Correlations', 'Positive and negative correlations', ...
        struct('mean1',0,'mean2',0,'mean3',0,'sd1',2,'sd2',2,'sd3',2,'r12',0.7,'r13',-0.7,'r23',0);

    'Shifted Clusters', 'Non-zero means with correlations', ...
        struct('mean1',3,'mean2',-2,'mean3',1,'sd1',1.5,'sd2',2,'sd3',1,'r12',0.5,'r13',0.3,'r23',-0.4);
};

% header
uicontrol('Parent', parent, 'Style', 'text', ...
    'String', 'PRESET SCENARIOS - Click LOAD to apply', ...
    'Units', 'normalized', 'Position', [0.02, 0.92, 0.96, 0.06], ...
    'BackgroundColor', bg_color, 'ForegroundColor', [0.4, 0.8, 1], ...
    'FontSize', 12, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');

% scenario buttons
n_scenarios = size(scenarios, 1);
y_start = 0.85;
y_step = 0.14;

for i = 1:n_scenarios
    y_pos = y_start - (i-1) * y_step;
    name = scenarios{i, 1};
    desc = scenarios{i, 2};
    params = scenarios{i, 3};

    % scenario panel
    panel = uipanel('Parent', parent, 'Units', 'normalized', ...
        'Position', [0.02, y_pos - 0.12, 0.96, 0.13], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'BorderType', 'line', ...
        'HighlightColor', [0.3, 0.3, 0.3]);

    % name label
    uicontrol('Parent', panel, 'Style', 'text', ...
        'String', name, ...
        'Units', 'normalized', 'Position', [0.02, 0.55, 0.65, 0.4], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'ForegroundColor', [1, 0.9, 0.5], ...
        'FontSize', 11, 'FontWeight', 'bold', 'HorizontalAlignment', 'left');

    % description
    uicontrol('Parent', panel, 'Style', 'text', ...
        'String', desc, ...
        'Units', 'normalized', 'Position', [0.02, 0.1, 0.65, 0.45], ...
        'BackgroundColor', [0.15, 0.15, 0.15], 'ForegroundColor', text_color, ...
        'FontSize', 9, 'HorizontalAlignment', 'left');

    % load button
    uicontrol('Parent', panel, 'Style', 'pushbutton', ...
        'String', 'LOAD', ...
        'Units', 'normalized', 'Position', [0.75, 0.2, 0.22, 0.6], ...
        'BackgroundColor', [0.2, 0.5, 0.3], 'ForegroundColor', 'w', ...
        'FontSize', 10, 'FontWeight', 'bold', ...
        'Callback', @(~,~) load_scenario(main_fig, params));
end
end

function load_scenario(fig, params)
% apply preset params to main fig

if ~isvalid(fig)
    errordlg('Main window was closed. Please restart PCA Builder.', 'Error');
    return;
end

% get current state
state = getappdata(fig, 'state');
if isempty(state)
    return;
end

% apply params
fields = fieldnames(params);
for i = 1:length(fields)
    state.(fields{i}) = params.(fields{i});
end

% update sliders - same pattern as update_param
% mean sliders
slider = findobj(fig, 'Tag', 'mean1_slider');
if ~isempty(slider), set(slider, 'Value', state.mean1); end
slider = findobj(fig, 'Tag', 'mean2_slider');
if ~isempty(slider), set(slider, 'Value', state.mean2); end
slider = findobj(fig, 'Tag', 'mean3_slider');
if ~isempty(slider), set(slider, 'Value', state.mean3); end

% sd sliders
slider = findobj(fig, 'Tag', 'sd1_slider');
if ~isempty(slider), set(slider, 'Value', state.sd1); end
slider = findobj(fig, 'Tag', 'sd2_slider');
if ~isempty(slider), set(slider, 'Value', state.sd2); end
slider = findobj(fig, 'Tag', 'sd3_slider');
if ~isempty(slider), set(slider, 'Value', state.sd3); end

% correlation sliders
slider = findobj(fig, 'Tag', 'corr12_slider');
if ~isempty(slider), set(slider, 'Value', state.r12); end
slider = findobj(fig, 'Tag', 'corr13_slider');
if ~isempty(slider), set(slider, 'Value', state.r13); end
slider = findobj(fig, 'Tag', 'corr23_slider');
if ~isempty(slider), set(slider, 'Value', state.r23); end

% update value labels
% mean labels
label = findobj(fig, 'Tag', 'mean1_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.mean1)); end
label = findobj(fig, 'Tag', 'mean2_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.mean2)); end
label = findobj(fig, 'Tag', 'mean3_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.mean3)); end

% sd labels
label = findobj(fig, 'Tag', 'sd1_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.sd1)); end
label = findobj(fig, 'Tag', 'sd2_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.sd2)); end
label = findobj(fig, 'Tag', 'sd3_val');
if ~isempty(label), set(label, 'String', sprintf('%.1f', state.sd3)); end

% correlation labels
label = findobj(fig, 'Tag', 'corr12_val');
if ~isempty(label), set(label, 'String', sprintf('%.2f', state.r12)); end
label = findobj(fig, 'Tag', 'corr13_val');
if ~isempty(label), set(label, 'String', sprintf('%.2f', state.r13)); end
label = findobj(fig, 'Tag', 'corr23_val');
if ~isempty(label), set(label, 'String', sprintf('%.2f', state.r23)); end

% regen data and update - same as update_param
state = generate_data(state);
setappdata(fig, 'state', state);
update_display(fig);
end