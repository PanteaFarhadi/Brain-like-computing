%% ===============================================================
%     THREE INDEPENDENT VO2 ENERGY-BASED NETWORKS (6×6 patterns)
%     Classes: I, J, L
%     Each network trains ONLY on its own class
%     Classification = network with MINIMUM ENERGY
% ===============================================================

clear; clc; close all;

%% ---------------------------------------------------------------
% PARAMETERS
%% ---------------------------------------------------------------
num_inputs  = 36;     % 6×6 image flattened
num_hidden  = 20;     % adjustable
num_outputs = 1;      % each network has one output
n_pulses    = 5;

E0a = 2;
E0b = 2;
E1  = 0.02;

T0  = 1000;
t0  = 1000;
t = {};

%% ---------------------------------------------------------------
% DEFINE 6×6 BASE PATTERNS
%% ---------------------------------------------------------------

I_base = [ ...
    0 0 1 0 0 0;
    0 0 1 0 0 0;
    0 0 1 0 0 0;
    0 0 1 0 0 0;
    0 0 1 0 0 0;
    0 0 1 0 0 0 ];

J_base = [ ...
    0 0 0 0 1 0;
    0 0 0 0 1 0;
    0 0 0 0 1 0;
    0 0 0 0 1 0;
    0 0 0 0 1 0;
    0 0 0 1 1 0 ];

L_base = [ ...
    1 0 0 0 0 0;
    1 0 0 0 0 0;
    1 0 0 0 0 0;
    1 0 0 0 0 0;
    1 0 0 0 0 0;
    1 1 1 1 1 1 ];

Bases = {I_base(:), J_base(:), L_base(:)};
Names = {'I','J','L'};

%% ---------------------------------------------------------------
% DATASET GENERATION (0 or 1 pixel flipped)
%% ---------------------------------------------------------------
num_samples = 80;   % per class
rng(1);

Datasets = cell(3,1);

for cls = 1:3
    base = Bases{cls};
    samples = zeros(num_inputs, num_samples);

    for n = 1:num_samples
        sample = base;

        if rand < 0.5
            idx = randi(num_inputs);
            sample(idx) = 1 - sample(idx);
        end

        samples(:,n) = sample;
    end

    Datasets{cls} = samples;
end

disp("Datasets generated for 6×6 patterns.");

%% ---------------------------------------------------------------
% TRAIN 3 SEPARATE NETWORKS
%% ---------------------------------------------------------------

fprintf("\nTraining NETWORK I...\n");
Net_I = train_single_network(Datasets{1}, num_hidden, n_pulses, E0a, E0b, E1, T0, t0);

fprintf("Training NETWORK J...\n");
Net_J = train_single_network(Datasets{2}, num_hidden, n_pulses, E0a, E0b, E1, T0, t0);

fprintf("Training NETWORK L...\n");
Net_L = train_single_network(Datasets{3}, num_hidden, n_pulses, E0a, E0b, E1, T0, t0);

fprintf("All 3 networks trained.\n");


%% ---------------------------------------------------------------
% VISUALIZE THE PRODUCT MATRICES
%% ---------------------------------------------------------------
figure('Position',[200 200 900 300]);
nets = {Net_I, Net_J, Net_L};

for i = 1:3
    subplot(1,3,i);
    imagesc(reshape(nets{i}.Product,6,6));
    colormap(jet); colorbar; axis equal tight;
    title(sprintf('Network %s Energy Map', Names{i}));
end


%% ---------------------------------------------------------------
% TEST CLASSIFICATION ON CLEAN BASE PATTERNS
%% ---------------------------------------------------------------
fprintf("\n=============== TEST CLASSIFICATION (CLEAN 6×6) ===============\n");

for cls = 1:3
    A = Bases{cls};
    E_I = compute_energy(A, Net_I);
    E_J = compute_energy(A, Net_J);
    E_L = compute_energy(A, Net_L);

    energies = [E_I, E_J, E_L];
    [~, pred] = min(energies);

    fprintf("%s → predicted %s   (E = %.3f, %.3f, %.3f)\n", ...
        Names{cls}, Names{pred}, energies(1), energies(2), energies(3));
end


fprintf("\nDone.\n");


%% ---------------------------------------------------------------
% GENERATE A NEW UNCLASSIFIED SAMPLE
% Requirements:
%  - Not in training set
%  - Not equal to any clean base
%  - Looks somewhat like J, but different from all J training samples
%% ---------------------------------------------------------------

fprintf("\n=============== GENERATE UNCLASSIFIED SAMPLE ===============\n");

% Start from J_base, but make bigger modifications than training flips
U = J_base(:);

% Modify 3–5 pixels randomly so it's untrained and distinct
num_mods = randi([3,5]);  

mods = randperm(num_inputs, num_mods);
U(mods) = 1 - U(mods);

fprintf("Unclassified sample generated (with %d random flips):\n", num_mods);
disp(reshape(U,6,6));

%% ---------------------------------------------------------------
% RUN CLASSIFICATION
%% ---------------------------------------------------------------

E_I = compute_energy(U, Net_I);
E_J = compute_energy(U, Net_J);
E_L = compute_energy(U, Net_L);

energies = [E_I, E_J, E_L];
[~, pred] = min(energies);

fprintf("\nUNCLASSIFIED SAMPLE → predicted class %s\n", Names{pred});
fprintf("Energy vector = [I: %.3f, J: %.3f, L: %.3f]\n", E_I, E_J, E_L);

%% ---------------------------------------------------------------
% VISUALIZE THE UNCLASSIFIED SAMPLE
%% ---------------------------------------------------------------

figure;
imagesc(reshape(U,6,6));
colormap(gray); axis equal tight;
title(sprintf("Unclassified Test Sample → predicted %s", Names{pred}));
colorbar;


%% ---------------------------------------------------------------
% FUNCTION: Compute energy of a network for input A
%% ---------------------------------------------------------------
function E = compute_energy(A, Net)
    E = A' * Net.Product;   % A is 36×1, Product is 36×1 → scalar
end


%% ---------------------------------------------------------------
% FUNCTION: TRAIN ONE NETWORK
%% ---------------------------------------------------------------
function Net = train_single_network(samples, num_hidden, n_pulses, E0a, E0b, E1, T0, t0)

    num_inputs = size(samples,1);

    Ea = E0a * ones(num_inputs, num_hidden);
    Eb = E0b * ones(num_hidden, 1);

    for s = 1:size(samples,2)
    A = samples(:,s);

    S = 0;                     % reset cumulative pulse-age sum

    for tk = 1:n_pulses

        % ----- accumulate pulse-age sum S -----
        % S = Σ 1/(t0 - j)     where j runs from 0 to tk-1
        S = S + 1/(t0 - (tk-1));

        % ----- compute delta -----
        delta = E1 * log(T0 * S);

        % ----- INPUT → HIDDEN updates -----
        for i = 1:num_inputs
            if A(i)==1
                Ea(i,:) = Ea(i,:) - delta;
            end
        end

        % ----- HIDDEN → OUTPUT updates -----
        Eb(:,1) = Eb(:,1) - delta;

    end
end


    Ea = max(Ea, 0.01);
    Eb = max(Eb, 0.01);

    Net.Ea = Ea;
    Net.Eb = Eb;
    Net.Product = Ea * Eb;  % 36×20 × 20×1 = 36×1
end
