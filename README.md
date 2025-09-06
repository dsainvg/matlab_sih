for the augmented file

% Inputs:
%   augmentedCubes - cell array of hyperspectral cubes [HxWxB]
%   augmentedGTs   - cell array of ground truth labels [HxW]
%   varianceToKeep - scalar (percentage) variance to preserve, e.g. 97
%
% Output:
%   allCnnData - struct array with fields:
%       features   - PCA features [numFeatures x numValidPixels]
%       labels     - Labels for valid pixels [numValidPixels x 1]
%       coeff      - PCA coefficients
%       variance   - Variance retained
%       imageSize  - Original image size [H, W, B]
