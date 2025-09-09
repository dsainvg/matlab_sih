classdef PatchNet
    properties
        net            % The pretrained neural network
        patchSize      % Size of patches (default: 16)
        numClasses     % Number of output classes (k)
    end
    
    methods
        function obj = PatchNet(trainedNet, patchSize, numClasses)
            % Constructor
            if nargin < 2
                patchSize = 16;
            end
            obj.net = trainedNet;
            obj.patchSize = patchSize;
            obj.numClasses = numClasses;
        end
        
        function [outputMap, classProbs] = predict(obj, input_matrix)
            % Main prediction function
            % Input: input_matrix - 3D matrix [H x W x D]
            % Output: outputMap - 2D matrix [H x W] with class labels
            %         classProbs - 3D matrix [H x W x K] with class probabilities
            
            [original_height, original_width, ~] = size(input_matrix);
            
            % Extract patches using the integrated function
            patches = obj.extract_patches_3d(input_matrix, obj.patchSize);
            
            % Get predictions from neural network
            % Assuming net.predict returns [patch_size x patch_size x k x num_patches]
            patchPredictions = predict(obj.net, patches);
            
            % Reconstruct to original dimensions
            [classProbs, outputMap] = obj.reconstruct_patches_to_2d(patchPredictions, ...
                original_height, original_width, input_matrix);
        end
        
        function patches = extract_patches_3d(obj, input_matrix, patch_size)
            % EXTRACT_PATCHES_3D Extract fixed-size patches from a 3D matrix with uniform padding
            %
            % Inputs:
            % input_matrix - 3D matrix of size [H x W x D]
            % patch_size - Size of patches for height and width dimensions
            %
            % Outputs:
            % patches - 4D matrix of size [patch_size x patch_size x D x num_patches]
            
            if nargin < 3
                patch_size = obj.patchSize;
            end
            
            % Get dimensions of input matrix
            [height, width, depth] = size(input_matrix);
            
            % Calculate required total padding to make dimensions divisible by patch_size
            total_pad_height = mod(patch_size - mod(height, patch_size), patch_size);
            total_pad_width = mod(patch_size - mod(width, patch_size), patch_size);
            
            % Distribute padding uniformly on both sides
            pad_top = floor(total_pad_height / 2);
            pad_bottom = total_pad_height - pad_top;
            pad_left = floor(total_pad_width / 2);
            pad_right = total_pad_width - pad_left;
            
            % Calculate new dimensions after padding
            new_height = height + total_pad_height;
            new_width = width + total_pad_width;
            
            % Apply padding if needed
            if total_pad_height > 0 || total_pad_width > 0
                fprintf('Original size: [%d x %d x %d]\n', height, width, depth);
                fprintf('Uniform padding applied:\n');
                fprintf(' Top: %d, Bottom: %d (Total height padding: %d)\n', pad_top, pad_bottom, total_pad_height);
                fprintf(' Left: %d, Right: %d (Total width padding: %d)\n', pad_left, pad_right, total_pad_width);
                fprintf('Padded size: [%d x %d x %d]\n', new_height, new_width, depth);
                
                % Apply uniform padding using padarray with 'replicate' method
                padded_matrix = padarray(input_matrix, [pad_top, pad_left, 0], 'replicate', 'pre');
                padded_matrix = padarray(padded_matrix, [pad_bottom, pad_right, 0], 'replicate', 'post');
            else
                padded_matrix = input_matrix;
                fprintf('No padding required. Matrix size: [%d x %d x %d]\n', height, width, depth);
            end
            
            % Calculate number of patches in each dimension
            num_patches_h = new_height / patch_size;
            num_patches_w = new_width / patch_size;
            total_patches = num_patches_h * num_patches_w;
            
            % Initialize output matrix
            patches = zeros(patch_size, patch_size, depth, total_patches);
            
            % Extract patches from padded matrix
            patch_idx = 1;
            for i = 1:num_patches_h
                for j = 1:num_patches_w
                    % Calculate patch boundaries
                    row_start = (i-1) * patch_size + 1;
                    row_end = i * patch_size;
                    col_start = (j-1) * patch_size + 1;
                    col_end = j * patch_size;
                    
                    % Extract patch
                    patches(:, :, :, patch_idx) = padded_matrix(row_start:row_end, col_start:col_end, :);
                    patch_idx = patch_idx + 1;
                end
            end
        end
        
        function [classProbs, outputMap] = reconstruct_patches_to_2d(obj, patchPredictions, ...
                original_height, original_width, original_matrix)
            % Reconstruct patch predictions back to original 2D dimensions
            % 
            % Inputs:
            % patchPredictions - 4D matrix [patch_size x patch_size x k x num_patches]
            % original_height, original_width - original image dimensions
            % original_matrix - original input matrix for padding calculation
            %
            % Outputs:
            % classProbs - 3D matrix [H x W x K] with class probabilities
            % outputMap - 2D matrix [H x W] with class labels (argmax)
            
            [~, ~, depth] = size(original_matrix);
            
            % Calculate padding (same logic as extract_patches_3d)
            total_pad_height = mod(obj.patchSize - mod(original_height, obj.patchSize), obj.patchSize);
            total_pad_width = mod(obj.patchSize - mod(original_width, obj.patchSize), obj.patchSize);
            
            pad_top = floor(total_pad_height / 2);
            pad_left = floor(total_pad_width / 2);
            
            new_height = original_height + total_pad_height;
            new_width = original_width + total_pad_width;
            
            % Calculate number of patches
            num_patches_h = new_height / obj.patchSize;
            num_patches_w = new_width / obj.patchSize;
            
            % Initialize reconstructed matrix (padded dimensions)
            reconstructed = zeros(new_height, new_width, obj.numClasses);
            
            % Reconstruct patches back to full matrix
            patch_idx = 1;
            for i = 1:num_patches_h
                for j = 1:num_patches_w
                    row_start = (i-1) * obj.patchSize + 1;
                    row_end = i * obj.patchSize;
                    col_start = (j-1) * obj.patchSize + 1;
                    col_end = j * obj.patchSize;
                    
                    reconstructed(row_start:row_end, col_start:col_end, :) = ...
                        patchPredictions(:, :, :, patch_idx);
                    patch_idx = patch_idx + 1;
                end
            end
            
            % Remove padding to get back to original dimensions
            classProbs = reconstructed(pad_top+1:pad_top+original_height, ...
                                     pad_left+1:pad_left+original_width, :);
            
            % Generate 2D output map with argmax along class dimension
            [~, outputMap] = max(classProbs, [], 3);
        end
    end
end

