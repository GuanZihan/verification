function [bounds, times, status] = test_iris(eps, ~, dims, label, method)
% clc;
% clear all;
% clf;
addpath('../../DeepSDP/');
%%
rng('default');

warning off;

% eps = 0.2;
sample = load('sample.mat').input;
% sample = cell2mat(sample);
xc_in = double(sample);
x_min = xc_in - eps;
x_max = xc_in + eps;
% Xin = rect2d(x_min,x_max);


options.language = 'yalmip';
options.solver = 'mosek';
options.verbose = false;
options.bounds = "crown";

net = nnsequential(dims,'relu');

net.weights = load('weights.mat').weights;
for i = 1:length(dims)-1
    net.weights{i} = double(net.weights{i});
end
net.weights{end} = double(net.weights{end});
net.biases = load('ias.mat').bias;
for i = 1:length(dims)-1
    net.biases{i} = double(net.biases{i});
end
net.biases{end} = double(net.biases{end});

status = 1;
if method == 1
    for target =1:3
        if target~= label
            [primal, time_primal, ~] = sdr_multi(net,x_min,x_max,label,target,options);
            bounds(target) = primal;
            times(target) = time_primal;
            if primal > 0
                status = 0;
                break;
            end
        end
    end
end


if method == 2
    for target = 1:3
        if target ~= label
            [dual, time_dual, ~] = deepsdp_multi(net,x_min,x_max,label,target,options);
            bounds(target) = dual;
            times(target) = time_dual;
            if dual > 0
                status = 0;
                break;
            end
        end
    end
end

if method == 3
    for target = 1:3
        if target ~= label
            [dual, time_dual, ~] = deeplus(net,x_min,x_max,label,target,options);
            bounds(target) = dual;
            times(target) = time_dual;
            if dual > 0
                status = 0;
                break;
            end
        end
    end
end
end
