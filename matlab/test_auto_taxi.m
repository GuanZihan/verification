function [bounds, times, status] = test_mnist(eps, pred, dims, label, method)
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

% Auto_Taxi

if method == 1
    [bounds, times, ~] = sdr_multi(net,x_min,x_max,1,label,options);
    if norm(pred - bounds) > 0.5
        status = 0
    end
end

if method == 2
    [bounds, times, ~] = deepsdp_multi(net,x_min,x_max,1,label,options);
    if norm(pred - bounds) > 0.5
        status = 0
    end
end

end
