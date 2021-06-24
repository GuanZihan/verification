function [bounds, times, status] = test_auto_taxi(eps, pred, dims, label, method)
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
opt_v = 9999
if method == 1
    for i = 1:2
        [bounds, times, ~] = sdr_multi(net, x_min,x_max,i,label,options);
        if norm(pred - bounds) < opt_v
            opt_v = norm(pred - bounds)
        end
    end
    if opt_v > 0.3
            status = 0
    end
    bounds = opt_v
end

if method == 2
    for i = 1:2
       [bounds, times, ~] = deepsdp_multi(net,x_min,x_max,i,label,options);
        if norm(pred - bounds) < opt_v
            opt_v = norm(pred - bounds)
        end
    end
    if opt_v > 0.3
            status = 0
    end
    bounds = opt_v
end

if method == 3
    for i = 1:2
        [bounds, times, ~] = deeplus(net,x_min,x_max,i,label,options);
       if norm(pred - bounds) < opt_v
           opt_v = norm(pred - bounds)
        end
    end
    if opt_v > 0.3
            status = 0
    end
    bounds = opt_v
end

end
