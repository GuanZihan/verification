function [bounds, times, status] = test_mnist(eps, ~, dims, label, target, lower, upper, num, method)
% clc;
% clear all;
% clf;
%%
rng('default');

warning off;

sample = load('sample.mat').input;
xc_in = double(sample);
x_min = max(xc_in - eps, lower);
x_max = min(xc_in + eps, upper);


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
    [bounds, times, ~] = sdr_multi(net,x_min,x_max,label,target,options);
    if bounds > 0
        status = 0;
    end
end


if method == 2
        [bounds, times, ~] = deepsdp_multi(net,x_min,x_max,label,target,options);
        if bounds > 0
            status = 0;
        end
end

if method == 3
        [bounds, times, ~] = sdpnet(net,x_min,x_max,label,target, num, options);
        if bounds > 0
            status = 0;
        end
end
end
