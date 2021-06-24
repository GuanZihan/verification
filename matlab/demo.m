function demo(net,X_min,X_max,label,target,options)
language = options.language;
solver = options.solver;
verbose = options.verbose;

weights = net.weights;
biases = net.biases;
dims = net.dims;
dim_out = dims(end);
num_layers = length(net.dims)-2;

k = 1;
x_min{k} = X_min;
x_max{k} = X_max;

for k=1:num_layers+1
    
    y_min{k} = max(net.weights{k},0)*x_min{k}+min(net.weights{k},0)*x_max{k}+net.biases{k}(:);
    y_max{k} = min(net.weights{k},0)*x_min{k}+max(net.weights{k},0)*x_max{k}+net.biases{k}(:);
    
    if(k<=num_layers)
        x_min{k+1} = max(y_min{k},0);
        x_max{k+1} = max(y_max{k},0);
    end
end

num_hidden_layers = length(dims)-2;

size_big_matrix = 1 + sum(dims(1:end-1));

prob.bardim = [size_big_matrix];

end