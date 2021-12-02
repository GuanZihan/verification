function [bound, time,status] =  deepsdp(net,x_min,x_max,label,target,options)
solver = "MOSEK";
weights = net.weights;
biases = net.biases;
dims = net.dims;

[Y_min,Y_max,X_min,X_max,~,~] = net.interval_arithmetic(x_min,x_max);

X_min = [x_min; X_min];
X_max = [x_max; X_max];

% input dimension
dim_in = dims(1);

% the number of hidden layers
num_hidden_layers = length(dims) - 2;

constraints_index = 1; % index of constraints
t_subi = []; % total subi list
t_subj = []; % total subj list
t_subk = []; % total subk list
t_subl = []; % total subllist
t_val = []; % total val list
t_blc = []; % total blc list
t_buc = []; % total buc list

label_dim = label; % the true label dimension
target_dim = target; % the target attack dimension

% construct prob
[r, res] = mosekopt('symbcon');
prob.c = [];
% part 1: construct prob.barc
prob.bardim    = [1+sum(dims(1:end-1))];
prob.barc.subj = [];
prob.barc.subk = [];
prob.barc.subl = [];
prob.barc.val  = [];

t_weight = cell2mat(weights(end));
t_bias = cell2mat(biases(end));
W_i_j_label = t_weight(label_dim,:); % the j-th row of W_i
W_i_j_target = t_weight(target_dim,:);
prev_prev_dims_sum = sum(dims(1: (end - 2))); %
pointer = prev_prev_dims_sum;
for k = 1:(dims(end-1))
   prob.barc.subj = [prob.barc.subj, 1];
   prob.barc.subk = [prob.barc.subk, pointer + 2];
   prob.barc.subl = [prob.barc.subl, 1];
   prob.barc.val = [prob.barc.val, 0.5 * (W_i_j_target(k) - W_i_j_label(k))];
   pointer = pointer + 1;
end


prob.barc.subj = [prob.barc.subj, 1];
prob.barc.subk = [prob.barc.subk, 1];
prob.barc.subl = [prob.barc.subl, 1];
prob.barc.val = [prob.barc.val, t_bias(target_dim) - t_bias(label_dim)];

% part 2: construct prob.bara
for i = 1: num_hidden_layers + 1
    cur_layer = i; % current layer from 1 to l-1
    prev_dims = sum(dims(1:cur_layer));
    prev_prev_dims = sum(dims(1:(cur_layer-1)));
    cur_dims = dims(cur_layer + 1);
    [subi, subj, subk, subl, val, blc, buc, ret_index] = construct_constriants(constraints_index, cur_layer, dims, cell2mat(weights(cur_layer)), cell2mat(biases(cur_layer)), X_min(prev_prev_dims + 1: prev_prev_dims + dims(cur_layer)), X_max(prev_prev_dims + 1: prev_prev_dims + dims(cur_layer)));
    t_subi = [t_subi, subi];
    t_subj = [t_subj, subj];
    t_subk = [t_subk, subk];
    t_subl = [t_subl, subl];
    t_val = [t_val, val];
    t_blc = [t_blc blc];
    t_buc = [t_buc, buc];
    constraints_index = ret_index;
end

prob.a = sparse([], [], [], constraints_index - 1, 0);
prob.bara.subi = t_subi;
prob.bara.subj = t_subj;
prob.bara.subk = t_subk;
prob.bara.subl = t_subl;
prob.bara.val = t_val;
prob.blc = t_blc;
prob.buc = t_buc;


[r,res] = mosekopt('maximize info',prob);
bound = res.sol.itr.pobjval;
time = res.info.MSK_DINF_OPTIMIZER_TIME;
status = res.sol.itr.solsta;
message = ['method: sdr', '| solver: ', solver, '| bound: ', num2str(bound), '| time: ', num2str(time), '| status: ', status];
disp(message);
end
