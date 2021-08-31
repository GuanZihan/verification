function a = init()
	unzip('yalmip.zip','yalmip')
	addpath(genpath([pwd filesep 'yalmip'])); % add yalmip to the path
	addpath('/homes/zg620/demo/mosek/9.2/tools/platform/linux64x86/bin'); % add mosek to the path
	addpath('/homes/zg620/demo/mosek/9.2/toolbox/r2015a');% add mosek to the path
end

