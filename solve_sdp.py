import matlab.engine

engine = matlab.engine.start_matlab()

engine.hidden_units_test(nargout=0)
