% Normalization
% input: [number of observations, number of features]
function result = normalization(input)

    [m, ~] = size(input);
    maxs = ones(m, 1) * max(double(input));
    mins = ones(m, 1) * min(double(input));
    result = (double(input) - mins) ./ (maxs - mins);

end