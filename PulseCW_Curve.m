% Define the x range
x = linspace(-10, 10, 10000);

% Calculate the y values using the function
y = 1 - (1 ./ (1 + x.^2)) .* (sin(3.2826.*pi ./ 2 .* sqrt(1 + x.^2))).^2;

% Plot the function
figure;
plot(x, y);
xlabel('x');
ylabel('f(x)');
title('Plot of the function f(x) = 1 - 1/(1+x^2) * sin(10.3126/2 * sqrt(1+x^2))^2');
xlim([-10 10]); % Set the x-axis limits
grid on; % Add a grid for better visibility
%% 
Detune_Ratio = 2.6754;
Amp_Ratio = 3.2826;
(1 / (1 + Detune_Ratio^2)) * (sin(Amp_Ratio * pi / 2 * sqrt(1 + Detune_Ratio^2)))^2