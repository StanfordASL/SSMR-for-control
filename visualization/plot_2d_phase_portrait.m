function [X, Y, X_DOT, Y_DOT] = plot_2d_phase_portrait(xy_dot_fun, x_lim, y_lim, x_n_points, y_n_points, x_label, y_label)
%PLOT_2D_PHASE_PORTRAIT Summary of this function goes here
%   Detailed explanation goes here
arguments
    xy_dot_fun
    x_lim(1,2) double
    y_lim(1,2) double
    x_n_points (1,1) double
    y_n_points (1,1) double
    x_label string
    y_label string
end

% figure;
x_lin = linspace(x_lim(1), x_lim(2), x_n_points);
y_lin = linspace(y_lim(1), y_lim(2), y_n_points);
[X,Y] = meshgrid(x_lin, y_lin);
[X_DOT, Y_DOT] = xy_dot_fun(X, Y);

quiver(X, Y, X_DOT, Y_DOT)
xlim([x_lim(1), x_lim(end)])
ylim([y_lim(1), y_lim(end)])
% axis equal;
ylabel(y_label)
xlabel(x_label)


end

