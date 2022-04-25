function [] = plot_eigenvalues_in_complex_plane(eigenvalues)
figure;
plot(real(eigenvalues),imag(eigenvalues ),'o', 'MarkerSize', 12, 'LineWidth', 2) %   Plot real and imaginary parts
%axis equal;
% xl = xlim;
xlim([min(real(eigenvalues))-.5, max([max(real(eigenvalues)) 0]) + .5]);
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
box off;
xlh = xlabel('Real');
% xlh.Position(1) = 1.5;
xlh.Position(2) = .05;
ylabel('Im', 'Interpreter','latex');
xlabel("Re", 'Interpreter','latex')
grid on;

end

