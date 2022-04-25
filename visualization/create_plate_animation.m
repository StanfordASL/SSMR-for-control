function [fig] = create_plate_animation(tsol, thetasol, xsol, ysol, a, b, l, h, t1, video_writer, fps)
%CREATE_PLATE_ANIMATION Creates animation for the spring - plate model
%   Takes solution trajectories of x, y, theta and creates a 'movie' of a
% rectangle, changing position and orientation accordingly over time.
arguments
    tsol (1,:) double
    thetasol (1,:) double
    xsol (1,:) double
    ysol (1,:) double
    a (1,1) double
    b (1,1) double
    l (1,1) double
    h (1,1) double
    t1 (1,1) double = tsol(end)
    video_writer = []
    fps (1,1) double = 30
end

% if isempty(t1)
%     t1 = tsol(-1);
% end

if ~isempty(video_writer)
    video_writer.FrameRate = fps;
    open(video_writer);
end

rect_x = [a/2, -a/2, -a/2, a/2, a/2];
rect_y = [b/2, b/2, -b/2, -b/2, b/2];
rect_stacked_hom = vertcat(rect_x, rect_y, ones(1,5));
xymax = [l/2, h/2];

figure;
set(gcf,'Visible','off');
set(gcf,'color','w');
tiledlayout(1,1, 'Padding', 'none', 'TileSpacing', 'compact'); 
ph = plot(rect_stacked_hom(1,:), rect_stacked_hom(2,:));
axis equal;
xlim([-xymax(1), xymax(1)]);
ylim([-xymax(2), xymax(2)]);

F(length(tsol)-1) = struct('cdata',[],'colormap',[]);

t_rec = linspace(0, t1, fps*t1);

for i = progress(1:length(t_rec))
    % pick closest timestamp
    [~, k] = min(abs(t_rec(i)-tsol));
    
    %tf = double(subs(T_m, [x(t), y(t), theta(t)], [xsol(k), ysol(k), thetasol(k)]));
    tf = rotz(rad2deg(thetasol(k)));
    tf = tf(1:2,1:3);
    tf(1,3) = xsol(k);
    tf(2,3) = ysol(k);

    rect_now = tf * rect_stacked_hom;
    title(sprintf('t = %.1f', tsol(k)));
    
    ph.XData = rect_now(1,:);
    ph.YData = rect_now(2,:);
    
    drawnow;
    
    %plot(rect_now(1,:), rect_now(2,:));
    
    %hold all
    F(i) = getframe(gcf);

    if ~isempty(video_writer)
        writeVideo(video_writer, F(i))
    end
    %pause(0.01);
end

fig = figure;
movie(fig,F,1, fps);

if ~isempty(video_writer)
    close(video_writer);
end

%fig = figure;


end

